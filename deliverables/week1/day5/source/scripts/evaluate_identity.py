#!/usr/bin/env python3
"""Run a fixed identity evaluation against a base model or LoRA adapter.

Heavy ML dependencies are imported only inside ``run_evaluation`` so the
validation helpers remain testable on the local Mac without installing CUDA
packages or model weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REQUIRED_PROMPT_KEYS = {"id", "messages", "expected_name", "expected_author"}


def validate_prompts(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not prompts:
        raise ValueError("prompt list must not be empty")
    ids: set[str] = set()
    for index, prompt in enumerate(prompts):
        missing = REQUIRED_PROMPT_KEYS - prompt.keys()
        if missing:
            raise ValueError(f"prompt {index} missing keys: {sorted(missing)}")
        prompt_id = prompt["id"]
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt {index} has an invalid id")
        if prompt_id in ids:
            raise ValueError(f"duplicate prompt id: {prompt_id}")
        ids.add(prompt_id)
        messages = prompt["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"prompt {prompt_id} has no messages")
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"prompt {prompt_id} has an invalid role")
            if not isinstance(message.get("content"), str) or not message["content"]:
                raise ValueError(f"prompt {prompt_id} has empty content")
        for key in ("expected_name", "expected_author"):
            if not isinstance(prompt[key], str) or not prompt[key].strip():
                raise ValueError(f"prompt {prompt_id} has an invalid {key}")
    return prompts


def _contains(text: str, term: str) -> bool:
    return term.casefold() in text.casefold()


def score_response(prompt: dict[str, Any], response: str) -> dict[str, Any]:
    conflict_terms = prompt.get("conflict_terms", [])
    name_match = _contains(response, prompt["expected_name"])
    author_match = _contains(response, prompt["expected_author"])
    matched_conflicts = [term for term in conflict_terms if _contains(response, term)]
    nonempty = bool(response.strip())
    return {
        "name_match": name_match,
        "author_match": author_match,
        "has_conflict": bool(matched_conflicts),
        "matched_conflict_terms": matched_conflicts,
        "nonempty": nonempty,
        "passed": name_match and author_match and not matched_conflicts and nonempty,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    passed = sum(bool(record["score"]["passed"]) for record in records)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
    }


def load_prompts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("prompt file must contain a JSON list")
    return validate_prompts(payload)


def build_model_load_kwargs(dtype: Any) -> dict[str, Any]:
    """Return the BF16 load plan shared by pre/post-training evaluation.

    The installed bitsandbytes 0.43.1 cannot use Accelerate's 4-bit
    ``device_map='auto'`` dispatch path. Evaluation therefore uses BF16 on the
    24 GB RTX 3090; QLoRA training remains 4-bit as required.
    """
    return {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }


def run_evaluation(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **build_model_load_kwargs(torch.bfloat16),
    )
    model.to(torch.device("cuda:0"))
    adapter_loaded = False
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        adapter_loaded = True
    model.eval()
    device = str(next(model.parameters()).device)
    dtype = str(next(model.parameters()).dtype)
    generation = {
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }

    records: list[dict[str, Any]] = []
    for prompt in prompts:
        messages = prompt["messages"]
        chat_template_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            chat_template_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated = output[0, input_tokens:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        record = {
            "id": prompt["id"],
            "label": args.label,
            "model": str(args.model),
            "adapter": str(args.adapter) if args.adapter else None,
            "adapter_loaded": adapter_loaded,
            "messages": messages,
            "chat_template_text": chat_template_text,
            "generation": generation,
            "input_tokens": input_tokens,
            "output_tokens": int(generated.shape[-1]),
            "elapsed_seconds": round(elapsed, 6),
            "device": device,
            "dtype": dtype,
            "response": response,
            "score": score_response(prompt, response),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize_records(records)
    print("IDENTITY_EVALUATION_SUMMARY=" + json.dumps(summary, ensure_ascii=False))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
