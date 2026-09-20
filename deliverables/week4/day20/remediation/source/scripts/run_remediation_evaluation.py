#!/usr/bin/env python3
"""Run deterministic development inference for an SFT model or one PEFT adapter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any


GENERATION = {"do_sample": False, "max_new_tokens": 512, "repetition_penalty": 1.0, "seed": 42, "dtype": "bfloat16", "device": "cuda:0"}


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) not in {15, 35}:
        raise ValueError("prompt_file_must_have_15_or_35_rows")
    ids = [row.get("id") for row in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate_prompt_ids")
    return prompts


def run(model_path: Path, adapter_path: Path | None, prompts: list[dict[str, str]], alias: str, output: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_required")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt["text"]}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(input_ids=inputs, do_sample=False, max_new_tokens=512, repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
            answer_ids = generated[0, inputs.shape[-1]:]
            row = {
                "schema_version": "1.0", "prompt_id": prompt["id"], "prompt_type": prompt["prompt_type"], "prompt_text": prompt["text"],
                "model_alias": alias, "model_path": str(model_path), "adapter_path": str(adapter_path) if adapter_path else None,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "generation_config": GENERATION,
                "response": tokenizer.decode(answer_ids, skip_special_tokens=True).strip(), "input_tokens": int(inputs.shape[-1]), "output_tokens": int(answer_ids.shape[-1]),
                "elapsed_seconds": round(time.perf_counter() - started, 6), "completed_at_utc": datetime.now(timezone.utc).isoformat(), "status": "completed",
            }
            if not row["response"]:
                raise RuntimeError(f"empty_response:{prompt['id']}")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
    return {"valid": True, "model_alias": alias, "prompt_count": len(prompts), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "generation_config": GENERATION}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model_path, args.adapter_path, load_prompts(args.prompts), args.model_alias, args.output)
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

