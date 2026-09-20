#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HARMLESS_PROMPT = "客户反馈导出报表缺少最近一天数据，请先给出三个澄清问题。"


def build_payload(
    *,
    model_path: Path,
    response: str,
    input_tokens: int,
    output_tokens: int,
    model_class: str,
    tokenizer_class: str,
) -> dict[str, Any]:
    if not response.strip():
        raise ValueError("smoke response must not be empty")
    return {
        "status": "PASS",
        "merged_model_load": "PASS",
        "harmless_generation": "PASS",
        "model_path": str(model_path),
        "model_class": model_class,
        "tokenizer_class": tokenizer_class,
        "local_files_only": True,
        "prompt": HARMLESS_PROMPT,
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 128,
            "repetition_penalty": 1.0,
            "seed": 42,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently load and smoke-test the Day 20 merged model.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal merged-model smoke test")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_path = args.model_path.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    messages = [{"role": "user", "content": HARMLESS_PROMPT}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=128,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output_ids = generated[0, input_ids.shape[-1] :]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    payload = build_payload(
        model_path=model_path,
        response=response,
        input_tokens=int(input_ids.shape[-1]),
        output_tokens=int(output_ids.shape[-1]),
        model_class=model.__class__.__name__,
        tokenizer_class=tokenizer.__class__.__name__,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_tokens": payload["output_tokens"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
