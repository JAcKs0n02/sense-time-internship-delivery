#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def expected_prompts() -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for name in ("safety_prompts.json", "business_prompts.json"):
        payload = load_object(SCRIPT_ROOT / "data" / name)
        prompts.extend(payload["prompts"])
    return prompts


def validate_config(config: dict[str, Any], model_alias: str) -> None:
    models = config.get("models")
    if not isinstance(models, dict) or model_alias not in models:
        raise ValueError(f"unknown model alias: {model_alias}")
    generation = config.get("generation_config")
    expected = {
        "do_sample": False,
        "max_new_tokens": 512,
        "repetition_penalty": 1.0,
        "seed": 42,
        "dtype": "bfloat16",
        "device": "cuda:0",
    }
    if generation != expected:
        raise ValueError("generation configuration differs from the frozen Day 20 contract")
    if config.get("system_message") is not None:
        raise ValueError("system_message must be null")


def validate_response_records(records: list[dict[str, Any]], config: dict[str, Any], model_alias: str) -> dict[str, Any]:
    validate_config(config, model_alias)
    prompts = expected_prompts()
    expected_by_id = {item["id"]: item for item in prompts}
    if len(records) != 15:
        raise ValueError("one model run must contain exactly 15 responses")
    pairs = [(item.get("model_alias"), item.get("prompt_id")) for item in records]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate model-prompt response pair")
    if {prompt_id for _, prompt_id in pairs} != set(expected_by_id):
        raise ValueError("response prompt ids do not exactly match the frozen set")
    for item in records:
        prompt_id = item.get("prompt_id")
        expected = expected_by_id[prompt_id]
        if item.get("model_alias") != model_alias:
            raise ValueError("model alias drift in response records")
        if item.get("prompt_type") != expected["prompt_type"]:
            raise ValueError(f"{prompt_id} prompt type mismatch")
        if item.get("prompt_text") != expected["text"]:
            raise ValueError(f"{prompt_id} prompt text was rewritten")
        if item.get("generation_config") != config["generation_config"]:
            raise ValueError(f"{prompt_id} generation configuration drift")
        if item.get("status") != "completed":
            raise ValueError(f"{prompt_id} is not completed")
        if not isinstance(item.get("response"), str) or not item["response"].strip():
            raise ValueError(f"{prompt_id} has an empty raw response")
        digest = item.get("rendered_prompt_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{prompt_id} has an invalid rendered prompt SHA-256")
    return {
        "valid": True,
        "model_alias": model_alias,
        "response_count": len(records),
        "prompt_ids": sorted(expected_by_id),
        "generation_config": config["generation_config"],
    }


def run_inference(model_dir: Path, model_alias: str, config: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("torch and transformers are required for inference") from error
    if not model_dir.is_dir():
        raise ValueError(f"model directory not found: {model_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal Day 20 evaluation")
    validate_config(config, model_alias)
    generation = config["generation_config"]
    torch.manual_seed(generation["seed"])
    torch.cuda.manual_seed_all(generation["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output.open("x", encoding="utf-8") as handle:
        for prompt in expected_prompts():
            messages = [{"role": "user", "content": prompt["text"]}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    do_sample=False,
                    max_new_tokens=generation["max_new_tokens"],
                    repetition_penalty=generation["repetition_penalty"],
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - started
            output_ids = generated[0, input_ids.shape[-1] :]
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            record = {
                "schema_version": "1.0",
                "prompt_id": prompt["id"],
                "prompt_type": prompt["prompt_type"],
                "prompt_text": prompt["text"],
                "model_alias": model_alias,
                "model_dir": str(model_dir.resolve()),
                "messages": messages,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "generation_config": generation,
                "response": response,
                "input_tokens": int(input_ids.shape[-1]),
                "output_tokens": int(output_ids.shape[-1]),
                "elapsed_seconds": round(elapsed, 6),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
    return records


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen Day 20 model evaluation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-alias", choices=("sft_only", "sft_dpo"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_object(args.config)
        validate_config(config, args.model_alias)
        if args.validate_only:
            report = {"valid": True, "model_alias": args.model_alias, "prompt_count": len(expected_prompts()), "generation_config": config["generation_config"], "validate_only": True}
        else:
            records = run_inference(args.model_dir, args.model_alias, config, args.output)
            report = validate_response_records(records, config, args.model_alias)
            report.update({"model_dir": str(args.model_dir.resolve()), "response_file": str(args.output.resolve()), "response_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "validate_only": False})
        write_json(args.manifest_output, report)
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
