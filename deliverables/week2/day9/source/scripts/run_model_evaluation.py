#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Day 9 questions against one local Hugging Face "
            "model with deterministic decoding."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs without importing torch or loading a model.",
    )
    return parser.parse_args()


def read_and_validate_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("question specification must be a JSON object")
    generation = spec.get("generation_config")
    questions = spec.get("questions")
    if not isinstance(generation, dict):
        raise ValueError("generation_config must be an object")
    if generation.get("do_sample") is not False:
        raise ValueError("do_sample must be false for a fair comparison")
    if not isinstance(generation.get("seed"), int):
        raise ValueError("generation_config.seed must be an integer")
    max_new_tokens = generation.get("max_new_tokens")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or not 1 <= max_new_tokens <= 2048
    ):
        raise ValueError("max_new_tokens must be an integer from 1 to 2048")
    repetition_penalty = generation.get("repetition_penalty")
    if not isinstance(repetition_penalty, (int, float)):
        raise ValueError("repetition_penalty must be numeric")
    if not isinstance(questions, list) or len(questions) != 3:
        raise ValueError("the frozen evaluation must contain exactly 3 questions")

    question_ids: list[str] = []
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"question {index} must be an object")
        question_id = question.get("id")
        messages = question.get("messages")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"question {index} has an invalid id")
        if not isinstance(messages, list) or len(messages) != 1:
            raise ValueError(
                f"{question_id} must contain exactly one user message"
            )
        message = messages[0]
        if (
            not isinstance(message, dict)
            or message.get("role") != "user"
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError(f"{question_id} has an invalid user message")
        question_ids.append(question_id)
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("question ids must be unique")
    return spec


def build_messages(
    system_message: str | None,
    question_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.extend(question_messages)
    return messages


def build_manifest(
    args: argparse.Namespace,
    spec: dict[str, Any],
    *,
    valid: bool,
) -> dict[str, Any]:
    return {
        "valid": valid,
        "evaluation_name": spec.get("evaluation_name"),
        "model_label": args.model_label,
        "model_dir": str(args.model_dir),
        "questions_file": str(args.questions.resolve()),
        "question_count": len(spec["questions"]),
        "question_ids": [item["id"] for item in spec["questions"]],
        "generation_config": spec["generation_config"],
        "device": args.device,
        "dtype": args.dtype,
        "validate_only": args.validate_only,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_inference(
    args: argparse.Namespace,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "torch and transformers are required for model inference"
        ) from error

    if not args.model_dir.is_dir():
        raise ValueError(f"model directory not found: {args.model_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal Day 9 evaluation")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    seed = spec["generation_config"]["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    generation_config = spec["generation_config"]
    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": generation_config["max_new_tokens"],
        "repetition_penalty": generation_config["repetition_penalty"],
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as handle:
        for question in spec["questions"]:
            messages = build_messages(
                spec.get("system_message"),
                question["messages"],
            )
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    **generation_kwargs,
                )
            elapsed_seconds = time.perf_counter() - started
            generated_ids = generated[0, input_ids.shape[-1] :]
            response = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            record = {
                "evaluation_name": spec.get("evaluation_name"),
                "question_id": question["id"],
                "purpose": question.get("purpose"),
                "model_label": args.model_label,
                "model_dir": str(args.model_dir.resolve()),
                "messages": messages,
                "generation_config": generation_config,
                "response": response,
                "input_tokens": int(input_ids.shape[-1]),
                "output_tokens": int(generated_ids.shape[-1]),
                "elapsed_seconds": round(elapsed_seconds, 6),
                "peak_gpu_memory_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
    return records


def main() -> int:
    args = parse_args()
    try:
        spec = read_and_validate_spec(args.questions)
        manifest = build_manifest(args, spec, valid=True)
        if args.validate_only:
            write_json(args.manifest_output, manifest)
            return 0
        records = run_inference(args, spec)
        manifest["response_count"] = len(records)
        manifest["completed_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        write_json(args.manifest_output, manifest)
    except (
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
