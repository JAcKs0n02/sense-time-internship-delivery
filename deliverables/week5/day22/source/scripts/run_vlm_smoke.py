#!/usr/bin/env python3
"""Run one offline Qwen2-VL image-to-text smoke test on AutoDL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def choose_dtype(torch_module: Any) -> tuple[Any, str]:
    if torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16, "bfloat16"
    return torch_module.float16, "float16"


def build_processor_size(min_pixels: int, max_pixels: int) -> dict[str, int]:
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("pixel budget must satisfy 0 < min_pixels <= max_pixels")
    # transformers 4.50 reads the Qwen2-VL resize budget from `size`; passing
    # only min_pixels/max_pixels updates attributes but leaves the effective
    # default longest edge unchanged.
    return {"shortest_edge": min_pixels, "longest_edge": max_pixels}


def validate_inputs(model_dir: Path, image_path: Path, revision: str) -> None:
    if not model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {model_dir}")
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise ValueError("model directory is missing model.safetensors.index.json")
    if not image_path.is_file():
        raise ValueError(f"smoke image does not exist: {image_path}")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("revision must be a 40-character lowercase hexadecimal commit")


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    validate_inputs(args.model_dir, args.image, args.revision)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype, dtype_name = choose_dtype(torch)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        local_files_only=True,
        size=build_processor_size(args.min_pixels, args.max_pixels),
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    memory_after_load = torch.cuda.memory_allocated()

    messages = build_messages(args.image, args.prompt)
    rendered_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "completed_at": utc_now(),
        "repository": args.repository,
        "revision": args.revision,
        "remote_model_path": str(args.model_dir),
        "local_files_only": True,
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
        },
        "image": {
            "path": str(args.image),
            "sha256": sha256(args.image),
        },
        "prompt": args.prompt,
        "raw_output": output_text,
        "generation": {
            "seed": args.seed,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "input_tokens": int(inputs.input_ids.shape[-1]),
            "output_tokens": int(generated_ids_trimmed[0].shape[-1]),
        },
        "processor": {
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "effective_size": dict(processor.image_processor.size),
            "image_grid_thw": inputs.image_grid_thw.tolist(),
            "pixel_value_rows": int(inputs.pixel_values.shape[0]),
        },
        "runtime": {
            "dtype": dtype_name,
            "device_map": getattr(model, "hf_device_map", None),
            "load_seconds": round(load_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "cuda_memory_after_load_bytes": int(memory_after_load),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="请用一句中文描述这张图片，只陈述可见内容。")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    # Qwen2-VL-7B in BF16 loads at about 15.5 GiB. Keep the Day 22 single-image
    # smoke input to 384 visual tokens so a 24 GiB GPU retains enough headroom
    # for vision activations and deterministic generation.
    parser.add_argument("--max-pixels", type=int, default=384 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_smoke(args)
        write_json(args.output, result)
        print(result["raw_output"])
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "1.0",
            "status": "FAIL",
            "failed_at": utc_now(),
            "repository": args.repository,
            "revision": args.revision,
            "remote_model_path": str(args.model_dir),
            "image_path": str(args.image),
            "local_files_only": True,
            "processor": {
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            },
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(args.output, failure)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
