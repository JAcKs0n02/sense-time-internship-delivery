#!/usr/bin/env python3
"""Run the frozen ten-case Day25 Qwen2-VL hallucination evaluation offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_IMAGE_IDS = {
    "w5-table-01",
    "w5-scene-01",
    "w5-logo-01",
    "w5-formula-01",
    "w5-ui-01",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_processor_size(min_pixels: int, max_pixels: int) -> dict[str, int]:
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("pixel budget must satisfy 0 < min_pixels <= max_pixels")
    return {"shortest_edge": min_pixels, "longest_edge": max_pixels}


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


def validate_formal_cases(records: list[dict[str, Any]]) -> None:
    if len(records) != 10:
        raise ValueError(f"formal Day25 evaluation must contain ten cases, got {len(records)}")
    expected_ids = [f"D25-{index:02d}" for index in range(1, 11)]
    if [row.get("case_id") for row in records] != expected_ids:
        raise ValueError("case IDs must remain the ordered sequence D25-01..D25-10")
    if [row.get("execution_index") for row in records] != list(range(1, 11)):
        raise ValueError("execution_index must remain the ordered sequence 1..10")
    image_counts = Counter(row.get("image_id") for row in records)
    if set(image_counts) != EXPECTED_IMAGE_IDS or set(image_counts.values()) != {2}:
        raise ValueError("the frozen matrix requires exactly two cases for each of five images")
    required = {
        "prompt",
        "question",
        "fake_answer",
        "ground_truth",
        "expected_behavior",
        "error_category",
        "image_relative_path",
        "image_sha256",
        "generation_config_sha256",
        "model_revision",
    }
    for row in records:
        missing = sorted(field for field in required if not row.get(field))
        if missing:
            raise ValueError(f"{row.get('case_id')} missing required fields: {missing}")


def validate_static_inputs(
    *,
    model_dir: Path,
    data_root: Path,
    config_path: Path,
    config: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise ValueError("model directory is incomplete")
    actual_config_sha = sha256(config_path)
    if {row["generation_config_sha256"] for row in records} != {actual_config_sha}:
        raise ValueError("case generation config hash does not match the config file")
    if {row["model_revision"] for row in records} != {config.get("revision")}:
        raise ValueError("case model revision does not match the config file")
    if config.get("do_sample") is not False or config.get("seed") != 42:
        raise ValueError("formal generation must use do_sample=false and seed=42")
    expected_size = build_processor_size(config["min_pixels"], config["max_pixels"])
    if config.get("processor_size") != expected_size:
        raise ValueError("processor_size does not match the frozen pixel budget")
    for row in records:
        image_path = data_root / row["image_relative_path"]
        if not image_path.is_file() or sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"image integrity check failed: {row['image_id']}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    cases_payload = read_json(args.cases)
    records = cases_payload.get("records")
    if not isinstance(records, list):
        raise ValueError("cases records must be a list")
    validate_formal_cases(records)
    validate_static_inputs(
        model_dir=args.model_dir,
        data_root=args.data_root,
        config_path=args.config,
        config=config,
        records=records,
    )

    selected = records
    if args.case_id:
        selected = [row for row in records if row["case_id"] == args.case_id]
        if len(selected) != 1:
            raise ValueError(f"case_id not found: {args.case_id}")

    existing = read_jsonl(args.output)
    if existing and not args.resume:
        raise ValueError(f"output already exists; use --resume to preserve first results: {args.output}")
    existing_ids = {row.get("case_id") for row in existing}
    selected = [row for row in selected if row["case_id"] not in existing_ids]

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the frozen bfloat16 configuration is unsupported by this GPU")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.empty_cache()

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        local_files_only=True,
        size=build_processor_size(config["min_pixels"], config["max_pixels"]),
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = round(time.perf_counter() - load_started, 3)
    memory_after_load_mib = round(torch.cuda.memory_allocated() / 1024 / 1024, 3)
    effective_size = dict(processor.image_processor.size)
    if effective_size != config["processor_size"]:
        raise RuntimeError(
            f"effective processor size changed: {effective_size} != {config['processor_size']}"
        )

    pass_count = 0
    fail_count = 0
    for item in selected:
        image_path = args.data_root / item["image_relative_path"]
        started = time.perf_counter()
        base = {
            **item,
            "schema_version": "1.0",
            "run_id": args.run_id,
            "repository": config["repository"],
            "model_revision": config["revision"],
            "local_files_only": True,
            "dtype": config["dtype"],
            "do_sample": config["do_sample"],
            "seed": config["seed"],
            "max_new_tokens": config["max_new_tokens"],
            "processor_size": config["processor_size"],
        }
        try:
            torch.cuda.reset_peak_memory_stats()
            messages = build_messages(image_path, item["prompt"])
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[rendered],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    do_sample=False,
                )
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_started
            trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_response = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            if not raw_response.strip():
                raise RuntimeError("model returned an empty response")
            output_token_count = int(trimmed[0].shape[-1])
            payload = {
                **base,
                "status": "PASS",
                "raw_response": raw_response,
                "input_tokens": int(inputs.input_ids.shape[-1]),
                "output_tokens": output_token_count,
                "reached_token_ceiling": output_token_count >= config["max_new_tokens"],
                "latency_seconds": round(time.perf_counter() - started, 3),
                "generation_seconds": round(generation_seconds, 3),
                "peak_gpu_memory_mib": round(
                    torch.cuda.max_memory_allocated() / 1024 / 1024, 3
                ),
                "image_grid_thw": inputs.image_grid_thw.tolist(),
                "completed_at": utc_now(),
                "error_type": "",
                "error": "",
            }
            pass_count += 1
        except Exception as exc:
            payload = {
                **base,
                "status": "FAIL",
                "raw_response": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "reached_token_ceiling": False,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "generation_seconds": 0.0,
                "peak_gpu_memory_mib": round(
                    torch.cuda.max_memory_allocated() / 1024 / 1024, 3
                ),
                "image_grid_thw": [],
                "completed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            fail_count += 1
        append_jsonl(args.output, payload)
        print(
            f"[{item['execution_index']:02d}/10] {item['case_id']} {payload['status']} "
            f"tokens={payload['output_tokens']} latency={payload['latency_seconds']}s",
            flush=True,
        )

    all_rows = read_jsonl(args.output)
    summary = {
        "schema_version": "1.0",
        "status": "PASS" if fail_count == 0 else "FAIL",
        "run_id": args.run_id,
        "completed_at": utc_now(),
        "requested_record_count": len(selected),
        "new_pass_count": pass_count,
        "new_fail_count": fail_count,
        "total_output_records": len(all_rows),
        "total_pass_records": sum(row.get("status") == "PASS" for row in all_rows),
        "model_load_seconds": load_seconds,
        "memory_after_load_mib": memory_after_load_mib,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "cases_sha256": sha256(args.cases),
        "generation_config_sha256": sha256(args.config),
    }
    write_json(args.run_summary, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
