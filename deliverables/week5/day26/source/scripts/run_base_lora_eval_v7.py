#!/usr/bin/env python3
"""Run hash-bound Day26 v7 Base-vs-LoRA development or final inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = ("natural_scene", "ocr", "chart_table", "ui", "formula")
EXPECTED_MODES = {"direct", "verification"}


def _load_legacy_runner():
    path = Path(__file__).with_name("run_base_lora_eval.py")
    spec = importlib.util.spec_from_file_location("day26_hash_bound_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluation runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUNNER = _load_legacy_runner()
validate_resume_rows = _RUNNER.validate_resume_rows
verify_adapter_identity = _RUNNER.verify_adapter_identity
verify_base_model_identity = _RUNNER.verify_base_model_identity
apply_lora_scale = _RUNNER.apply_lora_scale


def build_run_fingerprint(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_eval_records(records: list[dict], split: str) -> None:
    if split not in {"dev", "final"}:
        raise ValueError("split must be dev or final")
    if len(records) != 20 or len({row.get("id") for row in records}) != 20:
        raise ValueError(f"{split} comparison must contain exactly 20 unique cases")
    if Counter(row.get("category") for row in records) != Counter(
        {category: 4 for category in CATEGORIES}
    ):
        raise ValueError("each category must contribute exactly four cases")
    by_image: dict[str, set[str]] = defaultdict(set)
    for row in records:
        required = (
            "id",
            "split",
            "category",
            "mode",
            "source_image_id",
            "image_relative_path",
            "image_sha256",
            "user_instruction",
            "reference_answer",
        )
        if any(not row.get(field) for field in required):
            raise ValueError(f"case is missing required data: {row.get('id')}")
        if row["split"] != split:
            raise ValueError(f"case belongs to {row['split']}, not {split}")
        by_image[row["source_image_id"]].add(row["mode"])
    if len(by_image) != 10 or any(modes != EXPECTED_MODES for modes in by_image.values()):
        raise ValueError("each image must have exactly the direct and verification modes")


def run(args: argparse.Namespace) -> dict:
    records = json.loads(args.cases.read_text(encoding="utf-8"))
    validate_eval_records(records, args.split)
    args.config = args.generation_config
    original_validator = _RUNNER.validate_eval_records
    _RUNNER.validate_eval_records = validate_eval_records
    try:
        summary = _RUNNER.run(args)
    finally:
        _RUNNER.validate_eval_records = original_validator
    summary["protocol_version"] = "day26-v7-immutable-eval-1"
    summary["stage"] = args.split
    _RUNNER.write_json(args.summary, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-model-manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--scoring-config", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--blind-packet", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--run-ledger", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("dev", "final"), required=True)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
