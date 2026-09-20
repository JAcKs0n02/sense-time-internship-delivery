#!/usr/bin/env python3
"""Fail-closed validation for Day 23 raw and reviewed inference records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


STRENGTH_LABELS = {"strong", "acceptable", "weak"}
HALLUCINATION_LABELS = {"none", "minor", "severe"}
REVIEWER_TYPE = "codex_assisted_visual_review"


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
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_records(
    matrix_records: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    reviewed_records: list[dict[str, Any]],
    expected_config_sha256: str,
    expected_model_revision: str,
) -> dict[str, Any]:
    errors: list[str] = []
    labels: dict[str, list[str]] = {
        "record_count": [],
        "matrix_completeness": [],
        "successful_outputs": [],
        "runtime_metrics": [],
        "frozen_identity": [],
        "review_completeness": [],
        "raw_immutability": [],
    }

    def add(label: str, message: str) -> None:
        error = f"{label}: {message}"
        labels[label].append(error)
        errors.append(error)

    if not (len(matrix_records) == len(raw_records) == len(reviewed_records) == 25):
        add(
            "record_count",
            f"matrix={len(matrix_records)}, raw={len(raw_records)}, reviewed={len(reviewed_records)}",
        )
    matrix_ids = [row.get("record_id") for row in matrix_records]
    raw_ids = [row.get("record_id") for row in raw_records]
    reviewed_ids = [row.get("record_id") for row in reviewed_records]
    if len(set(matrix_ids)) != 25:
        add("matrix_completeness", "matrix record IDs are not 25 unique values")
    pairs = {(row.get("image_id"), row.get("question_type")) for row in matrix_records}
    if len(pairs) != 25:
        add("matrix_completeness", "matrix does not contain 25 unique image/question pairs")
    if raw_ids != matrix_ids or reviewed_ids != matrix_ids:
        add("matrix_completeness", "raw/reviewed order does not match the frozen matrix")

    raw_by_id = {row.get("record_id"): row for row in raw_records}
    for row in raw_records:
        record_id = row.get("record_id")
        if row.get("status") != "PASS" or not str(row.get("raw_response", "")).strip():
            add("successful_outputs", f"{record_id} is not a successful non-empty output")
        for field in ("input_tokens", "output_tokens", "latency_seconds", "peak_gpu_memory_mib"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                add("runtime_metrics", f"{record_id} invalid {field}: {value}")
        output_tokens = row.get("output_tokens")
        max_new_tokens = row.get("max_new_tokens")
        if row.get("generation_config_sha256") != expected_config_sha256:
            add("frozen_identity", f"{record_id} generation config hash mismatch")
        if row.get("model_revision") != expected_model_revision:
            add("frozen_identity", f"{record_id} model revision mismatch")
        if row.get("image_sha256") != next(
            (item.get("image_sha256") for item in matrix_records if item.get("record_id") == record_id),
            None,
        ):
            add("frozen_identity", f"{record_id} image hash mismatch")

    for row in reviewed_records:
        record_id = row.get("record_id")
        if row.get("strength_label") not in STRENGTH_LABELS:
            add("review_completeness", f"{record_id} invalid strength_label")
        if row.get("hallucination_label") not in HALLUCINATION_LABELS:
            add("review_completeness", f"{record_id} invalid hallucination_label")
        if not str(row.get("reviewer_reason", "")).strip():
            add("review_completeness", f"{record_id} reviewer_reason is empty")
        if row.get("reviewer_type") != REVIEWER_TYPE:
            add("review_completeness", f"{record_id} reviewer_type must be {REVIEWER_TYPE}")
        raw = raw_by_id.get(record_id)
        if (
            raw is not None
            and isinstance(raw.get("output_tokens"), (int, float))
            and isinstance(raw.get("max_new_tokens"), (int, float))
            and raw["output_tokens"] >= raw["max_new_tokens"]
            and row.get("strength_label") != "weak"
        ):
            add(
                "review_completeness",
                f"{record_id} token limit output must be labeled weak",
            )
        if raw is not None and row.get("raw_response") != raw.get("raw_response"):
            add("raw_immutability", f"{record_id} raw_response changed during review")

    checks = [
        {"check": label, "status": "FAIL" if values else "PASS"}
        for label, values in labels.items()
    ]
    return {
        "schema_version": "1.0",
        "status": "FAIL" if errors else "PASS",
        "summary": {
            "matrix_records": len(matrix_records),
            "raw_records": len(raw_records),
            "reviewed_records": len(reviewed_records),
            "successful_records": sum(row.get("status") == "PASS" for row in raw_records),
            "unique_image_question_pairs": len(pairs),
            "token_ceiling_records": sum(
                isinstance(row.get("output_tokens"), (int, float))
                and isinstance(row.get("max_new_tokens"), (int, float))
                and row["output_tokens"] >= row["max_new_tokens"]
                for row in raw_records
            ),
        },
        "checks": checks,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--reviewed-csv", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = read_json(args.matrix)
    config = read_json(args.generation_config)
    result = validate_records(
        matrix_records=matrix.get("records", []),
        raw_records=read_jsonl(args.raw_jsonl),
        reviewed_records=read_csv(args.reviewed_csv),
        expected_config_sha256=sha256(args.generation_config),
        expected_model_revision=config.get("revision", ""),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
