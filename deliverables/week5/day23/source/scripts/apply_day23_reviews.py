#!/usr/bin/env python3
"""Join transparent Codex review annotations to immutable raw Day 23 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


STRENGTH_LABELS = {"strong", "acceptable", "weak"}
HALLUCINATION_LABELS = {"none", "minor", "severe"}
REVIEWER_TYPE = "codex_assisted_visual_review"
FULL_FIELDS = (
    "record_id",
    "execution_index",
    "image_id",
    "image_type",
    "question_type",
    "prompt_template_id",
    "prompt",
    "raw_response",
    "status",
    "error_type",
    "input_tokens",
    "output_tokens",
    "latency_seconds",
    "generation_seconds",
    "peak_gpu_memory_mib",
    "strength_label",
    "hallucination_label",
    "reviewer_reason",
    "reviewer_type",
    "model_revision",
    "image_sha256",
    "generation_config_sha256",
    "completed_at",
)
REVIEW_FIELDS = (
    "record_id",
    "image_id",
    "question_type",
    "strength_label",
    "hallucination_label",
    "reviewer_reason",
    "reviewer_type",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} must be a JSON object")
        rows.append(row)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("review annotations must be a JSON object")
    return payload


def join_reviews(
    raw_rows: list[dict[str, Any]], review_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    if review_payload.get("reviewer_type") != REVIEWER_TYPE:
        raise ValueError(f"reviewer_type must equal {REVIEWER_TYPE}")
    reviews = review_payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("reviews must be a list")
    raw_ids = [row.get("record_id") for row in raw_rows]
    review_ids = [row.get("record_id") for row in reviews if isinstance(row, dict)]
    if (
        len(raw_ids) != len(set(raw_ids))
        or len(review_ids) != len(set(review_ids))
        or set(raw_ids) != set(review_ids)
    ):
        raise ValueError("every raw record must be reviewed exactly once")
    review_by_id = {row["record_id"]: row for row in reviews}
    joined = []
    for raw in raw_rows:
        review = review_by_id[raw["record_id"]]
        if review.get("strength_label") not in STRENGTH_LABELS:
            raise ValueError(f"invalid strength label: {raw['record_id']}")
        if review.get("hallucination_label") not in HALLUCINATION_LABELS:
            raise ValueError(f"invalid hallucination label: {raw['record_id']}")
        if not str(review.get("reviewer_reason", "")).strip():
            raise ValueError(f"reviewer_reason is empty: {raw['record_id']}")
        joined.append(
            {
                **raw,
                "strength_label": review["strength_label"],
                "hallucination_label": review["hallucination_label"],
                "reviewer_reason": review["reviewer_reason"],
                "reviewer_type": REVIEWER_TYPE,
            }
        )
    return joined


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    args = parser.parse_args()
    joined = join_reviews(read_jsonl(args.raw_jsonl), read_json(args.reviews))
    write_csv(args.output_csv, joined, FULL_FIELDS)
    write_csv(args.review_csv, joined, REVIEW_FIELDS)
    print(json.dumps({"record_count": len(joined)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
