#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


DIMENSIONS = ["accuracy", "completeness", "logic", "safety", "format"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_review_file(path: Path, *, expected_rows: int = 200) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    review_ids = [str(row.get("review_id", "")) for row in rows]
    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")
    if len(set(review_ids)) != len(review_ids) or any(not item for item in review_ids):
        errors.append("review_id values must be non-empty and unique")
    reviewer_ids = {str(row.get("reviewer_id", "")) for row in rows}
    if len(reviewer_ids) != 1 or "" in reviewer_ids:
        errors.append("reviewer_id must be one non-empty value")

    completed_rows = 0
    blank_score_rows = 0
    invalid_score_rows = 0
    missing_reason_rows = 0
    for row in rows:
        raw_scores = [str(row.get(dimension, "")).strip() for dimension in DIMENSIONS]
        if any(not score for score in raw_scores):
            blank_score_rows += 1
            continue
        try:
            scores = [float(score) for score in raw_scores]
        except ValueError:
            invalid_score_rows += 1
            continue
        if any(score < 0 or score > 5 for score in scores):
            invalid_score_rows += 1
            continue
        if not str(row.get("reason", "")).strip():
            missing_reason_rows += 1
            continue
        completed_rows += 1
    if invalid_score_rows:
        errors.append(f"found {invalid_score_rows} rows with invalid scores")
    if missing_reason_rows:
        errors.append(f"found {missing_reason_rows} scored rows without a reason")
    status = "completed" if completed_rows == expected_rows and not errors else "pending"
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "status": status,
        "expected_rows": expected_rows,
        "row_count": len(rows),
        "completed_rows": completed_rows,
        "blank_score_rows": blank_score_rows,
        "invalid_score_rows": invalid_score_rows,
        "missing_reason_rows": missing_reason_rows,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Day 14 blind-review completion.")
    parser.add_argument("--reviewer-1", type=Path, required=True)
    parser.add_argument("--reviewer-2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        reviewers = [
            inspect_review_file(args.reviewer_1, expected_rows=args.expected_rows),
            inspect_review_file(args.reviewer_2, expected_rows=args.expected_rows),
        ]
        result = {
            "schema_version": 1,
            "status": (
                "ready_for_unblinding"
                if all(item["status"] == "completed" for item in reviewers)
                else "awaiting_human_scores"
            ),
            "reviewers": reviewers,
            "aggregation_allowed": all(
                item["status"] == "completed" for item in reviewers
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, csv.Error, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
