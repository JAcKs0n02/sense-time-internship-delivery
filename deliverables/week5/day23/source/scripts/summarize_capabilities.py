#!/usr/bin/env python3
"""Aggregate reviewed Day 23 records by question type and image type."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STRENGTH_LABELS = ("strong", "acceptable", "weak")
HALLUCINATION_LABELS = ("none", "minor", "severe")


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strengths = Counter(row["strength_label"] for row in rows)
    hallucinations = Counter(row["hallucination_label"] for row in rows)
    return {
        "record_count": len(rows),
        "strength_counts": {label: strengths[label] for label in STRENGTH_LABELS},
        "hallucination_counts": {
            label: hallucinations[label] for label in HALLUCINATION_LABELS
        },
        "severe_hallucinations": hallucinations["severe"],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_image_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[row["question_type"]].append(row)
        by_image_type[row["image_type"]].append(row)
    overall = _counts(rows)
    overall.pop("severe_hallucinations")
    return {
        "schema_version": "1.0",
        "overall": overall,
        "by_question_type": {
            key: _counts(values) for key, values in sorted(by_question.items())
        },
        "by_image_type": {
            key: _counts(values) for key, values in sorted(by_image_type.items())
        },
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(read_csv(args.reviewed_csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
