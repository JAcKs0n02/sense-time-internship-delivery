#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any


DIMENSIONS = ["accuracy", "completeness", "logic", "safety", "format"]


def weighted_score(row: dict[str, Any], rubric: dict[str, Any]) -> float:
    total = sum(float(row[dimension]) * float(rubric["weights"][dimension]) for dimension in DIMENSIONS)
    return round(total, 6)


def validate_score_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for dimension in DIMENSIONS:
        try:
            score = float(row[dimension])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"missing or invalid {dimension} for {row.get('review_id')}") from error
        if not 0 <= score <= 5:
            raise ValueError(f"{dimension} outside 0..5 for {row.get('review_id')}")
        normalized[dimension] = score
    if not str(row.get("reason", "")).strip():
        raise ValueError(f"missing reason for {row.get('review_id')}")
    return normalized


def aggregate_human_scores(
    rows: list[dict[str, Any]],
    rubric: dict[str, Any],
    mapping: dict[str, Any],
) -> list[dict[str, Any]]:
    reviewers = {str(row.get("reviewer_id")) for row in rows}
    if len(reviewers) != 2 or "" in reviewers or "None" in reviewers:
        raise ValueError("final aggregation requires exactly two distinct reviewers")
    lookup = {item["review_id"]: item for item in mapping["records"]}
    expected_review_ids = set(lookup)
    by_reviewer = {reviewer: {str(row.get("review_id")) for row in rows if row.get("reviewer_id") == reviewer} for reviewer in reviewers}
    if any(ids != expected_review_ids for ids in by_reviewer.values()):
        raise ValueError("both reviewers must score every blinded response exactly once")
    if len(rows) != len(expected_review_ids) * 2:
        raise ValueError("duplicate human score rows detected")
    validated = [validate_score_row(row) for row in rows]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in validated:
        model_id = lookup[row["review_id"]]["candidate_id"]
        enriched = dict(row)
        enriched["weighted_total"] = weighted_score(row, rubric)
        by_model.setdefault(model_id, []).append(enriched)
    results: list[dict[str, Any]] = []
    for model_id in sorted(by_model):
        model_rows = by_model[model_id]
        answer_ids = {lookup[row["review_id"]]["question_id"] for row in model_rows}
        reviewer_totals = []
        for reviewer in sorted(reviewers):
            reviewer_totals.append(statistics.mean(row["weighted_total"] for row in model_rows if row["reviewer_id"] == reviewer))
        results.append(
            {
                "model_id": model_id,
                "answer_count": len(answer_ids),
                "rater_count": 2,
                **{
                    dimension: round(statistics.mean(row[dimension] for row in model_rows), 6)
                    for dimension in DIMENSIONS
                },
                "weighted_total": round(statistics.mean(row["weighted_total"] for row in model_rows), 6),
                "rater_total_disagreement": round(abs(reviewer_totals[0] - reviewer_totals[1]), 6),
            }
        )
    return results


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model_id", "answer_count", "rater_count", *DIMENSIONS, "weighted_total", "rater_total_disagreement"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate two frozen human review files.")
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--reviewer-1", type=Path, required=True)
    parser.add_argument("--reviewer-2", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
        rows = read_csv(args.reviewer_1) + read_csv(args.reviewer_2)
        write_csv(args.output, aggregate_human_scores(rows, rubric, mapping))
    except (OSError, csv.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
