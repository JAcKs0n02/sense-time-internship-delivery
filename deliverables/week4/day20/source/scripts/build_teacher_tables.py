#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


MODEL_ALIASES = ("sft_only", "sft_dpo")
DIMENSIONS = ("accuracy", "completeness", "logic", "safety", "format")


def read_responses(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def response_lookup(records: list[dict[str, Any]], prompt_type: str) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for item in records:
        if item.get("prompt_type") != prompt_type:
            continue
        key = (str(item.get("prompt_id")), str(item.get("model_alias")))
        if key in lookup:
            raise ValueError(f"duplicate response pair: {key}")
        lookup[key] = item
    return lookup


def build_safety_rows(records: list[dict[str, Any]], reviews: list[dict[str, str]]) -> list[dict[str, Any]]:
    responses = response_lookup(records, "safety")
    scores = {(row["prompt_id"], row["model_alias"]): row for row in reviews}
    expected_ids = [f"w4-eval-safety-{index:02d}" for index in range(1, 11)]
    expected = {(prompt_id, alias) for prompt_id in expected_ids for alias in MODEL_ALIASES}
    if set(responses) != expected or set(scores) != expected:
        raise ValueError("safety responses and reviews must cover exactly 10 prompts for both models")
    rows: list[dict[str, Any]] = []
    for prompt_id in expected_ids:
        sft = responses[(prompt_id, "sft_only")]
        dpo = responses[(prompt_id, "sft_dpo")]
        sft_review = scores[(prompt_id, "sft_only")]
        dpo_review = scores[(prompt_id, "sft_dpo")]
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_text": sft["prompt_text"],
                "sft_only_response": sft["response"],
                "sft_only_score": int(sft_review["score"]),
                "sft_only_reason": sft_review["reason"],
                "sft_dpo_response": dpo["response"],
                "sft_dpo_score": int(dpo_review["score"]),
                "sft_dpo_reason": dpo_review["reason"],
                "reviewer_type": "codex_review",
            }
        )
    return rows


def build_business_rows(records: list[dict[str, Any]], comparison: list[dict[str, str]]) -> list[dict[str, Any]]:
    responses = response_lookup(records, "business")
    scores = {(row["prompt_id"], row["model_alias"]): row for row in comparison}
    expected_ids = [f"w4-eval-business-{index:02d}" for index in range(1, 6)]
    expected = {(prompt_id, alias) for prompt_id in expected_ids for alias in MODEL_ALIASES}
    if set(responses) != expected or set(scores) != expected:
        raise ValueError("business responses and scores must cover exactly 5 prompts for both models")
    rows: list[dict[str, Any]] = []
    for prompt_id in expected_ids:
        sft_response = responses[(prompt_id, "sft_only")]
        dpo_response = responses[(prompt_id, "sft_dpo")]
        sft = scores[(prompt_id, "sft_only")]
        dpo = scores[(prompt_id, "sft_dpo")]
        sft_weighted = float(sft["weighted_score"])
        dpo_weighted = float(dpo["weighted_score"])
        outcome = "sft_dpo" if dpo_weighted > sft_weighted else "sft_only" if sft_weighted > dpo_weighted else "tie"
        row: dict[str, Any] = {
            "prompt_id": prompt_id,
            "prompt_text": sft_response["prompt_text"],
            "sft_only_response": sft_response["response"],
        }
        row.update({f"sft_only_{name}": float(sft[name]) for name in DIMENSIONS})
        row.update({"sft_only_weighted_score": sft_weighted, "sft_only_reason": sft["reason"], "sft_dpo_response": dpo_response["response"]})
        row.update({f"sft_dpo_{name}": float(dpo[name]) for name in DIMENSIONS})
        row.update({"sft_dpo_weighted_score": dpo_weighted, "sft_dpo_reason": dpo["reason"], "outcome": outcome, "reviewer_type": "codex_review"})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build teacher-facing Day 20 safety and business tables.")
    parser.add_argument("--responses", type=Path, nargs=2, required=True)
    parser.add_argument("--safety-reviews", type=Path, required=True)
    parser.add_argument("--business-comparison", type=Path, required=True)
    parser.add_argument("--safety-output", type=Path, required=True)
    parser.add_argument("--business-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        records = read_responses(args.responses)
        safety = build_safety_rows(records, read_csv(args.safety_reviews))
        business = build_business_rows(records, read_csv(args.business_comparison))
        write_csv(args.safety_output, safety)
        write_csv(args.business_output, business)
    except (OSError, csv.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
