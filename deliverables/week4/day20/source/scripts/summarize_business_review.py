#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


DIMENSIONS = ("accuracy", "completeness", "logic", "safety", "format")
WEIGHTS = {"accuracy": 0.30, "completeness": 0.25, "logic": 0.20, "safety": 0.15, "format": 0.10}
EXPECTED_IDS = {f"w4-eval-business-{index:02d}" for index in range(1, 6)}
MODEL_ALIASES = {"sft_only", "sft_dpo"}


def validate_and_unblind(score_rows: list[dict[str, str]], mapping: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(score_rows) != 10 or len(mapping) != 10:
        raise ValueError("scores and mapping must each contain exactly 10 rows")
    mapping_by_key: dict[tuple[str, str], str] = {}
    for item in mapping:
        key = (item.get("prompt_id", ""), item.get("candidate_label", ""))
        alias = item.get("model_alias", "")
        if key in mapping_by_key or alias not in MODEL_ALIASES:
            raise ValueError("mapping has duplicate or invalid identities")
        mapping_by_key[key] = alias
    comparison: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in score_rows:
        key = (row.get("prompt_id", ""), row.get("candidate_label", ""))
        if key in seen or key not in mapping_by_key:
            raise ValueError("score row does not match the blind mapping")
        seen.add(key)
        if key[0] not in EXPECTED_IDS or key[1] not in ("A", "B"):
            raise ValueError("invalid prompt_id or candidate_label")
        if row.get("reviewer_type") != "codex_review":
            raise ValueError("reviewer_type must be codex_review")
        if not row.get("reason", "").strip():
            raise ValueError("every business score requires a reason")
        values: dict[str, float] = {}
        for dimension in DIMENSIONS:
            try:
                value = float(row.get(dimension, ""))
            except ValueError as error:
                raise ValueError(f"{dimension} must be numeric from 0 to 5") from error
            if not 0 <= value <= 5:
                raise ValueError(f"{dimension} must be from 0 to 5")
            values[dimension] = value
        weighted = round(sum(values[name] * WEIGHTS[name] for name in DIMENSIONS), 4)
        comparison.append({"prompt_id": key[0], "candidate_label": key[1], "model_alias": mapping_by_key[key], **values, "weighted_score": weighted, "reason": row["reason"], "reviewer_type": "codex_review"})
    if set(mapping_by_key) != seen:
        raise ValueError("score rows do not cover the complete blind mapping")
    by_model = {alias: [row["weighted_score"] for row in comparison if row["model_alias"] == alias] for alias in sorted(MODEL_ALIASES)}
    if any(len(values) != 5 for values in by_model.values()):
        raise ValueError("unblinded mapping must produce exactly five scores per model")
    means = {alias: round(statistics.mean(values), 4) for alias, values in by_model.items()}
    dimension_means = {
        alias: {
            dimension: round(
                statistics.mean(
                    row[dimension]
                    for row in comparison
                    if row["model_alias"] == alias
                ),
                4,
            )
            for dimension in DIMENSIONS
        }
        for alias in sorted(MODEL_ALIASES)
    }
    outcomes = {"sft_dpo_wins": 0, "sft_only_wins": 0, "ties": 0}
    for prompt_id in sorted(EXPECTED_IDS):
        prompt_scores = {
            row["model_alias"]: row["weighted_score"]
            for row in comparison
            if row["prompt_id"] == prompt_id
        }
        if prompt_scores["sft_dpo"] > prompt_scores["sft_only"]:
            outcomes["sft_dpo_wins"] += 1
        elif prompt_scores["sft_only"] > prompt_scores["sft_dpo"]:
            outcomes["sft_only_wins"] += 1
        else:
            outcomes["ties"] += 1
    report = {
        "schema_version": "1.0",
        "reviewer_type": "codex_review",
        "model_prompt_counts": {alias: len(values) for alias, values in by_model.items()},
        "model_weighted_means": means,
        "weighted_mean_delta_sft_dpo_minus_sft_only": round(
            means["sft_dpo"] - means["sft_only"], 4
        ),
        "dimension_means": dimension_means,
        "prompt_outcomes": outcomes,
        "interpretation": "Descriptive evidence from five frozen prompts; no statistical significance claim.",
    }
    return comparison, report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate, unblind, and summarize Day 20 business scores.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with args.scores.open(encoding="utf-8", newline="") as handle:
            scores = list(csv.DictReader(handle))
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
        comparison, report = validate_and_unblind(scores, mapping)
        report["blinded_scores_sha256"] = hashlib.sha256(args.scores.read_bytes()).hexdigest()
        write_csv(args.comparison_output, comparison)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
