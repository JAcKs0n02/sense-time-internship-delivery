#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


def select_best_model(
    human_scores: list[dict[str, Any]],
    run_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    base_rows = [row for row in human_scores if row["model_id"] == "base"]
    sft_rows = [row for row in human_scores if row["model_id"] != "base"]
    if len(base_rows) != 1 or not sft_rows:
        raise ValueError("human scores must contain one base and at least one SFT model")
    resources = {row["run_id"]: row for row in run_summaries}
    for row in sft_rows:
        if row["model_id"] not in resources:
            raise ValueError(f"missing run summary for {row['model_id']}")

    def key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        resource = resources[row["model_id"]]
        return (
            -float(row["weighted_total"]),
            -float(row["accuracy"]),
            float(row["rater_total_disagreement"]),
            float(resource.get("adapter_size_bytes") or float("inf")),
            float(resource.get("train_runtime_seconds") or float("inf")),
        )

    ranked = sorted(sft_rows, key=key)
    winner = ranked[0]
    tie_breaker = "weighted_total"
    if len(ranked) > 1:
        runner_up = ranked[1]
        if float(winner["weighted_total"]) == float(runner_up["weighted_total"]):
            tie_breaker = "human_accuracy"
            if float(winner["accuracy"]) == float(runner_up["accuracy"]):
                tie_breaker = "rater_agreement"
                if float(winner["rater_total_disagreement"]) == float(runner_up["rater_total_disagreement"]):
                    tie_breaker = "resource_cost"
    base = base_rows[0]
    return {
        "status": "selected_from_two_human_raters",
        "model_id": winner["model_id"],
        "weighted_total": float(winner["weighted_total"]),
        "accuracy": float(winner["accuracy"]),
        "tie_breaker": tie_breaker,
        "adapter_path": resources[winner["model_id"]].get("adapter_path"),
        "automatic_metrics_used_for_ranking": False,
        "base_comparison": {
            "base_weighted_total": float(base["weighted_total"]),
            "best_sft_weighted_total": float(winner["weighted_total"]),
            "best_sft_exceeds_base": float(winner["weighted_total"]) > float(base["weighted_total"]),
        },
        "ranking": [row["model_id"] for row in ranked],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best SFT using human-only ranking keys.")
    parser.add_argument("--human-scores", type=Path, required=True)
    parser.add_argument("--run-summaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with args.human_scores.open(newline="", encoding="utf-8") as handle:
            scores = list(csv.DictReader(handle))
        payload = json.loads(args.run_summaries.read_text(encoding="utf-8"))
        summaries = payload.get("runs", payload) if isinstance(payload, dict) else payload
        selected = select_best_model(scores, summaries)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, csv.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
