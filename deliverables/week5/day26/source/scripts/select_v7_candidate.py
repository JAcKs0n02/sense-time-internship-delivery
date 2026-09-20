#!/usr/bin/env python3
"""Select one Day26 v7 adapter using development evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEV_THRESHOLDS = {
    "mean_gain_min": 0.35,
    "direct_gain_min": 0.0,
    "lora_wins_min": 12,
    "hallucination_rate_not_worse": True,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_candidate(summaries: list[dict], *, expected_dev_cases_sha256: str) -> dict:
    if not summaries:
        raise ValueError("at least one development summary is required")
    assessed = []
    for summary in summaries:
        if summary.get("split") != "dev":
            raise ValueError("candidate selection accepts development evidence only")
        if summary.get("cases_sha256") != expected_dev_cases_sha256:
            raise ValueError("development case SHA-256 does not match the frozen set")
        checks = {
            "mean_gain": summary.get("mean_gain", float("-inf"))
            >= DEV_THRESHOLDS["mean_gain_min"],
            "direct_gain": summary.get("direct_gain", float("-inf"))
            >= DEV_THRESHOLDS["direct_gain_min"],
            "lora_wins": summary.get("lora_wins", 0)
            >= DEV_THRESHOLDS["lora_wins_min"],
            "hallucination_rate": summary.get("lora_hallucination_rate", 1.0)
            <= summary.get("base_hallucination_rate", 0.0),
        }
        assessed.append({**summary, "checks": checks, "passed": all(checks.values())})

    passing = [row for row in assessed if row["passed"]]
    selected = max(
        passing,
        key=lambda row: (
            row["mean_gain"],
            row["direct_gain"],
            row["lora_wins"],
            -row["lora_hallucination_rate"],
        ),
        default=None,
    )
    result = {
        "schema_version": "day26-v7-candidate-selection-1",
        "status": "PASS" if selected else "FAIL",
        "selection_data": "development_only",
        "dev_cases_sha256": expected_dev_cases_sha256,
        "thresholds": DEV_THRESHOLDS,
        "candidates": assessed,
        "selected_candidate_id": selected.get("candidate_id") if selected else None,
        "selected_lora_scale": selected.get("lora_scale") if selected else None,
        "selected_adapter_model_sha256": (
            selected.get("adapter_model_sha256") if selected else None
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-cases", type=Path, required=True)
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary]
    result = select_candidate(summaries, expected_dev_cases_sha256=sha256(args.dev_cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
