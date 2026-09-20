#!/usr/bin/env python3
"""Validate normalized OpenCompass results and write the four-row score table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_ORDER = [
    ("base", "ceval"),
    ("base", "cmmlu"),
    ("best_sft", "ceval"),
    ("best_sft", "cmmlu"),
]


def build_score_table(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.get("model", "")), str(row.get("dataset", "")))
        if key in by_key:
            raise ValueError(f"duplicate formal result: {key}")
        if row.get("status") != "completed":
            raise ValueError(f"formal result is not completed: {key}")
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid score for {key}") from exc
        if not 0.0 <= score <= 100.0:
            raise ValueError(f"score out of range for {key}: {score}")
        normalized = dict(row)
        normalized["score"] = score
        by_key[key] = normalized

    missing = [key for key in EXPECTED_ORDER if key not in by_key]
    extra = [key for key in by_key if key not in EXPECTED_ORDER]
    if missing:
        raise ValueError(f"missing formal result: {missing}")
    if extra:
        raise ValueError(f"unexpected formal result: {extra}")

    base_scores = {dataset: by_key[("base", dataset)]["score"] for dataset in ("ceval", "cmmlu")}
    result: list[dict] = []
    for key in EXPECTED_ORDER:
        row = dict(by_key[key])
        row["delta_vs_base"] = round(row["score"] - base_scores[key[1]], 6)
        result.append(row)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload["results"] if isinstance(payload, dict) else payload
    table = build_score_table(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(table[0])
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
