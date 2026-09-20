#!/usr/bin/env python3
"""Export DPO reward metrics from a Hugging Face trainer_state.json."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


REQUIRED = ("step", "rewards/chosen", "rewards/rejected", "rewards/margins")
FIELDS = (
    "step",
    "epoch",
    "loss",
    "rewards/chosen",
    "rewards/rejected",
    "rewards/margins",
    "rewards/accuracies",
    "logps/chosen",
    "logps/rejected",
    "learning_rate",
    "grad_norm",
    "rewards/chosen_ma",
    "rewards/rejected_ma",
    "rewards/margins_ma",
)


def extract_rows(state_path: Path) -> list[dict[str, float | int | None]]:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    history = payload.get("log_history")
    if not isinstance(history, list):
        raise ValueError("trainer_state.json has no log_history list")
    rows: list[dict[str, float | int | None]] = []
    for entry in history:
        if not isinstance(entry, dict) or not all(key in entry for key in REQUIRED):
            continue
        row: dict[str, float | int | None] = {}
        for field in FIELDS:
            value = entry.get(field)
            if value is not None:
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"{field} must be finite at step {entry.get('step')}")
            row[field] = value
        rows.append(row)
    return rows


def add_moving_averages(
    rows: list[dict[str, Any]], *, window: int = 20
) -> list[dict[str, Any]]:
    if window < 1:
        raise ValueError("moving-average window must be positive")
    enriched: list[dict[str, Any]] = []
    mapping = {
        "rewards/chosen": "rewards/chosen_ma",
        "rewards/rejected": "rewards/rejected_ma",
        "rewards/margins": "rewards/margins_ma",
    }
    for index, row in enumerate(rows):
        copied = dict(row)
        start = max(0, index - window + 1)
        current = rows[start : index + 1]
        for source, destination in mapping.items():
            copied[destination] = sum(float(item[source]) for item in current) / len(current)
        enriched.append(copied)
    return enriched


def summarize(
    rows: list[dict[str, Any]], *, trend_window: int = 20
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No DPO reward metrics found")
    if trend_window < 1:
        raise ValueError("trend window must be positive")
    effective_window = min(trend_window, max(1, len(rows) // 2))
    first, last = rows[0], rows[-1]
    chosen_delta = float(last["rewards/chosen"]) - float(first["rewards/chosen"])
    rejected_delta = float(last["rewards/rejected"]) - float(first["rewards/rejected"])
    margin_delta = float(last["rewards/margins"]) - float(first["rewards/margins"])
    first_window = rows[:effective_window]
    last_window = rows[-effective_window:]

    def window_delta(key: str) -> float:
        first_mean = sum(float(row[key]) for row in first_window) / effective_window
        last_mean = sum(float(row[key]) for row in last_window) / effective_window
        return last_mean - first_mean

    chosen_window_delta = window_delta("rewards/chosen")
    rejected_window_delta = window_delta("rewards/rejected")
    margin_window_delta = window_delta("rewards/margins")
    return {
        "logged_steps": len(rows),
        "first_step": first["step"],
        "last_step": last["step"],
        "chosen_reward_delta": chosen_delta,
        "rejected_reward_delta": rejected_delta,
        "margin_delta": margin_delta,
        "trend_window": effective_window,
        "chosen_reward_window_delta": chosen_window_delta,
        "rejected_reward_window_delta": rejected_window_delta,
        "margin_window_delta": margin_window_delta,
        "teacher_trend_observed": chosen_window_delta > 0 and rejected_window_delta < 0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainer_state", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = extract_rows(args.trainer_state)
    summary = summarize(rows)
    write_csv(args.csv, add_moving_averages(rows, window=20))
    with args.summary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
