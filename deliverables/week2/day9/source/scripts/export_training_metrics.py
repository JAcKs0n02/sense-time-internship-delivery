#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


CSV_FIELDS = ["step", "epoch", "loss", "learning_rate", "grad_norm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export every optimizer-step loss from trainer_state.json."
    )
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--moving-average-window", type=int, default=100)
    return parser.parse_args()


def load_loss_records(path: Path) -> list[dict[str, Any]]:
    state = json.loads(path.read_text(encoding="utf-8"))
    history = state.get("log_history")
    if not isinstance(history, list):
        raise ValueError("trainer_state.json does not contain log_history")

    records: list[dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict) or "loss" not in entry:
            continue
        if "step" not in entry:
            raise ValueError("loss record is missing step")
        loss = float(entry["loss"])
        if not math.isfinite(loss):
            raise ValueError(f"non-finite loss at step {entry['step']}")
        records.append(
            {
                "step": int(entry["step"]),
                "epoch": entry.get("epoch", ""),
                "loss": loss,
                "learning_rate": entry.get("learning_rate", ""),
                "grad_norm": entry.get("grad_norm", ""),
            }
        )
    if not records:
        raise ValueError("no loss records found")
    return records


def validate_steps(records: list[dict[str, Any]], expected_steps: int) -> None:
    actual = [record["step"] for record in records]
    expected = list(range(1, expected_steps + 1))
    if actual != expected:
        raise ValueError(f"expected contiguous steps 1..{expected_steps}")


def linear_slope(values: list[float]) -> float:
    count = len(values)
    x_mean = (count + 1) / 2
    y_mean = statistics.fmean(values)
    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values, start=1)
    )
    denominator = sum(
        (index - x_mean) ** 2 for index in range(1, count + 1)
    )
    return numerator / denominator if denominator else 0.0


def moving_averages(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("moving-average window must be positive")
    effective_window = min(window, len(values))
    return [
        statistics.fmean(values[index : index + effective_window])
        for index in range(len(values) - effective_window + 1)
    ]


def build_summary(
    records: list[dict[str, Any]],
    expected_steps: int,
    moving_average_window: int,
) -> dict[str, Any]:
    losses = [float(record["loss"]) for record in records]
    decile_size = max(1, math.ceil(len(losses) * 0.1))
    effective_window = min(moving_average_window, len(losses))
    averages = moving_averages(losses, moving_average_window)
    first_decile = statistics.fmean(losses[:decile_size])
    last_decile = statistics.fmean(losses[-decile_size:])
    slope = linear_slope(losses)
    return {
        "expected_steps": expected_steps,
        "loss_record_count": len(losses),
        "step_sequence_complete": True,
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "nonfinite_loss_count": sum(
            not math.isfinite(value) for value in losses
        ),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_loss": max(losses),
        "mean_loss": statistics.fmean(losses),
        "decile_size": decile_size,
        "first_decile_mean": first_decile,
        "last_decile_mean": last_decile,
        "moving_average_window": effective_window,
        "moving_average_first": averages[0],
        "moving_average_last": averages[-1],
        "linear_loss_slope": slope,
        "overall_downward_trend": last_decile < first_decile and slope < 0,
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    try:
        records = load_loss_records(args.trainer_state)
        validate_steps(records, args.expected_steps)
        summary = build_summary(
            records,
            args.expected_steps,
            args.moving_average_window,
        )
        write_csv(args.csv_output, records)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
