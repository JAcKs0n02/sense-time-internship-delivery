#!/usr/bin/env python3
"""Extract and validate per-step SFT loss records from LLaMA-Factory JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate per-step loss completeness and summarize its trend."
    )
    parser.add_argument("--trainer-log", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    return parser.parse_args()


def load_loss_records(path: Path) -> list[dict[str, float | int | None]]:
    records: list[dict[str, float | int | None]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload: dict[str, Any] = json.loads(line)
        if "loss" not in payload:
            continue
        raw_step = payload.get("step", payload.get("current_steps"))
        if raw_step is None:
            raise ValueError(f"loss record on line {line_number} has no step")
        loss = float(payload["loss"])
        if not math.isfinite(loss):
            raise ValueError(
                f"non-finite loss on line {line_number}: {payload['loss']!r}"
            )
        learning_rate = payload.get("learning_rate", payload.get("lr"))
        epoch = payload.get("epoch")
        records.append(
            {
                "step": int(raw_step),
                "loss": loss,
                "learning_rate": (
                    float(learning_rate) if learning_rate is not None else None
                ),
                "epoch": float(epoch) if epoch is not None else None,
            }
        )
    if not records:
        raise ValueError("no per-step loss records found")
    return records


def linear_slope(records: list[dict[str, float | int | None]]) -> float:
    xs = [float(record["step"]) for record in records]
    ys = [float(record["loss"]) for record in records]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / denominator


def build_summary(
    records: list[dict[str, float | int | None]],
    expected_steps: int,
) -> dict[str, Any]:
    steps = [int(record["step"]) for record in records]
    losses = [float(record["loss"]) for record in records]
    expected_sequence = list(range(1, expected_steps + 1))
    window_size = min(100, max(1, len(records) // 10))
    first_window_mean = sum(losses[:window_size]) / window_size
    last_window_mean = sum(losses[-window_size:]) / window_size
    slope = linear_slope(records)

    return {
        "expected_steps": expected_steps,
        "loss_record_count": len(records),
        "first_step": steps[0],
        "last_step": steps[-1],
        "step_sequence_complete": steps == expected_sequence,
        "all_losses_finite": all(math.isfinite(loss) for loss in losses),
        "minimum_loss": min(losses),
        "maximum_loss": max(losses),
        "mean_loss": sum(losses) / len(losses),
        "window_size": window_size,
        "first_window_mean": first_window_mean,
        "last_window_mean": last_window_mean,
        "last_minus_first_window": last_window_mean - first_window_mean,
        "linear_slope": slope,
        "overall_downward_trend": (
            last_window_mean < first_window_mean and slope < 0
        ),
    }


def write_csv(
    path: Path,
    records: list[dict[str, float | int | None]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("step", "loss", "learning_rate", "epoch"),
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    try:
        records = load_loss_records(args.trainer_log)
        summary = build_summary(records, args.expected_steps)
        if not summary["step_sequence_complete"]:
            raise ValueError(
                "per-step loss sequence is incomplete: "
                f"found {summary['loss_record_count']} records ending at "
                f"step {summary['last_step']}, expected 1..{args.expected_steps}"
            )
        write_csv(args.csv_output, records)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"analysis failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
