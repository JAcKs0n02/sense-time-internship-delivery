#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any


DIMENSIONS = ["accuracy", "completeness", "logic", "safety", "format"]


def build_radar_rows(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model_id": row["model_id"],
            **{dimension: float(row[dimension]) for dimension in DIMENSIONS},
        }
        for row in scores
    ]


def plot_radar(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles = [index / len(DIMENSIONS) * 2 * math.pi for index in range(len(DIMENSIONS))]
    closed_angles = angles + angles[:1]
    figure, axis = plt.subplots(figsize=(10, 8), subplot_kw={"polar": True})
    for row in rows:
        values = [row[dimension] for dimension in DIMENSIONS]
        axis.plot(closed_angles, values + values[:1], linewidth=1.4, label=row["model_id"])
    axis.set_xticks(angles)
    axis.set_xticklabels(["Accuracy", "Completeness", "Logic", "Safety", "Format"])
    axis.set_ylim(0, 5)
    axis.set_yticks([1, 2, 3, 4, 5])
    axis.set_title("Week 3 Day 14 — Two-human blind-review means")
    axis.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot human-only five-dimension radar scores.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with args.scores.open(newline="", encoding="utf-8") as handle:
            rows = build_radar_rows(list(csv.DictReader(handle)))
        if not rows:
            raise ValueError("score table is empty")
        plot_radar(rows, args.output)
    except (OSError, csv.Error, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
