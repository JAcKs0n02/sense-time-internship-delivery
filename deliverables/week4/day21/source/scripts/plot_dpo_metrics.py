#!/usr/bin/env python3
"""Normalize the frozen Day 19 DPO metrics and render Day 21 Rewards curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELD_MAP = {
    "step": "step",
    "epoch": "epoch",
    "loss": "loss",
    "chosen": "rewards/chosen",
    "rejected": "rewards/rejected",
    "margin": "rewards/margins",
    "accuracy": "rewards/accuracies",
    "chosen_ma20": "rewards/chosen_ma",
    "rejected_ma20": "rewards/rejected_ma",
    "margin_ma20": "rewards/margins_ma",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_normalize(path: Path) -> list[dict[str, float | int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, float | int]] = []
    for source in source_rows:
        row: dict[str, float | int] = {"step": int(source[FIELD_MAP["step"]])}
        for target, original in FIELD_MAP.items():
            if target == "step":
                continue
            value = float(source[original])
            if not math.isfinite(value):
                raise ValueError(f"{target} must be finite")
            row[target] = value
        if not math.isclose(
            float(row["margin"]),
            float(row["chosen"]) - float(row["rejected"]),
            abs_tol=1e-6,
        ):
            raise ValueError("margin must equal chosen minus rejected")
        rows.append(row)
    if not rows:
        raise ValueError("metrics must contain at least one row")
    expected_steps = list(range(1, len(rows) + 1))
    if [int(row["step"]) for row in rows] != expected_steps:
        raise ValueError(f"steps must be unique and contiguous from 1 through {len(rows)}")
    return rows


def load_validation(path: Path) -> list[dict[str, float | int]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for source in document["evaluation_history"]:
        row = {
            "step": int(source["step"]),
            "chosen": float(source["eval_rewards/chosen"]),
            "rejected": float(source["eval_rewards/rejected"]),
            "margin": float(source["eval_rewards/margins"]),
            "accuracy": float(source["eval_rewards/accuracies"]),
            "loss": float(source["eval_loss"]),
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError("validation metrics must be finite")
        rows.append(row)
    steps = [int(row["step"]) for row in rows]
    if not steps or steps != sorted(set(steps)) or steps[0] < 1:
        raise ValueError("validation steps must be positive, unique, and increasing")
    return rows


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def render_plot(
    rows: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    output: Path,
    beta: float,
    moving_average_window: int,
) -> None:
    steps = _series(rows, "step")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    total_steps = int(rows[-1]["step"])
    fig.suptitle(
        f"Week 4 DPO Rewards — {total_steps} Optimizer Steps (beta={beta}, MA={moving_average_window})",
        fontsize=16,
        fontweight="bold",
    )

    reward_axis = axes[0, 0]
    reward_axis.plot(steps, _series(rows, "chosen"), color="#2563eb", alpha=0.20, linewidth=0.8, label="Chosen raw")
    reward_axis.plot(steps, _series(rows, "rejected"), color="#dc2626", alpha=0.20, linewidth=0.8, label="Rejected raw")
    reward_axis.plot(steps, _series(rows, "chosen_ma20"), color="#1d4ed8", linewidth=2.2, label="Chosen MA20")
    reward_axis.plot(steps, _series(rows, "rejected_ma20"), color="#b91c1c", linewidth=2.2, label="Rejected MA20")
    reward_axis.axhline(0, color="#64748b", linewidth=0.8)
    reward_axis.set_title("Training Chosen vs Rejected Reward")
    reward_axis.set_ylabel("DPO reward")
    reward_axis.legend(ncol=2, fontsize=8)

    margin_axis = axes[0, 1]
    margin_axis.plot(steps, _series(rows, "margin"), color="#16a34a", alpha=0.22, linewidth=0.8, label="Margin raw")
    margin_axis.plot(steps, _series(rows, "margin_ma20"), color="#15803d", linewidth=2.3, label="Margin MA20")
    margin_axis.axhline(0, color="#334155", linestyle="--", linewidth=1, label="Zero margin")
    margin_axis.set_title("Training Reward Margin")
    margin_axis.set_ylabel("Chosen − rejected")
    margin_axis.legend(fontsize=8)

    accuracy_axis = axes[1, 0]
    accuracy_axis.plot(steps, _series(rows, "accuracy"), color="#7c3aed", alpha=0.24, linewidth=0.8, label="Accuracy raw")
    window = moving_average_window
    accuracy_values = _series(rows, "accuracy")
    accuracy_ma = [sum(accuracy_values[max(0, index - window + 1) : index + 1]) / min(index + 1, window) for index in range(len(accuracy_values))]
    accuracy_axis.plot(steps, accuracy_ma, color="#6d28d9", linewidth=2.3, label="Accuracy MA20")
    accuracy_axis.axhline(0.5, color="#64748b", linestyle="--", linewidth=1, label="0.5 reference")
    accuracy_axis.set_ylim(-0.03, 1.03)
    accuracy_axis.set_title("Training Reward Accuracy")
    accuracy_axis.set_ylabel("Fraction with chosen > rejected")
    accuracy_axis.legend(fontsize=8)

    validation_axis = axes[1, 1]
    validation_steps = _series(validation, "step")
    validation_axis.plot(validation_steps, _series(validation, "chosen"), marker="o", linewidth=2, color="#1d4ed8", label="Eval chosen")
    validation_axis.plot(validation_steps, _series(validation, "rejected"), marker="o", linewidth=2, color="#b91c1c", label="Eval rejected")
    validation_axis.plot(validation_steps, _series(validation, "margin"), marker="o", linewidth=2.4, color="#15803d", label="Eval margin")
    validation_axis.set_title("Validation Reward Checkpoints")
    validation_axis.set_ylabel("Validation reward")
    validation_accuracy_axis = validation_axis.twinx()
    validation_accuracy_axis.plot(validation_steps, _series(validation, "accuracy"), marker="s", linestyle="--", linewidth=1.7, color="#6d28d9", label="Eval accuracy")
    validation_accuracy_axis.set_ylim(-0.03, 1.03)
    validation_accuracy_axis.set_ylabel("Validation reward accuracy")
    lines, labels = validation_axis.get_legend_handles_labels()
    extra_lines, extra_labels = validation_accuracy_axis.get_legend_handles_labels()
    validation_axis.legend(lines + extra_lines, labels + extra_labels, fontsize=8, loc="center right")

    for axis in axes.flat:
        axis.set_xlabel("Optimizer step")
        axis.grid(True, alpha=0.20, linewidth=0.7)
        axis.set_xlim(1, total_steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--png-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_and_normalize(args.source)
    validation = load_validation(args.summary)
    write_csv(args.csv_output, rows)
    render_plot(rows, validation, args.png_output, beta=0.1, moving_average_window=20)
    manifest = {
        "valid": True,
        "source_path": str(args.source),
        "source_sha256": sha256(args.source),
        "summary_sha256": sha256(args.summary),
        "row_count": len(rows),
        "step_range": [1, int(rows[-1]["step"])],
        "validation_steps": [row["step"] for row in validation],
        "pref_beta": 0.1,
        "moving_average_window": 20,
        "csv_sha256": sha256(args.csv_output),
        "png_sha256": sha256(args.png_output),
        "final_train_ma20": {
            "chosen": rows[-1]["chosen_ma20"],
            "rejected": rows[-1]["rejected_ma20"],
            "margin": rows[-1]["margin_ma20"],
        },
        "final_validation": validation[-1],
    }
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
