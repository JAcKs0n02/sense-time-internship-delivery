#!/usr/bin/env python3
"""Normalize LLaMA-Factory trainer state into immutable candidate trend gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def _metric(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in row and isinstance(row[name], (int, float)):
            return float(row[name])
    return None


def summarize(state: dict[str, Any], adapter: Path) -> dict[str, Any]:
    history = state.get("log_history", [])
    train = [row for row in history if "loss" in row and "eval_loss" not in row]
    evaluations = [row for row in history if "eval_loss" in row]
    values = [float(value) for row in history for value in row.values() if isinstance(value, (int, float))]
    finite = bool(values) and all(math.isfinite(value) for value in values)
    first = train[:20]
    final = train[-20:]
    first_chosen = [_metric(row, "rewards/chosen", "rewards_chosen") for row in first]
    final_chosen = [_metric(row, "rewards/chosen", "rewards_chosen") for row in final]
    first_rejected = [_metric(row, "rewards/rejected", "rewards_rejected") for row in first]
    final_rejected = [_metric(row, "rewards/rejected", "rewards_rejected") for row in final]
    first_chosen = [x for x in first_chosen if x is not None]
    final_chosen = [x for x in final_chosen if x is not None]
    first_rejected = [x for x in first_rejected if x is not None]
    final_rejected = [x for x in final_rejected if x is not None]
    eval_margins = [_metric(row, "eval_rewards/margins", "eval_rewards_margins") for row in evaluations]
    eval_margins = [x for x in eval_margins if x is not None]
    eval_accuracy = [_metric(row, "eval_rewards/accuracies", "eval_rewards_accuracies") for row in evaluations]
    eval_accuracy = [x for x in eval_accuracy if x is not None]
    adapter_file = adapter / "adapter_model.safetensors"
    return {
        "training_complete": bool(state.get("global_step")) and state.get("global_step") == state.get("max_steps"),
        "finite_metrics": finite,
        "adapter_nonempty": adapter_file.is_file() and adapter_file.stat().st_size > 0,
        "validation_margin_improved": len(eval_margins) >= 2 and eval_margins[-1] > eval_margins[0],
        "train_window_chosen_improved": bool(first_chosen and final_chosen) and mean(final_chosen) > mean(first_chosen),
        "train_window_rejected_improved": bool(first_rejected and final_rejected) and mean(final_rejected) < mean(first_rejected),
        "validation_reward_accuracy": eval_accuracy[-1] if eval_accuracy else 0.0,
        "validation_margins": eval_margins,
        "first_window_chosen_mean": mean(first_chosen) if first_chosen else None,
        "final_window_chosen_mean": mean(final_chosen) if final_chosen else None,
        "first_window_rejected_mean": mean(first_rejected) if first_rejected else None,
        "final_window_rejected_mean": mean(final_rejected) if final_rejected else None,
        "training_cost_steps": state.get("global_step", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.trainer_state.read_text()), args.adapter_dir)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

