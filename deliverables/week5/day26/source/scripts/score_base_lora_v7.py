#!/usr/bin/env python3
"""Reveal blind Day26 v7 scores and recompute all quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


WEIGHTS = {
    "factual": 0.35,
    "instruction": 0.25,
    "completeness": 0.15,
    "usefulness": 0.15,
    "format": 0.10,
}
CATEGORIES = ("natural_scene", "ocr", "chart_table", "ui", "formula")
MODES = ("direct", "verification")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_score(scores: dict[str, float]) -> float:
    if set(scores) != set(WEIGHTS):
        raise ValueError("score dimensions do not match the frozen rubric")
    if any(not 1 <= scores[name] <= 5 for name in WEIGHTS):
        raise ValueError("every score dimension must be between 1 and 5")
    return sum(scores[name] * WEIGHTS[name] for name in WEIGHTS)


def validate_scored_rows(rows: list[dict]) -> None:
    if len(rows) != 20 or len({row.get("case_id") for row in rows}) != 20:
        raise ValueError("scoring requires exactly 20 unique cases")
    if Counter(row.get("category") for row in rows) != Counter(
        {category: 4 for category in CATEGORIES}
    ):
        raise ValueError("scored rows do not preserve category balance")
    if Counter(row.get("mode") for row in rows) != Counter({mode: 10 for mode in MODES}):
        raise ValueError("scored rows do not preserve direct/verification balance")
    for row in rows:
        for model in ("base", "lora"):
            payload = row.get(model)
            if not isinstance(payload, dict) or not isinstance(payload.get("hallucination"), bool):
                raise ValueError(f"invalid {model} scoring payload for {row.get('case_id')}")
            weighted_score(payload.get("scores", {}))


def score_comparison(rows: list[dict], thresholds: dict, *, split: str) -> dict:
    if split not in {"dev", "final"}:
        raise ValueError("split must be dev or final")
    validate_scored_rows(rows)
    base_scores = [weighted_score(row["base"]["scores"]) for row in rows]
    lora_scores = [weighted_score(row["lora"]["scores"]) for row in rows]
    direct_indices = [index for index, row in enumerate(rows) if row["mode"] == "direct"]
    mean_base = sum(base_scores) / len(base_scores)
    mean_lora = sum(lora_scores) / len(lora_scores)
    direct_base = sum(base_scores[index] for index in direct_indices) / len(direct_indices)
    direct_lora = sum(lora_scores[index] for index in direct_indices) / len(direct_indices)
    base_hallucinations = sum(row["base"]["hallucination"] for row in rows)
    lora_hallucinations = sum(row["lora"]["hallucination"] for row in rows)

    by_category: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_category[row["category"]].append(index)
    category_metrics = {}
    for category in CATEGORIES:
        indices = by_category[category]
        base_mean = sum(base_scores[index] for index in indices) / len(indices)
        lora_mean = sum(lora_scores[index] for index in indices) / len(indices)
        category_metrics[category] = {
            "mean_base": base_mean,
            "mean_lora": lora_mean,
            "mean_gain": lora_mean - base_mean,
            "lora_wins": sum(lora_scores[index] > base_scores[index] for index in indices),
        }

    mean_gain = mean_lora - mean_base
    direct_gain = direct_lora - direct_base
    lora_wins = sum(new > old for old, new in zip(base_scores, lora_scores))
    opportunity_indices = [
        index for index, base_score in enumerate(base_scores) if base_score < 5.0
    ]
    opportunity_count = len(opportunity_indices)
    opportunity_wins = sum(
        lora_scores[index] > base_scores[index] for index in opportunity_indices
    )
    opportunity_win_rate = (
        opportunity_wins / opportunity_count if opportunity_count else 0.0
    )
    base_hallucination_rate = base_hallucinations / len(rows)
    lora_hallucination_rate = lora_hallucinations / len(rows)
    checks = {
        "mean_gain": mean_gain >= thresholds["mean_gain_min"],
        "hallucination_rate": lora_hallucination_rate <= base_hallucination_rate,
    }
    if "lora_wins_min" in thresholds:
        checks["lora_wins"] = lora_wins >= thresholds["lora_wins_min"]
    if "opportunity_win_rate_min" in thresholds:
        checks["opportunity_win_rate"] = (
            opportunity_win_rate >= thresholds["opportunity_win_rate_min"]
        )
    if "direct_gain_min" in thresholds:
        checks["direct_gain"] = direct_gain >= thresholds["direct_gain_min"]
    return {
        "schema_version": "day26-v7-score-summary-1",
        "split": split,
        "case_count": 20,
        "mean_base": mean_base,
        "mean_lora": mean_lora,
        "mean_gain": mean_gain,
        "direct_base": direct_base,
        "direct_lora": direct_lora,
        "direct_gain": direct_gain,
        "lora_wins": lora_wins,
        "opportunity_count": opportunity_count,
        "opportunity_wins": opportunity_wins,
        "opportunity_win_rate": opportunity_win_rate,
        "base_hallucination_count": base_hallucinations,
        "lora_hallucination_count": lora_hallucinations,
        "base_hallucination_rate": base_hallucination_rate,
        "lora_hallucination_rate": lora_hallucination_rate,
        "category_metrics": category_metrics,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def reveal_blind_scores(blind_rows: list[dict], key_rows: list[dict]) -> list[dict]:
    if len(blind_rows) != 20 or len(key_rows) != 20:
        raise ValueError("blind scores and key must each contain 20 rows")
    keys = {row["case_id"]: row for row in key_rows}
    if len(keys) != 20 or {row.get("case_id") for row in blind_rows} != set(keys):
        raise ValueError("blind scores and key case IDs do not match")
    revealed = []
    for row in blind_rows:
        key = keys[row["case_id"]]
        if {key.get("a_model"), key.get("b_model")} != {"base", "lora"}:
            raise ValueError("private key contains an invalid A/B mapping")
        revealed.append(
            {
                "case_id": row["case_id"],
                "category": row["category"],
                "mode": row["mode"],
                key["a_model"]: row["response_a"],
                key["b_model"]: row["response_b"],
            }
        )
    return revealed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-scores", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--scoring-config", type=Path, required=True)
    parser.add_argument("--inference-summary", type=Path, required=True)
    parser.add_argument("--revealed-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.scoring_config.read_text(encoding="utf-8"))
    inference = json.loads(args.inference_summary.read_text(encoding="utf-8"))
    if config.get("weights") != WEIGHTS or config.get("split") != inference.get("split"):
        raise ValueError("scoring config does not match the inference stage")
    blind = json.loads(args.blind_scores.read_text(encoding="utf-8"))
    key = json.loads(args.blind_key.read_text(encoding="utf-8"))
    revealed = reveal_blind_scores(blind, key)
    summary = score_comparison(revealed, config["pass_thresholds"], split=config["split"])
    for field in (
        "run_id",
        "run_spec_sha256",
        "generation_config_sha256",
        "scoring_config_sha256",
        "cases_sha256",
        "base_model_revision",
        "base_model_files_digest_sha256",
        "base_model_manifest_sha256",
        "adapter_config_sha256",
        "adapter_model_sha256",
        "lora_scale",
    ):
        summary[field] = inference.get(field)
    if summary["scoring_config_sha256"] != sha256(args.scoring_config):
        raise ValueError("inference was not bound to this scoring config")
    args.revealed_output.parent.mkdir(parents=True, exist_ok=True)
    args.revealed_output.write_text(
        json.dumps(revealed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
