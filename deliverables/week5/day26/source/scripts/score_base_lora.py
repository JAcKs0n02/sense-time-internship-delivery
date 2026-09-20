#!/usr/bin/env python3
"""Compute the frozen Day26 Base-vs-LoRA quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


WEIGHTS = {
    "factual": 0.35,
    "instruction": 0.25,
    "completeness": 0.15,
    "usefulness": 0.15,
    "format": 0.10,
}


def load_scoring_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights")
    thresholds = payload.get("pass_thresholds")
    if weights != WEIGHTS:
        raise ValueError(f"evaluation weights drifted from the scoring contract: {weights}")
    required = {
        "mean_gain_min",
        "lora_wins_min",
        "hallucination_rate_not_worse",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != required:
        raise ValueError("evaluation pass thresholds are incomplete or unexpected")
    if thresholds["hallucination_rate_not_worse"] is not True:
        raise ValueError("hallucination_rate_not_worse must remain true")
    return {
        "weights": weights,
        "thresholds": thresholds,
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def weighted_score(scores: dict[str, float]) -> float:
    missing = set(WEIGHTS) - set(scores)
    if missing:
        raise ValueError(f"missing score dimensions: {sorted(missing)}")
    for key in WEIGHTS:
        if not 1 <= scores[key] <= 5:
            raise ValueError(f"{key} score must be between 1 and 5")
    return sum(scores[key] * weight for key, weight in WEIGHTS.items())


def reveal_blind_scores(blind_rows: list[dict], key_rows: list[dict]) -> list[dict]:
    blind_ids = [row.get("case_id") for row in blind_rows]
    key_ids = [row.get("case_id") for row in key_rows]
    if len(set(blind_ids)) != len(blind_ids) or set(blind_ids) != set(key_ids):
        raise ValueError("blind-score and private-key case IDs do not match")
    key_by_id = {row["case_id"]: row for row in key_rows}
    revealed = []
    for row in blind_rows:
        key = key_by_id[row["case_id"]]
        if {key.get("a_model"), key.get("b_model")} != {"base", "lora"}:
            raise ValueError(f"invalid private mapping for {row['case_id']}")
        revealed.append(
            {
                "case_id": row["case_id"],
                key["a_model"]: row["response_a"],
                key["b_model"]: row["response_b"],
            }
        )
    return revealed


def score_comparison(rows: list[dict], thresholds: dict | None = None) -> dict:
    if len(rows) != 20:
        raise ValueError("the frozen final comparison must contain exactly 20 cases")
    base = [weighted_score(row["base"]["scores"]) for row in rows]
    lora = [weighted_score(row["lora"]["scores"]) for row in rows]
    mean_base = sum(base) / len(base)
    mean_lora = sum(lora) / len(lora)
    mean_gain = mean_lora - mean_base
    lora_wins = sum(new > old for old, new in zip(base, lora))
    base_hallucinations = sum(bool(row["base"]["hallucination"]) for row in rows)
    lora_hallucinations = sum(bool(row["lora"]["hallucination"]) for row in rows)
    base_rate = base_hallucinations / len(rows)
    lora_rate = lora_hallucinations / len(rows)
    thresholds = thresholds or {
        "mean_gain_min": 0.50,
        "lora_wins_min": 12,
        "hallucination_rate_not_worse": True,
    }
    checks = {
        "mean_gain": mean_gain >= thresholds["mean_gain_min"],
        "lora_wins": lora_wins >= thresholds["lora_wins_min"],
        "hallucination_rate": lora_rate <= base_rate,
    }
    return {
        "mean_base": mean_base,
        "mean_lora": mean_lora,
        "mean_gain": mean_gain,
        "lora_wins": lora_wins,
        "base_hallucination_rate": base_rate,
        "lora_hallucination_rate": lora_rate,
        "checks": checks,
        "passed": all(checks.values()),
    }


def read_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-scores", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--revealed-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_scoring_protocol(args.config)
    revealed = reveal_blind_scores(read_json(args.blind_scores), read_json(args.blind_key))
    summary = score_comparison(revealed, protocol["thresholds"])
    summary["scoring_config_sha256"] = protocol["config_sha256"]
    write_json(args.revealed_output, revealed)
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
