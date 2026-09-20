from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
RESULT = ROOT / "source" / "results" / "reward_corrective_40step"


def test_corrective_review_matches_raw_development_responses_and_arithmetic():
    responses_path = RESULT / "responses.jsonl"
    rows = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines()]
    review = json.loads((RESULT / "corrective_dev_review.json").read_text(encoding="utf-8"))
    assert len(rows) == 35
    assert len({row["prompt_id"] for row in rows}) == 35
    assert all(row["model_alias"] == "reward-corrective-40step" for row in rows)
    assert review["response_sha256"] == hashlib.sha256(responses_path.read_bytes()).hexdigest()
    safety = review["safety"]
    assert safety["harmful_denominator"] == 20
    assert safety["teacher_refusal_count"] == 20 - len(safety["unsafe_ids"])
    assert safety["teacher_refusal_rate"] == pytest.approx(safety["teacher_refusal_count"] / 20)
    assert safety["actionable_harm_count"] == len(safety["unsafe_ids"])
    assert safety["benign_appropriate_help_rate"] == pytest.approx(
        safety["benign_appropriate_help_count"] / safety["benign_denominator"]
    )
    scores = review["business"]["scores"]
    assert len(scores) == 5
    assert review["business"]["candidate_mean"] == pytest.approx(
        sum(row["weighted_score"] for row in scores) / len(scores)
    )
    assert review["business"]["difference"] == pytest.approx(
        review["business"]["candidate_mean"] - review["business"]["sft_only_mean"]
    )
    assert all(review["promotion_gates"].values())
    assert review["passes_all_promotion_gates"] is True


def test_corrective_training_summary_matches_raw_trainer_state_and_trend():
    summary = json.loads((RESULT / "corrective_training_summary.json").read_text(encoding="utf-8"))
    state = json.loads((RESULT / "trainer_output" / "trainer_state.json").read_text(encoding="utf-8"))
    trend = json.loads((RESULT / "reward_trend.json").read_text(encoding="utf-8"))
    evaluations = [row for row in state["log_history"] if "eval_loss" in row]
    assert summary["status"] == {"status": "completed", "returncode": 0}
    assert summary["global_step"] == state["global_step"] == 40
    assert summary["max_steps"] == state["max_steps"] == 40
    assert summary["evaluation_history"] == [
        {
            key: row[key]
            for key in (
                "epoch",
                "eval_loss",
                "eval_rewards/chosen",
                "eval_rewards/rejected",
                "eval_rewards/margins",
                "eval_rewards/accuracies",
                "step",
            )
        }
        for row in evaluations
    ]
    for key, value in trend.items():
        assert summary["reward_trend"][key] == value
    assert summary["reward_trend"]["teacher_trend_observed"] is True
    assert summary["adapter"]["bytes"] == 80_792_096
    assert len(summary["adapter"]["sha256"]) == 64
