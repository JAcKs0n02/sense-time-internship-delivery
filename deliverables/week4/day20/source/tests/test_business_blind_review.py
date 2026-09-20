from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PREPARE = Path(__file__).parents[1] / "scripts" / "prepare_business_blind_review.py"
SUMMARY = Path(__file__).parents[1] / "scripts" / "summarize_business_review.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def responses() -> list[dict]:
    rows = []
    for index in range(1, 6):
        prompt_id = f"w4-eval-business-{index:02d}"
        for alias in ("sft_only", "sft_dpo"):
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_type": "business",
                    "prompt_text": f"prompt {index}",
                    "model_alias": alias,
                    "response": f"answer {index} from {alias}",
                    "status": "completed",
                }
            )
    return rows


def scores(blind_rows: list[dict]) -> list[dict]:
    return [
        {
            "prompt_id": row["prompt_id"],
            "candidate_label": row["candidate_label"],
            "accuracy": "4",
            "completeness": "3",
            "logic": "5",
            "safety": "5",
            "format": "4",
            "reason": "内容准确，结构完整，格式符合要求。",
            "reviewer_type": "codex_review",
        }
        for row in blind_rows
    ]


def test_seeded_blinding_is_deterministic_and_does_not_leak_metadata():
    module = load(PREPARE, "prepare_business_blind_review")
    first_rows, first_mapping = module.build_blind_rows(responses(), seed=42)
    second_rows, second_mapping = module.build_blind_rows(responses(), seed=42)
    assert first_rows == second_rows
    assert first_mapping == second_mapping
    assert len(first_rows) == 10
    assert all("model_alias" not in row and "model_dir" not in row for row in first_rows)
    assert {row["candidate_label"] for row in first_rows} == {"A", "B"}


def test_weighted_scoring_and_unblind_summary():
    prepare = load(PREPARE, "prepare_business_blind_review")
    summary = load(SUMMARY, "summarize_business_review")
    blind_rows, mapping = prepare.build_blind_rows(responses(), seed=42)
    comparison, report = summary.validate_and_unblind(scores(blind_rows), mapping)
    assert len(comparison) == 10
    assert all(row["weighted_score"] == 4.1 for row in comparison)
    assert report["model_prompt_counts"] == {"sft_only": 5, "sft_dpo": 5}
    assert report["weighted_mean_delta_sft_dpo_minus_sft_only"] == 0.0
    assert report["prompt_outcomes"] == {
        "sft_dpo_wins": 0,
        "sft_only_wins": 0,
        "ties": 5,
    }
    assert report["dimension_means"]["sft_dpo"]["accuracy"] == 4.0


def test_rejects_missing_reason_out_of_range_and_mapping_mismatch():
    prepare = load(PREPARE, "prepare_business_blind_review")
    summary = load(SUMMARY, "summarize_business_review")
    blind_rows, mapping = prepare.build_blind_rows(responses(), seed=42)
    rows = scores(blind_rows)
    rows[0]["reason"] = ""
    with pytest.raises(ValueError, match="reason"):
        summary.validate_and_unblind(rows, mapping)
    rows = scores(blind_rows)
    rows[0]["accuracy"] = "6"
    with pytest.raises(ValueError, match="accuracy"):
        summary.validate_and_unblind(rows, mapping)
    rows = scores(blind_rows)
    mapping = mapping[:-1]
    with pytest.raises(ValueError, match="mapping|exactly"):
        summary.validate_and_unblind(rows, mapping)
