import csv
import inspect
import sys
from pathlib import Path


DAY14_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = DAY14_ROOT / "source" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from aggregate_human_scores import (  # noqa: E402
    aggregate_human_scores,
    weighted_score,
)
from prepare_blind_review import prepare_blind_records  # noqa: E402
from select_best_model import select_best_model  # noqa: E402


RUBRIC = {
    "weights": {
        "accuracy": 0.30,
        "completeness": 0.25,
        "logic": 0.20,
        "safety": 0.15,
        "format": 0.10,
    }
}


def test_automatic_metrics_are_not_ranking_inputs() -> None:
    assert "automatic_score" not in inspect.signature(select_best_model).parameters


def test_human_score_formula() -> None:
    row = {"accuracy": 5, "completeness": 4, "logic": 3, "safety": 5, "format": 4}

    assert weighted_score(row, RUBRIC) == 4.25


def test_blind_package_has_no_model_identity_or_path() -> None:
    responses = [
        {
            "candidate_id": "rank-r8",
            "adapter_path": "/secret/rank-r8",
            "question_id": "MATH-01",
            "messages": [{"role": "user", "content": "1+1=?"}],
            "raw_response": "2",
            "status": "completed",
        },
        {
            "candidate_id": "base",
            "adapter_path": None,
            "question_id": "MATH-01",
            "messages": [{"role": "user", "content": "1+1=?"}],
            "raw_response": "3",
            "status": "completed",
        },
    ]

    question_metadata = {
        "MATH-01": {
            "reference": {"answer": 2},
            "human_scoring_notes": "检查结果。",
        }
    }

    blind_rows, mapping = prepare_blind_records(
        responses,
        seed=20260814,
        question_metadata=question_metadata,
    )

    assert len(blind_rows) == 2
    assert all("candidate_id" not in row for row in blind_rows)
    assert all("adapter_path" not in row for row in blind_rows)
    assert all(row["prompt"] == "1+1=?" for row in blind_rows)
    assert all(row["reference"] == '{"answer": 2}' for row in blind_rows)
    assert all(row["human_scoring_notes"] == "检查结果。" for row in blind_rows)
    assert {item["candidate_id"] for item in mapping["records"]} == {"base", "rank-r8"}


def test_blinding_is_deterministic_but_position_is_randomized() -> None:
    responses = [
        {
            "candidate_id": model,
            "question_id": question,
            "messages": [{"role": "user", "content": question}],
            "raw_response": f"{model}-{question}",
            "status": "completed",
        }
        for question in ("Q1", "Q2")
        for model in ("base", "a", "b")
    ]

    first, first_mapping = prepare_blind_records(responses, seed=99)
    second, second_mapping = prepare_blind_records(responses, seed=99)

    assert first == second
    assert first_mapping == second_mapping
    assert [row["candidate_label"] for row in first[:3]] == ["Candidate-01", "Candidate-02", "Candidate-03"]


def test_aggregation_requires_two_distinct_complete_reviewers() -> None:
    rows = [
        {
            "review_id": "R1",
            "reviewer_id": "reviewer_1",
            "accuracy": 5,
            "completeness": 5,
            "logic": 5,
            "safety": 5,
            "format": 5,
            "reason": "完整正确",
        }
    ]
    mapping = {"records": [{"review_id": "R1", "candidate_id": "rank-r8", "question_id": "Q1"}]}

    try:
        aggregate_human_scores(rows, RUBRIC, mapping)
    except ValueError as error:
        assert "exactly two" in str(error)
    else:
        raise AssertionError("single-rater scores were accepted as final")


def test_two_rater_scores_are_aggregated_by_model() -> None:
    rows = []
    for reviewer_id, score in (("reviewer_1", 5), ("reviewer_2", 3)):
        rows.append(
            {
                "review_id": "R1",
                "reviewer_id": reviewer_id,
                "accuracy": score,
                "completeness": score,
                "logic": score,
                "safety": score,
                "format": score,
                "reason": "独立评分",
            }
        )
    mapping = {"records": [{"review_id": "R1", "candidate_id": "rank-r8", "question_id": "Q1"}]}

    result = aggregate_human_scores(rows, RUBRIC, mapping)

    assert result == [
        {
            "model_id": "rank-r8",
            "answer_count": 1,
            "rater_count": 2,
            "accuracy": 4.0,
            "completeness": 4.0,
            "logic": 4.0,
            "safety": 4.0,
            "format": 4.0,
            "weighted_total": 4.0,
            "rater_total_disagreement": 2.0,
        }
    ]


def test_tie_breakers_remain_human_first() -> None:
    human_scores = [
        {
            "model_id": "rank-r8",
            "weighted_total": 4.0,
            "accuracy": 4.0,
            "rater_total_disagreement": 0.2,
        },
        {
            "model_id": "rank-r32",
            "weighted_total": 4.0,
            "accuracy": 4.5,
            "rater_total_disagreement": 0.4,
        },
        {
            "model_id": "base",
            "weighted_total": 5.0,
            "accuracy": 5.0,
            "rater_total_disagreement": 0.0,
        },
    ]
    run_summaries = [
        {"run_id": "rank-r8", "adapter_size_bytes": 10, "train_runtime_seconds": 100},
        {"run_id": "rank-r32", "adapter_size_bytes": 20, "train_runtime_seconds": 90},
    ]

    winner = select_best_model(human_scores, run_summaries)

    assert winner["model_id"] == "rank-r32"
    assert winner["tie_breaker"] == "human_accuracy"
    assert winner["base_comparison"]["base_weighted_total"] == 5.0
