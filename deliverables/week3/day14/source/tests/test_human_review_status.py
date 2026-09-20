import csv
import sys
from pathlib import Path


DAY14_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = DAY14_ROOT / "source" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from check_human_review_status import inspect_review_file  # noqa: E402


FIELDS = [
    "review_id",
    "reviewer_id",
    "question_id",
    "candidate_label",
    "prompt",
    "reference",
    "human_scoring_notes",
    "response",
    "response_status",
    "accuracy",
    "completeness",
    "logic",
    "safety",
    "format",
    "reason",
]


def write_csv(path: Path, score: str = "", reason: str = "") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "review_id": "R1",
                "reviewer_id": "reviewer_1",
                "question_id": "Q1",
                "candidate_label": "Candidate-01",
                "prompt": "prompt",
                "response": "answer",
                "response_status": "completed",
                "accuracy": score,
                "completeness": score,
                "logic": score,
                "safety": score,
                "format": score,
                "reason": reason,
            }
        )


def test_blank_review_is_pending(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    write_csv(review)

    status = inspect_review_file(review, expected_rows=1)

    assert status["status"] == "pending"
    assert status["completed_rows"] == 0
    assert status["blank_score_rows"] == 1


def test_complete_review_is_completed(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    write_csv(review, score="4", reason="核心结论正确")

    status = inspect_review_file(review, expected_rows=1)

    assert status["status"] == "completed"
    assert status["completed_rows"] == 1
    assert status["errors"] == []
