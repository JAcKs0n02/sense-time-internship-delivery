import sys
from pathlib import Path


DAY14_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = DAY14_ROOT / "source" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_automatic_evidence import summarize_automatic_evidence  # noqa: E402


def test_summary_keeps_automatic_evidence_out_of_ranking() -> None:
    questions = [
        {"id": "MATH-01", "category": "math"},
        {"id": "CODE-01", "category": "code"},
    ]
    payload = {
        "ranking_input": False,
        "records": [
            {
                "candidate_id": "base",
                "question_id": "MATH-01",
                "automatic_evidence": {
                    "status": "checked",
                    "passed": True,
                },
            },
            {
                "candidate_id": "base",
                "question_id": "CODE-01",
                "automatic_evidence": {
                    "code_execution_status": "disabled_no_sandbox",
                    "tests_passed": None,
                    "reason": "namespace unavailable",
                },
            },
        ],
    }

    result = summarize_automatic_evidence(payload, questions)

    assert result["ranking_input"] is False
    assert result["candidate_count"] == 1
    assert result["record_count"] == 2
    assert result["rows"] == [
        {
            "candidate_id": "base",
            "category": "code",
            "question_count": 1,
            "checked_count": 0,
            "passed_count": 0,
            "failed_check_count": 0,
            "code_tests_passed_count": 0,
            "code_execution_disabled_count": 1,
            "not_evaluable_count": 0,
        },
        {
            "candidate_id": "base",
            "category": "math",
            "question_count": 1,
            "checked_count": 1,
            "passed_count": 1,
            "failed_check_count": 0,
            "code_tests_passed_count": 0,
            "code_execution_disabled_count": 0,
            "not_evaluable_count": 0,
        },
    ]


def test_summary_rejects_any_attempt_to_mark_evidence_as_ranking_input() -> None:
    try:
        summarize_automatic_evidence({"ranking_input": True, "records": []}, [])
    except ValueError as error:
        assert "ranking" in str(error)
    else:
        raise AssertionError("ranking-enabled automatic evidence was accepted")
