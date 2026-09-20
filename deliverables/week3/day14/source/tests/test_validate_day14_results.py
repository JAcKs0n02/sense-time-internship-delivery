import json
import sys
from pathlib import Path


DAY14_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = DAY14_ROOT / "source" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_day14_results import validate_results  # noqa: E402


def write_fixture(root: Path, *, response_count: int = 4) -> None:
    root.mkdir()
    (root / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "candidate_count": 2,
                "question_count": 2,
                "terminal_record_count": response_count,
                "failed_record_count": 0,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "candidate_id": candidate,
            "question_id": question,
            "question_sha256": f"hash-{question}",
            "status": "completed",
        }
        for candidate in ("base", "adapter")
        for question in ("Q1", "Q2")
    ][:response_count]
    (root / "all_responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    evidence = [
        {
            "candidate_id": row["candidate_id"],
            "question_id": row["question_id"],
            "automatic_evidence": {"status": "checked", "passed": True},
        }
        for row in rows
    ]
    (root / "automatic_evidence.json").write_text(
        json.dumps({"ranking_input": False, "records": evidence}),
        encoding="utf-8",
    )


def test_complete_unique_matrix_passes(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    write_fixture(output)

    result = validate_results(output, expected_candidates=2, expected_questions=2)

    assert result["valid"] is True
    assert result["terminal_record_count"] == 4
    assert result["automatic_evidence_count"] == 4
    assert result["candidate_question_counts"] == {"adapter": 2, "base": 2}


def test_incomplete_matrix_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    write_fixture(output, response_count=3)

    result = validate_results(output, expected_candidates=2, expected_questions=2)

    assert result["valid"] is False
    assert any("expected 4 terminal records" in error for error in result["errors"])
