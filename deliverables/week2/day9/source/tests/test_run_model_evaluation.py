import json
from pathlib import Path
import subprocess
import sys


DAY9_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = DAY9_ROOT / "source" / "scripts" / "run_model_evaluation.py"
QUESTIONS = DAY9_ROOT / "source" / "results" / "evaluation_questions.json"


def test_validate_only_accepts_frozen_three_question_spec(
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model-dir",
            "/models/example",
            "--model-label",
            "Model A",
            "--questions",
            str(QUESTIONS),
            "--output",
            str(tmp_path / "responses.jsonl"),
            "--manifest-output",
            str(output),
            "--validate-only",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["valid"] is True
    assert manifest["question_count"] == 3
    assert manifest["question_ids"] == ["Q1", "Q2", "Q3"]
    assert manifest["generation_config"]["do_sample"] is False
    assert manifest["generation_config"]["seed"] == 42


def test_validate_only_rejects_duplicate_question_ids(
    tmp_path: Path,
) -> None:
    spec = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    spec["questions"][1]["id"] = "Q1"
    invalid_questions = tmp_path / "questions.json"
    invalid_questions.write_text(
        json.dumps(spec, ensure_ascii=False),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model-dir",
            "/models/example",
            "--model-label",
            "Model A",
            "--questions",
            str(invalid_questions),
            "--output",
            str(tmp_path / "responses.jsonl"),
            "--manifest-output",
            str(tmp_path / "manifest.json"),
            "--validate-only",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "question ids must be unique" in result.stderr


def test_validate_only_rejects_sampling_for_fair_comparison(
    tmp_path: Path,
) -> None:
    spec = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    spec["generation_config"]["do_sample"] = True
    invalid_questions = tmp_path / "questions.json"
    invalid_questions.write_text(
        json.dumps(spec, ensure_ascii=False),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model-dir",
            "/models/example",
            "--model-label",
            "Model A",
            "--questions",
            str(invalid_questions),
            "--output",
            str(tmp_path / "responses.jsonl"),
            "--manifest-output",
            str(tmp_path / "manifest.json"),
            "--validate-only",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "do_sample must be false" in result.stderr
