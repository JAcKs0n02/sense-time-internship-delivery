import json
from pathlib import Path
import subprocess
import sys


DAY9_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = DAY9_ROOT / "source" / "scripts" / "build_evaluation_report.py"
RUBRIC = DAY9_ROOT / "source" / "results" / "evaluation_rubric.json"


def write_jsonl(path: Path, label: str) -> None:
    records = [
        {
            "question_id": question_id,
            "model_label": label,
            "messages": [
                {
                    "role": "user",
                    "content": f"Question {question_id}",
                }
            ],
            "generation_config": {"do_sample": False, "seed": 42},
            "response": f"{label} response {question_id}",
        }
        for question_id in ("Q1", "Q2", "Q3")
    ]
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def score_payload(model_a: int, model_b: int) -> dict:
    dimensions = (
        "instruction_completion",
        "correctness",
        "format_compliance",
        "relevance_completeness",
        "generation_quality",
    )
    return {
        "unblinding": {
            "Model A": "Base model",
            "Model B": "Merged model",
        },
        "scores": {
            label: {
                question_id: {
                    "dimensions": {
                        dimension: score
                        for dimension in dimensions
                    },
                    "notes": f"{label} {question_id}",
                }
                for question_id in ("Q1", "Q2", "Q3")
            }
            for label, score in (
                ("Model A", model_a),
                ("Model B", model_b),
            )
        },
    }


def test_report_declares_merged_better_when_total_is_higher_without_regression(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.jsonl"
    merged = tmp_path / "merged.jsonl"
    scores = tmp_path / "scores.json"
    write_jsonl(base, "Model A")
    write_jsonl(merged, "Model B")
    scores.write_text(
        json.dumps(score_payload(1, 2)),
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-responses",
            str(base),
            "--merged-responses",
            str(merged),
            "--rubric",
            str(RUBRIC),
            "--scores",
            str(scores),
            "--comparison-output",
            str(comparison),
            "--markdown-output",
            str(tmp_path / "comparison.md"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(comparison.read_text(encoding="utf-8"))
    assert report["totals"] == {"Model A": 15, "Model B": 30}
    assert report["generation_quality_regression"] is False
    assert report["merged_model_better"] is True
    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "Merged model better: **yes**" in markdown
    assert "Model B response Q3" in markdown


def test_report_rejects_missing_dimension_scores(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    merged = tmp_path / "merged.jsonl"
    scores = tmp_path / "scores.json"
    write_jsonl(base, "Model A")
    write_jsonl(merged, "Model B")
    payload = score_payload(1, 2)
    del payload["scores"]["Model B"]["Q2"]["dimensions"]["correctness"]
    scores.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-responses",
            str(base),
            "--merged-responses",
            str(merged),
            "--rubric",
            str(RUBRIC),
            "--scores",
            str(scores),
            "--comparison-output",
            str(tmp_path / "comparison.json"),
            "--markdown-output",
            str(tmp_path / "comparison.md"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "dimension scores must exactly match the rubric" in result.stderr
