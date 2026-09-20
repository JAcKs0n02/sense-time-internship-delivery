import json
from pathlib import Path
import subprocess
import sys


DAY9_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = DAY9_ROOT / "source" / "scripts" / "check_evaluation_novelty.py"


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def run_check(
    tmp_path: Path,
    questions: list[dict],
    training_records: list[dict],
    *,
    threshold: float = 0.8,
) -> subprocess.CompletedProcess[str]:
    questions_path = tmp_path / "evaluation_questions.json"
    questions_path.write_text(
        json.dumps({"questions": questions}, ensure_ascii=False),
        encoding="utf-8",
    )
    training_path = tmp_path / "train.jsonl"
    write_jsonl(training_path, training_records)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--questions",
            str(questions_path),
            "--training-data",
            str(training_path),
            "--json-output",
            str(tmp_path / "novelty_report.json"),
            "--text-output",
            str(tmp_path / "novelty_report.txt"),
            "--jaccard-threshold",
            str(threshold),
        ],
        capture_output=True,
        text=True,
    )


def test_accepts_questions_with_no_exact_or_fuzzy_training_match(
    tmp_path: Path,
) -> None:
    questions = [
        {
            "id": "Q1",
            "messages": [
                {"role": "user", "content": "请解释微调数据去重的重要性。"}
            ],
        },
        {
            "id": "Q2",
            "messages": [
                {"role": "user", "content": "把模糊需求改写为可验收的任务。"}
            ],
        },
    ]
    training = [
        {
            "instruction": "写一首关于春天的诗",
            "input": "",
            "output": "春风吹过树林。",
        },
        {
            "instruction": "解释二分查找",
            "input": "",
            "output": "每次将搜索范围减半。",
        },
    ]

    result = run_check(tmp_path, questions, training)

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(
        (tmp_path / "novelty_report.json").read_text(encoding="utf-8")
    )
    assert report["question_count"] == 2
    assert report["all_questions_novel"] is True
    assert all(item["exact_match"] is False for item in report["questions"])
    assert all(item["status"] == "novel" for item in report["questions"])


def test_rejects_normalized_exact_match(tmp_path: Path) -> None:
    questions = [
        {
            "id": "Q1",
            "messages": [
                {"role": "user", "content": "解释：二分查找！"}
            ],
        }
    ]
    training = [
        {
            "instruction": "解释二分查找",
            "input": "",
            "output": "每次将搜索范围减半。",
        }
    ]

    result = run_check(tmp_path, questions, training)

    assert result.returncode != 0
    report = json.loads(
        (tmp_path / "novelty_report.json").read_text(encoding="utf-8")
    )
    assert report["all_questions_novel"] is False
    assert report["questions"][0]["exact_match"] is True
    assert report["questions"][0]["status"] == "rejected_exact"


def test_rejects_high_jaccard_similarity(tmp_path: Path) -> None:
    questions = [
        {
            "id": "Q1",
            "messages": [
                {"role": "user", "content": "请说明训练数据清洗流程"}
            ],
        }
    ]
    training = [
        {
            "instruction": "请说明训练数据的清洗流程",
            "input": "",
            "output": "先过滤空值，再清理文本。",
        }
    ]

    result = run_check(tmp_path, questions, training, threshold=0.7)

    assert result.returncode != 0
    report = json.loads(
        (tmp_path / "novelty_report.json").read_text(encoding="utf-8")
    )
    item = report["questions"][0]
    assert item["exact_match"] is False
    assert item["maximum_jaccard"] >= 0.7
    assert item["status"] == "rejected_fuzzy"
