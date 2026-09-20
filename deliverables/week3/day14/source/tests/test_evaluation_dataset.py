import json
import sys
from collections import Counter
from pathlib import Path


DAY14_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = DAY14_ROOT / "source" / "scripts"
sys.path.insert(0, str(DAY14_ROOT))
sys.path.insert(0, str(SCRIPTS))

from eval_harness import load_questions, load_rubric  # noqa: E402
from check_evaluation_novelty import (  # noqa: E402
    evaluate_questions,
    load_training_prompts,
)


QUESTIONS_PATH = DAY14_ROOT / "source" / "data" / "evaluation_questions.json"
RUBRIC_PATH = DAY14_ROOT / "source" / "data" / "evaluation_rubric.json"


def test_question_distribution_and_unique_ids() -> None:
    questions = load_questions(QUESTIONS_PATH)

    assert len(questions) == 20
    assert len({question["id"] for question in questions}) == 20
    assert Counter(question["category"] for question in questions) == {
        "math": 7,
        "reasoning": 7,
        "code": 6,
    }


def test_every_question_has_the_frozen_contract() -> None:
    questions = load_questions(QUESTIONS_PATH)

    for question in questions:
        assert set(question) >= {
            "id",
            "category",
            "difficulty",
            "messages",
            "reference",
            "automatic_check",
            "human_scoring_notes",
        }
        assert question["difficulty"] in {"hard", "very_hard"}
        assert question["messages"][-1]["role"] == "user"
        assert question["messages"][-1]["content"].strip()


def test_code_questions_define_signature_and_assertions() -> None:
    questions = load_questions(QUESTIONS_PATH)

    for question in questions:
        if question["category"] == "code":
            check = question["automatic_check"]
            assert check["type"] == "python_tests"
            assert check["function_signature"].startswith("def ")
            assert check["assertions"]


def test_rubric_weights_and_behavioral_anchors() -> None:
    rubric = load_rubric(RUBRIC_PATH)

    assert rubric["weights"] == {
        "accuracy": 0.30,
        "completeness": 0.25,
        "logic": 0.20,
        "safety": 0.15,
        "format": 0.10,
    }
    for dimension in rubric["weights"]:
        assert set(rubric["dimensions"][dimension]["anchors"]) == {
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
        }


def test_novelty_report_retains_closest_candidate(tmp_path: Path) -> None:
    training_path = tmp_path / "train.jsonl"
    training_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "instruction": "说明二分查找",
                "input": "",
                "output": "每次缩小一半范围",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    prompts = load_training_prompts(training_path)
    questions = [
        {
            "id": "MATH-X",
            "messages": [{"role": "user", "content": "求解一个概率问题"}],
        }
    ]

    report = evaluate_questions(questions, prompts, threshold=0.8)

    item = report["questions"][0]
    assert item["status"] == "novel"
    assert item["closest_training_line"] == 1
    assert item["closest_sample_id"] == "sample-1"
