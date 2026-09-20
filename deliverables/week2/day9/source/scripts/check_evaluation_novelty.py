#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unicodedata
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check evaluation questions against Alpaca training data."
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--jaccard-threshold", type=float, default=0.8)
    return parser.parse_args()


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if character.isalnum())


def character_features(text: str) -> set[str]:
    if not text:
        return set()
    features = {f"u:{character}" for character in text}
    features.update(
        f"b:{text[index:index + 2]}" for index in range(len(text) - 1)
    )
    return features


def jaccard(left: str, right: str) -> float:
    left_features = character_features(left)
    right_features = character_features(right)
    union = left_features | right_features
    if not union:
        return 1.0
    return len(left_features & right_features) / len(union)


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions file must contain a non-empty questions list")
    return questions


def question_text(question: dict[str, Any]) -> str:
    messages = question.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"question {question.get('id')} is missing messages")
    user_contents = [
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    text = "\n".join(content for content in user_contents if content.strip())
    if not text:
        raise ValueError(f"question {question.get('id')} has no user content")
    return text


def load_training_prompts(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            instruction = str(record.get("instruction", ""))
            input_text = str(record.get("input", ""))
            prompt = "\n".join(
                part for part in (instruction, input_text) if part.strip()
            )
            if prompt:
                prompts.append(
                    {
                        "line_number": line_number,
                        "sample_id": record.get("sample_id"),
                        "text": prompt,
                        "normalized": normalize(prompt),
                    }
                )
    if not prompts:
        raise ValueError("training data contains no prompts")
    return prompts


def evaluate_questions(
    questions: list[dict[str, Any]],
    training_prompts: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("jaccard threshold must be between 0 and 1")

    results = []
    for question in questions:
        text = question_text(question)
        normalized = normalize(text)
        similarities = [
            jaccard(normalized, prompt["normalized"])
            for prompt in training_prompts
        ]
        best_index = max(
            range(len(similarities)),
            key=similarities.__getitem__,
        )
        best_prompt = training_prompts[best_index]
        maximum = similarities[best_index]
        exact_match = any(
            normalized == prompt["normalized"] for prompt in training_prompts
        )
        if exact_match:
            status = "rejected_exact"
        elif maximum >= threshold:
            status = "rejected_fuzzy"
        else:
            status = "novel"
        results.append(
            {
                "id": question.get("id"),
                "text": text,
                "normalized_text": normalized,
                "exact_match": exact_match,
                "maximum_jaccard": maximum,
                "closest_training_line": best_prompt["line_number"],
                "closest_sample_id": best_prompt["sample_id"],
                "status": status,
            }
        )
    return {
        "question_count": len(results),
        "training_prompt_count": len(training_prompts),
        "jaccard_threshold": threshold,
        "all_questions_novel": all(
            item["status"] == "novel" for item in results
        ),
        "questions": results,
    }


def write_reports(
    report: dict[str, Any],
    json_output: Path,
    text_output: Path,
) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"question_count={report['question_count']}",
        f"training_prompt_count={report['training_prompt_count']}",
        f"jaccard_threshold={report['jaccard_threshold']}",
        f"all_questions_novel={str(report['all_questions_novel']).lower()}",
    ]
    for item in report["questions"]:
        lines.append(
            " ".join(
                [
                    f"id={item['id']}",
                    f"status={item['status']}",
                    f"exact_match={str(item['exact_match']).lower()}",
                    f"maximum_jaccard={item['maximum_jaccard']:.6f}",
                    f"closest_training_line={item['closest_training_line']}",
                ]
            )
        )
    text_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        questions = load_questions(args.questions)
        prompts = load_training_prompts(args.training_data)
        report = evaluate_questions(
            questions,
            prompts,
            args.jaccard_threshold,
        )
        write_reports(report, args.json_output, args.text_output)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0 if report["all_questions_novel"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
