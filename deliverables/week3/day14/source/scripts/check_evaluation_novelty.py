#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unicodedata
from typing import Any


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def character_features(text: str) -> set[str]:
    if not text:
        return set()
    features = {f"u:{character}" for character in text}
    features.update(f"b:{text[index:index + 2]}" for index in range(len(text) - 1))
    return features


def jaccard(left: str, right: str) -> float:
    left_features = character_features(left)
    right_features = character_features(right)
    union = left_features | right_features
    return 1.0 if not union else len(left_features & right_features) / len(union)


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions file must contain a non-empty list")
    return questions


def question_text(question: dict[str, Any]) -> str:
    messages = question.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"question {question.get('id')} is missing messages")
    text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    )
    if not text.strip():
        raise ValueError(f"question {question.get('id')} has no user prompt")
    return text


def load_training_prompts(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            instruction = str(record.get("instruction", ""))
            input_text = str(record.get("input", ""))
            prompt = "\n".join(part for part in (instruction, input_text) if part.strip())
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
        raise ValueError("threshold must be between 0 and 1")
    results: list[dict[str, Any]] = []
    for question in questions:
        text = question_text(question)
        normalized = normalize(text)
        ranked = sorted(
            (
                (jaccard(normalized, prompt["normalized"]), prompt)
                for prompt in training_prompts
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        maximum, closest = ranked[0]
        exact_matches = [
            prompt for prompt in training_prompts if normalized == prompt["normalized"]
        ]
        status = "rejected_exact" if exact_matches else (
            "manual_review" if maximum >= threshold else "novel"
        )
        results.append(
            {
                "id": question.get("id"),
                "exact_match": bool(exact_matches),
                "maximum_jaccard": maximum,
                "closest_training_line": closest["line_number"],
                "closest_sample_id": closest["sample_id"],
                "closest_training_prompt": closest["text"],
                "status": status,
            }
        )
    return {
        "question_count": len(results),
        "training_prompt_count": len(training_prompts),
        "jaccard_threshold": threshold,
        "exact_match_count": sum(item["exact_match"] for item in results),
        "manual_review_count": sum(item["status"] == "manual_review" for item in results),
        "all_questions_novel": all(item["status"] == "novel" for item in results),
        "questions": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Day 14 prompts against training data.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = evaluate_questions(
            load_questions(args.questions),
            load_training_prompts(args.training_data),
            args.threshold,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0 if report["exact_match_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
