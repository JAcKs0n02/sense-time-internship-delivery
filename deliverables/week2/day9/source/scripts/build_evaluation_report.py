#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the blinded Day 9 base-versus-merged report."
    )
    parser.add_argument("--base-responses", type=Path, required=True)
    parser.add_argument("--merged-responses", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        records.append(record)
    return records


def index_responses(
    records: list[dict[str, Any]],
    expected_label: str,
) -> dict[str, dict[str, Any]]:
    if len(records) != 3:
        raise ValueError(
            f"{expected_label} must contain exactly 3 response records"
        )
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = record.get("question_id")
        if record.get("model_label") != expected_label:
            raise ValueError(f"unexpected model label for {question_id}")
        if not isinstance(question_id, str) or question_id in indexed:
            raise ValueError(f"invalid or duplicate question id: {question_id}")
        if not isinstance(record.get("response"), str):
            raise ValueError(f"{question_id} response must be text")
        indexed[question_id] = record
    if set(indexed) != {"Q1", "Q2", "Q3"}:
        raise ValueError("response question ids must be Q1, Q2, and Q3")
    return indexed


def validate_scores(
    payload: dict[str, Any],
    dimension_ids: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    scores = payload.get("scores")
    if not isinstance(scores, dict) or set(scores) != {
        "Model A",
        "Model B",
    }:
        raise ValueError("scores must contain Model A and Model B")
    for label in ("Model A", "Model B"):
        model_scores = scores[label]
        if not isinstance(model_scores, dict) or set(model_scores) != {
            "Q1",
            "Q2",
            "Q3",
        }:
            raise ValueError(f"{label} scores must contain Q1, Q2, and Q3")
        for question_id, assessment in model_scores.items():
            if not isinstance(assessment, dict):
                raise ValueError(f"{label} {question_id} score is invalid")
            dimensions = assessment.get("dimensions")
            if not isinstance(dimensions, dict) or set(dimensions) != set(
                dimension_ids
            ):
                raise ValueError(
                    "dimension scores must exactly match the rubric"
                )
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value > 2
                for value in dimensions.values()
            ):
                raise ValueError("every dimension score must be 0, 1, or 2")
            if not isinstance(assessment.get("notes"), str):
                raise ValueError("every assessment must include notes")
    return scores


def question_text(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ):
            return message["content"]
    return ""


def build_report(
    base_records: list[dict[str, Any]],
    merged_records: list[dict[str, Any]],
    rubric: dict[str, Any],
    score_payload: dict[str, Any],
) -> dict[str, Any]:
    base = index_responses(base_records, "Model A")
    merged = index_responses(merged_records, "Model B")
    dimensions = rubric.get("dimensions")
    if not isinstance(dimensions, list):
        raise ValueError("rubric dimensions must be a list")
    dimension_ids = [item.get("id") for item in dimensions]
    if any(not isinstance(item, str) for item in dimension_ids):
        raise ValueError("rubric dimension ids must be strings")
    scores = validate_scores(score_payload, dimension_ids)

    comparisons: list[dict[str, Any]] = []
    totals = {"Model A": 0, "Model B": 0}
    generation_quality_totals = {"Model A": 0, "Model B": 0}
    for question_id in ("Q1", "Q2", "Q3"):
        question_comparison: dict[str, Any] = {
            "question_id": question_id,
            "question": question_text(base[question_id]),
            "models": {},
        }
        for label, records in (
            ("Model A", base),
            ("Model B", merged),
        ):
            assessment = scores[label][question_id]
            total = sum(assessment["dimensions"].values())
            totals[label] += total
            generation_quality_totals[label] += assessment["dimensions"][
                "generation_quality"
            ]
            question_comparison["models"][label] = {
                "response": records[question_id]["response"],
                "scores": assessment["dimensions"],
                "total": total,
                "notes": assessment["notes"],
                "runtime": {
                    "input_tokens": records[question_id].get("input_tokens"),
                    "output_tokens": records[question_id].get("output_tokens"),
                    "elapsed_seconds": records[question_id].get(
                        "elapsed_seconds"
                    ),
                },
            }
        comparisons.append(question_comparison)

    generation_quality_regression = (
        generation_quality_totals["Model B"]
        < generation_quality_totals["Model A"]
    )
    merged_model_better = (
        totals["Model B"] > totals["Model A"]
        and not generation_quality_regression
    )
    return {
        "evaluation": "Week 2 Day 9 base-versus-merged comparison",
        "blind_labels": {
            "Model A": "Base model",
            "Model B": "Merged model",
        },
        "rubric": rubric,
        "comparisons": comparisons,
        "totals": totals,
        "generation_quality_totals": generation_quality_totals,
        "generation_quality_regression": generation_quality_regression,
        "merged_model_better": merged_model_better,
        "decision_rule": rubric.get("decision_rule"),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Base vs Merged Model Comparison",
        "",
        "The questions and deterministic generation settings were frozen "
        "before inference. Scoring used five dimensions, each from 0 to 2.",
        "",
        f"- Model A total: {report['totals']['Model A']}/30",
        f"- Model B total: {report['totals']['Model B']}/30",
        "- Generation-quality regression: "
        + ("yes" if report["generation_quality_regression"] else "no"),
        "- Merged model better: "
        + ("**yes**" if report["merged_model_better"] else "**no**"),
        "",
        "Unblinding: Model A is the base model; Model B is the merged model.",
        "",
    ]
    for item in report["comparisons"]:
        lines.extend(
            [
                f"## {item['question_id']}",
                "",
                f"Question: {item['question']}",
                "",
            ]
        )
        for label in ("Model A", "Model B"):
            model = item["models"][label]
            score_text = ", ".join(
                f"{name}={value}" for name, value in model["scores"].items()
            )
            quoted_response = "\n".join(
                f"> {line}" if line else ">"
                for line in model["response"].splitlines()
            )
            lines.extend(
                [
                    f"### {label}",
                    "",
                    quoted_response,
                    "",
                    f"Score: {model['total']}/10 ({score_text})",
                    "",
                    f"Notes: {model['notes']}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    try:
        report = build_report(
            read_jsonl(args.base_responses),
            read_jsonl(args.merged_responses),
            read_json(args.rubric),
            read_json(args.scores),
        )
        args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        args.markdown_output.write_text(
            markdown_report(report),
            encoding="utf-8",
        )
    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
