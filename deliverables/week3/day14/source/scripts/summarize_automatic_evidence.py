#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


SUMMARY_FIELDS = [
    "candidate_id",
    "category",
    "question_count",
    "checked_count",
    "passed_count",
    "failed_check_count",
    "code_tests_passed_count",
    "code_execution_disabled_count",
    "not_evaluable_count",
]


def summarize_automatic_evidence(
    payload: dict[str, Any],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    if payload.get("ranking_input") is not False:
        raise ValueError("automatic evidence must never be a ranking input")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("automatic evidence records must be a list")
    categories = {str(item["id"]): str(item["category"]) for item in questions}
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    candidates: set[str] = set()
    for record in records:
        candidate_id = str(record["candidate_id"])
        question_id = str(record["question_id"])
        if question_id not in categories:
            raise ValueError(f"unknown question id: {question_id}")
        key = (candidate_id, question_id)
        if key in seen:
            raise ValueError(f"duplicate automatic evidence record: {candidate_id}/{question_id}")
        seen.add(key)
        candidates.add(candidate_id)
        group_key = (candidate_id, categories[question_id])
        row = grouped.setdefault(
            group_key,
            {
                "candidate_id": candidate_id,
                "category": categories[question_id],
                "question_count": 0,
                "checked_count": 0,
                "passed_count": 0,
                "failed_check_count": 0,
                "code_tests_passed_count": 0,
                "code_execution_disabled_count": 0,
                "not_evaluable_count": 0,
            },
        )
        row["question_count"] += 1
        evidence = record.get("automatic_evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"invalid automatic evidence: {candidate_id}/{question_id}")
        check_status = evidence.get("status")
        code_status = evidence.get("code_execution_status")
        if check_status == "checked":
            row["checked_count"] += 1
            if evidence.get("passed") is True:
                row["passed_count"] += 1
            else:
                row["failed_check_count"] += 1
        elif code_status == "completed":
            row["checked_count"] += 1
            if evidence.get("tests_passed") is True:
                row["passed_count"] += 1
                row["code_tests_passed_count"] += 1
            else:
                row["failed_check_count"] += 1
        elif code_status == "disabled_no_sandbox":
            row["code_execution_disabled_count"] += 1
        else:
            row["not_evaluable_count"] += 1
    return {
        "schema_version": 1,
        "ranking_input": False,
        "candidate_count": len(candidates),
        "record_count": len(records),
        "rows": [grouped[key] for key in sorted(grouped)],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Day 14 automatic evidence without ranking models."
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = json.loads(args.evidence.read_text(encoding="utf-8"))
        question_payload = json.loads(args.questions.read_text(encoding="utf-8"))
        questions = question_payload.get("questions", question_payload)
        summary = summarize_automatic_evidence(payload, questions)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_csv(args.csv_output, summary["rows"])
    except (OSError, csv.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
