#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_results(
    output_dir: Path,
    *,
    expected_candidates: int = 10,
    expected_questions: int = 20,
) -> dict[str, Any]:
    manifest = json.loads(
        (output_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    responses = read_jsonl(output_dir / "all_responses.jsonl")
    automatic = json.loads(
        (output_dir / "automatic_evidence.json").read_text(encoding="utf-8")
    )
    automatic_records = automatic.get("records")
    if not isinstance(automatic_records, list):
        raise ValueError("automatic evidence records must be a list")

    errors: list[str] = []
    expected_total = expected_candidates * expected_questions
    if manifest.get("status") != "completed":
        errors.append(f"manifest status is not completed: {manifest.get('status')}")
    if manifest.get("candidate_count") != expected_candidates:
        errors.append("manifest candidate count differs from the frozen gate")
    if manifest.get("question_count") != expected_questions:
        errors.append("manifest question count differs from the frozen gate")
    if len(responses) != expected_total:
        errors.append(f"expected {expected_total} terminal records, found {len(responses)}")
    if manifest.get("terminal_record_count") != len(responses):
        errors.append("manifest terminal record count differs from JSONL")
    failed_count = sum(row.get("status") != "completed" for row in responses)
    if failed_count:
        errors.append(f"found {failed_count} failed terminal records")
    if manifest.get("failed_record_count") != failed_count:
        errors.append("manifest failed record count differs from JSONL")

    response_keys = [
        (str(row.get("candidate_id")), str(row.get("question_id")))
        for row in responses
    ]
    duplicate_keys = sorted(key for key, count in Counter(response_keys).items() if count != 1)
    if duplicate_keys:
        errors.append(f"duplicate candidate/question pairs: {duplicate_keys}")
    candidate_counts = Counter(candidate for candidate, _question in response_keys)
    if len(candidate_counts) != expected_candidates:
        errors.append(f"expected {expected_candidates} candidates, found {len(candidate_counts)}")
    for candidate, count in sorted(candidate_counts.items()):
        if count != expected_questions:
            errors.append(
                f"candidate {candidate} has {count} records; expected {expected_questions}"
            )

    hashes: dict[str, set[str]] = defaultdict(set)
    for row in responses:
        hashes[str(row.get("question_id"))].add(str(row.get("question_sha256")))
    inconsistent_hashes = sorted(question for question, values in hashes.items() if len(values) != 1)
    if inconsistent_hashes:
        errors.append(f"inconsistent question hashes: {inconsistent_hashes}")

    if automatic.get("ranking_input") is not False:
        errors.append("automatic evidence is not explicitly excluded from ranking")
    evidence_keys = [
        (str(row.get("candidate_id")), str(row.get("question_id")))
        for row in automatic_records
    ]
    if len(automatic_records) != expected_total:
        errors.append(
            f"expected {expected_total} automatic evidence records, "
            f"found {len(automatic_records)}"
        )
    if Counter(evidence_keys) != Counter(response_keys):
        errors.append("automatic evidence keys differ from response keys")

    return {
        "schema_version": 1,
        "valid": not errors,
        "expected_candidate_count": expected_candidates,
        "expected_question_count": expected_questions,
        "terminal_record_count": len(responses),
        "failed_record_count": failed_count,
        "automatic_evidence_count": len(automatic_records),
        "candidate_question_counts": dict(sorted(candidate_counts.items())),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the closed Day 14 result matrix.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=10)
    parser.add_argument("--expected-questions", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_results(
            args.output_dir,
            expected_candidates=args.expected_candidates,
            expected_questions=args.expected_questions,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
