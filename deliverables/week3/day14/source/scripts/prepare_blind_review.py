#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any


SCORE_FIELDS = ["accuracy", "completeness", "logic", "safety", "format", "reason"]


def prompt_text(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    return "\n\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    )


def prepare_blind_records(
    responses: list[dict[str, Any]],
    *,
    seed: int,
    question_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = question_metadata or {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for response in responses:
        grouped.setdefault(str(response["question_id"]), []).append(response)
    randomizer = random.Random(seed)
    blind_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for question_id in sorted(grouped):
        candidates = list(grouped[question_id])
        randomizer.shuffle(candidates)
        for position, response in enumerate(candidates, start=1):
            review_id = hashlib.sha256(
                f"{seed}:{question_id}:{position}".encode("utf-8")
            ).hexdigest()[:16]
            question_details = metadata.get(question_id, {})
            blind_rows.append(
                {
                    "review_id": review_id,
                    "question_id": question_id,
                    "candidate_label": f"Candidate-{position:02d}",
                    "prompt": prompt_text(response.get("messages")),
                    "reference": json.dumps(
                        question_details.get("reference", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "human_scoring_notes": question_details.get(
                        "human_scoring_notes",
                        "",
                    ),
                    "response": response.get("raw_response"),
                    "response_status": response.get("status"),
                }
            )
            mapping_rows.append(
                {
                    "review_id": review_id,
                    "question_id": question_id,
                    "candidate_id": response["candidate_id"],
                }
            )
    return blind_rows, {
        "schema_version": 1,
        "blinding_seed": seed,
        "records": mapping_rows,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_reviewer_csv(path: Path, rows: list[dict[str, Any]], reviewer_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["review_id", "reviewer_id", "question_id", "candidate_label", "prompt", "reference", "human_scoring_notes", "response", "response_status", *SCORE_FIELDS]
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "reviewer_id": reviewer_id, **{field: "" for field in SCORE_FIELDS}})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic blind-review packages.")
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--reviewer-1", type=Path, required=True)
    parser.add_argument("--reviewer-2", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        question_payload = json.loads(args.questions.read_text(encoding="utf-8"))
        question_rows = question_payload.get("questions", question_payload)
        question_metadata = {item["id"]: item for item in question_rows}
        blind_rows, mapping = prepare_blind_records(
            read_jsonl(args.responses),
            seed=args.seed,
            question_metadata=question_metadata,
        )
        write_reviewer_csv(args.reviewer_1, blind_rows, "reviewer_1")
        write_reviewer_csv(args.reviewer_2, blind_rows, "reviewer_2")
        args.mapping.parent.mkdir(parents=True, exist_ok=True)
        args.mapping.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
