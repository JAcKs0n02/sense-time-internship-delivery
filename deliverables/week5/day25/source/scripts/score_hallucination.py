#!/usr/bin/env python3
"""Merge frozen cases, raw outputs, and manual annotations; compute Day25 metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REVIEWER_TYPE = "codex_assisted_visual_review"
RESPONSE_TYPES = {
    "accepted_false_premise",
    "corrected_false_premise",
    "uncertain_but_grounded",
    "refusal_or_nonresponsive",
    "mixed_with_new_hallucination",
}
HALLUCINATION_BASES = {
    "none",
    "accepted_false_premise",
    "new_unsupported_claim",
    "both",
}
CSV_FIELDS = [
    "case_id",
    "execution_index",
    "image_id",
    "error_category",
    "question",
    "fake_answer",
    "ground_truth",
    "expected_behavior",
    "raw_response",
    "hallucination",
    "task_success",
    "response_type",
    "hallucination_basis",
    "evidence_excerpt",
    "reviewer_reason",
    "reviewer_type",
    "input_tokens",
    "output_tokens",
    "latency_seconds",
    "peak_gpu_memory_mib",
    "model_revision",
    "generation_config_sha256",
    "image_sha256",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"raw JSONL line {line_number} must be an object")
        rows.append(row)
    return rows


def ordered_ids(rows: list[dict[str, Any]]) -> list[Any]:
    return [row.get("case_id") for row in rows]


def require_ordered_inputs(
    cases: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    expected_count: int,
) -> None:
    if not (len(cases) == len(raw_records) == len(annotations) == expected_count):
        raise ValueError(
            "case/raw/annotation order requires identical fixed counts: "
            f"cases={len(cases)}, raw={len(raw_records)}, annotations={len(annotations)}, "
            f"expected={expected_count}"
        )
    ids = ordered_ids(cases)
    if len(set(ids)) != expected_count or any(not value for value in ids):
        raise ValueError("case IDs must be unique and non-empty")
    if ordered_ids(raw_records) != ids or ordered_ids(annotations) != ids:
        raise ValueError("case/raw/annotation order must match the frozen case order")


def validate_annotation(row: dict[str, Any]) -> None:
    if row.get("hallucination") not in {0, 1}:
        raise ValueError(f"{row.get('case_id')} hallucination must be binary 0 or 1")
    if row.get("task_success") not in {0, 1}:
        raise ValueError(f"{row.get('case_id')} task_success must be binary 0 or 1")
    if row.get("response_type") not in RESPONSE_TYPES:
        raise ValueError(f"{row.get('case_id')} invalid response_type")
    if row.get("hallucination_basis") not in HALLUCINATION_BASES:
        raise ValueError(f"{row.get('case_id')} invalid hallucination_basis")
    if row.get("reviewer_type") != REVIEWER_TYPE:
        raise ValueError(f"{row.get('case_id')} reviewer_type must be {REVIEWER_TYPE}")
    if not str(row.get("reviewer_reason", "")).strip():
        raise ValueError(f"{row.get('case_id')} reviewer_reason cannot be empty")
    if not str(row.get("evidence_excerpt", "")).strip():
        raise ValueError(f"{row.get('case_id')} evidence_excerpt cannot be empty")
    hallucination = row["hallucination"]
    basis = row["hallucination_basis"]
    if hallucination == 0 and basis != "none":
        raise ValueError(f"{row.get('case_id')} non-hallucination must use basis=none")
    if hallucination == 1 and basis == "none":
        raise ValueError(f"{row.get('case_id')} hallucination requires a non-none basis")


def score_records(
    cases: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    expected_count: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_ordered_inputs(cases, raw_records, annotations, expected_count)
    reviewed: list[dict[str, Any]] = []
    for case, raw, annotation in zip(cases, raw_records, annotations):
        if raw.get("status") != "PASS" or not str(raw.get("raw_response", "")).strip():
            raise ValueError(f"{case['case_id']} has no valid first successful raw response")
        if raw.get("raw_response") != annotation.get("raw_response", raw.get("raw_response")):
            raise ValueError(f"{case['case_id']} raw response changed during annotation")
        validate_annotation(annotation)
        reviewed.append({**case, **raw, **annotation, "raw_response": raw["raw_response"]})

    hallucination_count = sum(row["hallucination"] for row in reviewed)
    task_success_count = sum(row["task_success"] for row in reviewed)
    correction_count = sum(
        row["response_type"] == "corrected_false_premise" for row in reviewed
    )
    refusal_count = sum(
        row["response_type"] == "refusal_or_nonresponsive" for row in reviewed
    )

    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviewed:
        category_rows[row["error_category"]].append(row)
        image_rows[row["image_id"]].append(row)

    def grouped_metrics(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key in sorted(groups):
            rows = groups[key]
            count = sum(row["hallucination"] for row in rows)
            output[key] = {
                "sample_count": len(rows),
                "hallucination_count": count,
                "hallucination_rate": round(count / len(rows), 6),
                "hallucination_percentage": round(count / len(rows) * 100, 2),
            }
        return output

    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "metric_scope": "descriptive rate on the frozen ten-case Day25 set",
        "sample_count": expected_count,
        "fixed_denominator": expected_count,
        "hallucination_count": hallucination_count,
        "hallucination_rate": round(hallucination_count / expected_count, 6),
        "hallucination_percentage": round(
            hallucination_count / expected_count * 100, 2
        ),
        "non_hallucination_count": expected_count - hallucination_count,
        "correction_count": correction_count,
        "correction_rate": round(correction_count / expected_count, 6),
        "task_success_count": task_success_count,
        "task_success_rate": round(task_success_count / expected_count, 6),
        "refusal_count": refusal_count,
        "response_type_counts": dict(
            sorted(Counter(row["response_type"] for row in reviewed).items())
        ),
        "by_error_category": grouped_metrics(category_rows),
        "by_image": grouped_metrics(image_rows),
    }
    return reviewed, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    cases_payload = read_json(args.cases)
    annotation_payload = read_json(args.annotations)
    cases = cases_payload.get("records", [])
    annotations = (
        annotation_payload.get("records", [])
        if isinstance(annotation_payload, dict)
        else annotation_payload
    )
    reviewed, summary = score_records(cases, read_jsonl(args.raw_jsonl), annotations)
    write_csv(args.output_csv, reviewed)
    write_json(args.output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
