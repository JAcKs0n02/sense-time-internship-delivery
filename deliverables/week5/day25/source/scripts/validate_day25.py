#!/usr/bin/env python3
"""Fail-closed validation for the complete Day25 hallucination evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_CATEGORIES = {
    "nonexistent_object",
    "wrong_text_or_number",
    "wrong_attribute",
    "wrong_spatial_relation",
    "wrong_action_or_intent",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_binary(value: Any) -> int | None:
    if value in {0, "0"}:
        return 0
    if value in {1, "1"}:
        return 1
    return None


def validate_records(
    *,
    cases: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    reviewed_records: list[dict[str, Any]],
    summary: dict[str, Any],
    expected_count: int = 10,
) -> dict[str, Any]:
    errors: list[str] = []
    check_errors: dict[str, list[str]] = {
        "record_count_and_order": [],
        "preregistered_case_design": [],
        "successful_raw_outputs": [],
        "frozen_identity": [],
        "binary_review": [],
        "raw_immutability": [],
        "metric_formula": [],
        "grouped_metrics": [],
    }

    def add(check: str, message: str) -> None:
        item = f"{check}: {message}"
        check_errors[check].append(item)
        errors.append(item)

    if not (len(cases) == len(raw_records) == len(reviewed_records) == expected_count):
        add(
            "record_count_and_order",
            f"cases={len(cases)}, raw={len(raw_records)}, reviewed={len(reviewed_records)}, expected={expected_count}",
        )
    case_ids = [row.get("case_id") for row in cases]
    raw_ids = [row.get("case_id") for row in raw_records]
    reviewed_ids = [row.get("case_id") for row in reviewed_records]
    if len(set(case_ids)) != expected_count or any(not value for value in case_ids):
        add("record_count_and_order", "case IDs are not unique non-empty values")
    if raw_ids != case_ids or reviewed_ids != case_ids:
        add("record_count_and_order", "raw/reviewed order differs from frozen case order")

    if expected_count == 10:
        image_counts: dict[str, int] = {}
        for row in cases:
            image_counts[row.get("image_id", "")] = image_counts.get(row.get("image_id", ""), 0) + 1
        if sorted(image_counts.values()) != [2, 2, 2, 2, 2]:
            add("preregistered_case_design", f"expected two cases per image: {image_counts}")
        categories = {row.get("error_category") for row in cases}
        if categories != EXPECTED_CATEGORIES:
            add("preregistered_case_design", f"category coverage mismatch: {categories}")

    cases_by_id = {row.get("case_id"): row for row in cases}
    raw_by_id = {row.get("case_id"): row for row in raw_records}
    hallucination_labels: list[int] = []
    task_success_labels: list[int] = []
    for raw in raw_records:
        case_id = raw.get("case_id")
        case = cases_by_id.get(case_id, {})
        if raw.get("status") != "PASS" or not str(raw.get("raw_response", "")).strip():
            add("successful_raw_outputs", f"{case_id} is not a successful non-empty output")
        for field in ("input_tokens", "output_tokens", "latency_seconds", "peak_gpu_memory_mib"):
            value = raw.get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                add("successful_raw_outputs", f"{case_id} invalid {field}: {value}")
        for field in ("image_sha256", "generation_config_sha256", "model_revision"):
            if raw.get(field) != case.get(field):
                add("frozen_identity", f"{case_id} {field} changed")

    for reviewed in reviewed_records:
        case_id = reviewed.get("case_id")
        hallucination = as_binary(reviewed.get("hallucination"))
        task_success = as_binary(reviewed.get("task_success"))
        if hallucination is None:
            add("binary_review", f"{case_id} hallucination is not binary")
        else:
            hallucination_labels.append(hallucination)
        if task_success is None:
            add("binary_review", f"{case_id} task_success is not binary")
        else:
            task_success_labels.append(task_success)
        if not str(reviewed.get("reviewer_reason", "")).strip():
            add("binary_review", f"{case_id} reviewer_reason is empty")
        raw = raw_by_id.get(case_id)
        if raw is not None and reviewed.get("raw_response") != raw.get("raw_response"):
            add("raw_immutability", f"{case_id} raw response changed during scoring")

    computed_count = sum(hallucination_labels)
    computed_rate = computed_count / expected_count
    if summary.get("sample_count") != expected_count or summary.get("fixed_denominator") != expected_count:
        add("metric_formula", "summary does not preserve the fixed denominator")
    if summary.get("hallucination_count") != computed_count:
        add(
            "metric_formula",
            f"hallucination_count={summary.get('hallucination_count')} but labels sum to {computed_count}",
        )
    if summary.get("hallucination_rate") != round(computed_rate, 6):
        add(
            "metric_formula",
            f"hallucination_rate={summary.get('hallucination_rate')} but expected {round(computed_rate, 6)}",
        )
    if summary.get("hallucination_percentage") != round(computed_rate * 100, 2):
        add("metric_formula", "hallucination_percentage does not match the binary labels")
    if len(task_success_labels) == expected_count and summary.get("task_success_count") != sum(task_success_labels):
        add("metric_formula", "task_success_count does not match the binary labels")

    grouped_total = 0
    grouped_hallucinations = 0
    for metrics in summary.get("by_error_category", {}).values():
        grouped_total += metrics.get("sample_count", 0)
        grouped_hallucinations += metrics.get("hallucination_count", 0)
    if grouped_total != expected_count or grouped_hallucinations != computed_count:
        add("grouped_metrics", "category totals do not reconcile to the primary metric")

    return {
        "schema_version": "1.0",
        "status": "FAIL" if errors else "PASS",
        "summary": {
            "expected_records": expected_count,
            "case_records": len(cases),
            "raw_records": len(raw_records),
            "reviewed_records": len(reviewed_records),
            "successful_raw_records": sum(row.get("status") == "PASS" for row in raw_records),
            "hallucination_count": computed_count,
            "hallucination_rate": round(computed_rate, 6),
        },
        "checks": [
            {"check": name, "status": "FAIL" if values else "PASS"}
            for name, values in check_errors.items()
        ],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--reviewed-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases_payload = read_json(args.cases)
    freeze = read_json(args.freeze_manifest)
    summary = read_json(args.summary)
    result = validate_records(
        cases=cases_payload.get("records", []),
        raw_records=read_jsonl(args.raw_jsonl),
        reviewed_records=read_csv(args.reviewed_csv),
        summary=summary,
    )

    artifact_errors: list[str] = []
    if freeze.get("cases_sha256") != sha256(args.cases):
        artifact_errors.append("artifact_integrity: frozen cases SHA256 mismatch")
    if freeze.get("generation_config_sha256") != sha256(args.generation_config):
        artifact_errors.append("artifact_integrity: generation config SHA256 mismatch")
    if freeze.get("post_inference_case_edits_allowed") is not False:
        artifact_errors.append("artifact_integrity: case mutation policy is not fail-closed")
    if not args.report.is_file() or not args.report.read_text(encoding="utf-8").strip():
        artifact_errors.append("teacher_report: report is missing or empty")
    result["checks"].extend(
        [
            {
                "check": "artifact_integrity",
                "status": "FAIL" if any(error.startswith("artifact_integrity") for error in artifact_errors) else "PASS",
            },
            {
                "check": "teacher_report",
                "status": "FAIL" if any(error.startswith("teacher_report") for error in artifact_errors) else "PASS",
            },
        ]
    )
    result["errors"].extend(artifact_errors)
    result["status"] = "FAIL" if result["errors"] else "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
