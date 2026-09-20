"""Deterministic Day32 failure taxonomy and prompt-comparison helpers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any


FAILURE_LABELS = (
    "wrong_tool",
    "argument_extraction_error",
    "dead_loop",
    "tool_error_unhandled",
    "premature_answer",
    "final_answer_error",
    "hallucinated_observation",
    "format_parse_error",
    "unknown_failure",
)

PRIMARY_PRIORITY = (
    "format_parse_error",
    "dead_loop",
    "wrong_tool",
    "argument_extraction_error",
    "tool_error_unhandled",
    "premature_answer",
    "final_answer_error",
    "hallucinated_observation",
    "unknown_failure",
)

_ERROR_STATUSES = {"rejected", "not_found", "ambiguous", "error"}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_false(checks: dict[str, object], name: str) -> bool:
    return checks.get(name) is False


def classify_failure(
    frozen_case: dict[str, Any], case_result: dict[str, Any]
) -> dict[str, Any]:
    """Classify one scored trace with one primary label and optional secondary labels."""
    case_id = case_result.get("case_id")
    if case_id != frozen_case.get("id"):
        raise ValueError("case id mismatch")
    checks = case_result.get("checks")
    raw_trace = case_result.get("raw_trace")
    if not isinstance(checks, dict) or not isinstance(raw_trace, dict):
        raise ValueError(f"malformed scored result: {case_id}")
    trace = raw_trace.get("trace")
    if not isinstance(trace, list):
        raise ValueError(f"malformed trace: {case_id}")

    predicates = frozen_case.get("success_predicates", {})
    expected = predicates.get("expected_tool_sequence", [])
    if not isinstance(expected, list) or any(not isinstance(name, str) for name in expected):
        raise ValueError(f"malformed expected tool sequence: {case_id}")
    actual = [row.get("action") for row in trace if isinstance(row, dict)]

    outcomes: list[dict[str, object]] = []
    for row in trace:
        if not isinstance(row, dict):
            continue
        observation = row.get("observation")
        if isinstance(observation, dict):
            outcomes.append(
                {
                    "status": observation.get("status"),
                    "error_code": observation.get("error_code"),
                }
            )

    final_answer = str(raw_trace.get("final_answer", ""))
    labels: set[str] = set()
    if "<tool_call>" in final_answer or "</tool_call>" in final_answer:
        labels.add("format_parse_error")
    if _is_false(checks, "no_loops"):
        labels.add("dead_loop")
    if _is_false(checks, "first_tool_choice") or _is_false(
        checks, "full_tool_sequence"
    ):
        labels.add("wrong_tool")
    if _is_false(checks, "argument_correctness"):
        labels.add("argument_extraction_error")
    if _is_false(checks, "no_hallucinated_observations") or _is_false(
        checks, "observation_integrity"
    ):
        labels.add("hallucinated_observation")
    if _is_false(checks, "semantic_correctness") or _is_false(
        checks, "final_answer_grounded"
    ):
        labels.add("final_answer_error")
    if _is_false(checks, "completion"):
        if any(outcome["status"] in _ERROR_STATUSES for outcome in outcomes):
            labels.add("tool_error_unhandled")
        elif not labels and len(actual) < len(expected):
            labels.add("premature_answer")
        elif not labels:
            labels.add("final_answer_error")

    passed = case_result.get("passed") is True
    if not passed and not labels:
        labels.add("unknown_failure")
    primary = next((label for label in PRIMARY_PRIORITY if label in labels), None)
    secondary = [label for label in PRIMARY_PRIORITY if label in labels and label != primary]
    return {
        "case_id": case_id,
        "passed": passed,
        "primary_label": primary,
        "secondary_labels": secondary,
        "expected_tool_sequence": expected,
        "actual_tool_sequence": actual,
        "observation_outcomes": outcomes,
        "failed_checks": [name for name, value in checks.items() if value is False],
    }


def analyze_evaluation(
    frozen_cases: list[dict[str, Any]], report: dict[str, Any]
) -> dict[str, Any]:
    """Recompute row-level failure labels and aggregate counts from one evaluation."""
    if report.get("status") != "completed":
        raise ValueError("evaluation report is not completed")
    case_results = report.get("case_results")
    if not isinstance(case_results, list):
        raise ValueError("evaluation case_results missing")
    if report.get("case_count") != len(frozen_cases) or len(case_results) != len(
        frozen_cases
    ):
        raise ValueError("evaluation case count mismatch")
    frozen_by_id = {case.get("id"): case for case in frozen_cases}
    if len(frozen_by_id) != len(frozen_cases) or None in frozen_by_id:
        raise ValueError("frozen case ids must be unique strings")
    result_ids = [result.get("case_id") for result in case_results]
    if len(set(result_ids)) != len(result_ids) or set(result_ids) != set(frozen_by_id):
        raise ValueError("evaluation result ids do not match frozen cases")

    rows = [classify_failure(frozen_by_id[result["case_id"]], result) for result in case_results]
    primary_counts = Counter(
        row["primary_label"] for row in rows if row["primary_label"] is not None
    )
    occurrence_counts: Counter[str] = Counter()
    for row in rows:
        if row["primary_label"] is not None:
            occurrence_counts[row["primary_label"]] += 1
        occurrence_counts.update(row["secondary_labels"])

    protocol = report.get("protocol") if isinstance(report.get("protocol"), dict) else {}
    frozen_sha = protocol.get("frozen_eval_file_sha256") or _canonical_sha256(
        frozen_cases
    )
    model_identity = report.get("model_identity", {})
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("evaluation metrics missing")
    return {
        "schema_version": "1.0",
        "status": "completed",
        "case_count": len(rows),
        "frozen_eval_sha256": frozen_sha,
        "generation_config": report.get("generation_config", {}),
        "model_identity_sha256": _canonical_sha256(model_identity),
        "metrics": metrics,
        "unique_failed_cases": sum(not row["passed"] for row in rows),
        "primary_label_counts": {
            label: primary_counts.get(label, 0) for label in FAILURE_LABELS
        },
        "label_occurrence_counts": {
            label: occurrence_counts.get(label, 0) for label in FAILURE_LABELS
        },
        "case_analysis": rows,
    }


def compare_evaluations(
    baseline: dict[str, Any], optimized: dict[str, Any]
) -> dict[str, Any]:
    """Compare prompt-only runs while rejecting changed model or evaluation inputs."""
    if baseline.get("case_count") != optimized.get("case_count"):
        raise ValueError("case count changed")
    if baseline.get("frozen_eval_sha256") != optimized.get("frozen_eval_sha256"):
        raise ValueError("frozen evaluation changed")
    if baseline.get("generation_config") != optimized.get("generation_config"):
        raise ValueError("generation protocol changed")
    if baseline.get("model_identity_sha256") != optimized.get("model_identity_sha256"):
        raise ValueError("model identity changed")

    baseline_rows = {row["case_id"]: row for row in baseline.get("case_analysis", [])}
    optimized_rows = {row["case_id"]: row for row in optimized.get("case_analysis", [])}
    if set(baseline_rows) != set(optimized_rows):
        raise ValueError("case result set changed")
    improved = sorted(
        case_id
        for case_id in baseline_rows
        if not baseline_rows[case_id]["passed"] and optimized_rows[case_id]["passed"]
    )
    regressions = sorted(
        case_id
        for case_id in baseline_rows
        if baseline_rows[case_id]["passed"] and not optimized_rows[case_id]["passed"]
    )
    metric_names = sorted(
        set(baseline.get("metrics", {})) | set(optimized.get("metrics", {}))
    )
    metric_deltas = {
        name: optimized["metrics"].get(name, 0) - baseline["metrics"].get(name, 0)
        for name in metric_names
    }
    return {
        "schema_version": "1.0",
        "status": "completed",
        "case_count": baseline["case_count"],
        "frozen_eval_sha256": baseline["frozen_eval_sha256"],
        "generation_config": baseline["generation_config"],
        "model_identity_sha256": baseline["model_identity_sha256"],
        "baseline_metrics": baseline["metrics"],
        "optimized_metrics": optimized["metrics"],
        "metric_deltas": metric_deltas,
        "baseline_unique_failed_cases": baseline["unique_failed_cases"],
        "optimized_unique_failed_cases": optimized["unique_failed_cases"],
        "improved_cases": improved,
        "regression_cases": regressions,
    }
