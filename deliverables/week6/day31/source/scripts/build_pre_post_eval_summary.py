#!/usr/bin/env python3
"""Deterministically summarize hash-bound baseline/post frozen-evaluation files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from day31_harness import sha256_file


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"evaluation JSON object required: {path}")
    return value


def _validate_pair(baseline: dict[str, Any], post: dict[str, Any]) -> None:
    if baseline.get("status") != "completed" or post.get("status") != "completed":
        raise ValueError("both raw evaluations must be completed")
    if baseline.get("model_identity", {}).get("mode") != "baseline" or post.get("model_identity", {}).get("mode") != "post":
        raise ValueError("evaluation modes must be baseline and post")
    if baseline.get("case_count") != post.get("case_count") or not isinstance(baseline.get("case_count"), int):
        raise ValueError("evaluation case counts differ")
    if baseline.get("generation_config") != post.get("generation_config") or baseline.get("protocol") != post.get("protocol"):
        raise ValueError("evaluation generation protocol differs")
    baseline_ids = [case.get("case_id") for case in baseline.get("case_results", [])]
    post_ids = [case.get("case_id") for case in post.get("case_results", [])]
    if len(baseline_ids) != baseline["case_count"] or baseline_ids != post_ids or len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError("evaluation case IDs differ")
    if set(baseline.get("metrics", {})) != set(post.get("metrics", {})):
        raise ValueError("evaluation metric keys differ")
    if baseline.get("applicable_counts") != post.get("applicable_counts"):
        raise ValueError("evaluation applicability differs")


def build_summary(baseline_path: Path, post_path: Path) -> dict[str, Any]:
    baseline, post = _load(baseline_path), _load(post_path)
    _validate_pair(baseline, post)
    metrics = {
        name: {
            "baseline": baseline["metrics"][name],
            "post": post["metrics"][name],
            "delta": post["metrics"][name] - baseline["metrics"][name],
            "applicable_count": baseline["applicable_counts"][name],
        }
        for name in sorted(baseline["metrics"])
    }
    baseline_cases = {case["case_id"]: case for case in baseline["case_results"]}
    post_cases = {case["case_id"]: case for case in post["case_results"]}
    special_ids = (
        "eval-aurora-zero-shipping-calculator-required",
        "eval-code-no-unnecessary-knowledge",
    )
    special_cases = {
        case_id: {
            "baseline_passed": baseline_cases[case_id]["passed"],
            "post_passed": post_cases[case_id]["passed"],
            "baseline_checks": baseline_cases[case_id]["checks"],
            "post_checks": post_cases[case_id]["checks"],
        }
        for case_id in special_ids
        if case_id in baseline_cases
    }
    all_baseline = sum(case.get("passed") is True for case in baseline["case_results"])
    all_post = sum(case.get("passed") is True for case in post["case_results"])
    return {
        "schema_version": "1.1",
        "status": "completed",
        "baseline_sha256": sha256_file(baseline_path),
        "post_sha256": sha256_file(post_path),
        "case_count": baseline["case_count"],
        "frozen_eval_file_sha256": baseline["protocol"].get("frozen_eval_file_sha256"),
        "generation_config": baseline["generation_config"],
        "protocol": baseline["protocol"],
        "baseline_model_identity": baseline["model_identity"],
        "post_model_identity": post["model_identity"],
        "applicable_counts": baseline["applicable_counts"],
        "metrics": metrics,
        "all_checks_passed_cases": {
            "baseline": all_baseline,
            "post": all_post,
            "delta": all_post - all_baseline,
        },
        "special_cases": special_cases,
        "conclusion": {
            f"{name}_improved": values["delta"] > 0
            for name, values in metrics.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = build_summary(args.baseline, args.post)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"pre/post summary error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
