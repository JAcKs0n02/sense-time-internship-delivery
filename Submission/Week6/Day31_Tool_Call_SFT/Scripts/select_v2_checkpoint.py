#!/usr/bin/env python3
"""Select one raw-mode checkpoint from compatible Week6 v2 development runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


THRESHOLDS = {
    "strict_success": 0.90,
    "first_tool_choice": 0.95,
    "full_tool_sequence": 0.90,
    "argument_correctness": 0.95,
    "semantic_correctness": 0.90,
    "observation_integrity": 1.00,
    "no_loops": 1.00,
    "final_answer_grounded": 0.95,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_candidate(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol = report.get("protocol", {})
    identity = report.get("model_identity", {})
    cases = report.get("case_results", [])
    if (
        report.get("schema_version") != "2.0"
        or report.get("status") != "completed"
        or identity.get("mode") != "post"
        or protocol.get("evaluation_profile") != "v2"
        or protocol.get("split_role") != "development"
        or protocol.get("policy_mode") != "raw"
        or report.get("case_count") != len(cases)
        or not cases
    ):
        raise ValueError(f"not a completed raw v2 development result: {path}")
    metrics = report.get("metrics", {})
    selection_metrics = {
        "strict_success": sum(case.get("passed") is True for case in cases) / len(cases),
        "argument_correctness": float(metrics.get("argument_correctness", 0.0)),
        "full_tool_sequence": float(metrics.get("full_tool_sequence", 0.0)),
        "semantic_correctness": float(metrics.get("semantic_correctness", 0.0)),
    }
    return report, selection_metrics


def select_checkpoint(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one --candidate is required")
    candidates: list[dict[str, Any]] = []
    reference: tuple[Any, Any, Any] | None = None
    for path in paths:
        report, selection_metrics = _load_candidate(path)
        compatibility = (
            report["protocol"],
            report.get("generation_config"),
            [case.get("case_id") for case in report["case_results"]],
        )
        if reference is None:
            reference = compatibility
        elif compatibility != reference:
            raise ValueError("candidate evaluation protocols, generation settings, or cases differ")
        identity = report["model_identity"]
        candidates.append(
            {
                "result_path": str(path.resolve()),
                "result_sha256": _sha256(path),
                "checkpoint": identity.get("adapter_path"),
                "adapter_model_sha256": identity.get("adapter_model_sha256"),
                "selection_metrics": selection_metrics,
                "all_metrics": report.get("metrics", {}),
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["selection_metrics"]["strict_success"],
            item["selection_metrics"]["argument_correctness"],
            item["selection_metrics"]["full_tool_sequence"],
            item["selection_metrics"]["semantic_correctness"],
            str(item["checkpoint"]),
        ),
    )
    measured = {"strict_success": selected["selection_metrics"]["strict_success"], **selected["all_metrics"]}
    checks = {
        name: float(measured.get(name, 0.0)) >= threshold
        for name, threshold in THRESHOLDS.items()
    }
    return {
        "schema_version": "2.0",
        "status": "completed",
        "selection_split": "development",
        "policy_mode": "raw",
        "ranking_order": [
            "strict_success",
            "argument_correctness",
            "full_tool_sequence",
            "semantic_correctness",
        ],
        "selected_checkpoint": selected["checkpoint"],
        "selected_adapter_sha256": selected["adapter_model_sha256"],
        "selection_metrics": selected["selection_metrics"],
        "acceptance_gate": {
            "thresholds": THRESHOLDS,
            "checks": checks,
            "passed": all(checks.values()),
        },
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = select_checkpoint(args.candidate)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
