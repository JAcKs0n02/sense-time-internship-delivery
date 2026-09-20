#!/usr/bin/env python3
"""Validate the completed Day32 prompt-only evaluation and report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = SOURCE_ROOT / "results"


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=RESULTS / "v2_analysis/baseline_error_analysis.json")
    parser.add_argument("--optimized", type=Path, default=RESULTS / "v2_analysis/optimized_error_analysis.json")
    parser.add_argument("--comparison", type=Path, default=RESULTS / "v2_analysis/prompt_comparison.json")
    parser.add_argument("--raw-optimized", type=Path, default=RESULTS / "v2_dev_prompt_ablation.json")
    parser.add_argument("--prompt-manifest", type=Path, default=SOURCE_ROOT / "configs/prompt_manifest_v2.json")
    parser.add_argument("--report", type=Path, default=RESULTS / "AGENT_ERROR_ANALYSIS.md")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    facts: dict[str, object] = {}
    try:
        baseline = _read(args.baseline)
        optimized = _read(args.optimized)
        comparison = _read(args.comparison)
        raw = _read(args.raw_optimized)
        prompt = _read(args.prompt_manifest)
        report = args.report.read_text(encoding="utf-8")

        checks["completed_40_case_runs"] = all(
            value.get("status") == "completed" and value.get("case_count") == 40
            for value in (baseline, optimized, comparison, raw)
        )
        checks["same_frozen_eval"] = (
            baseline.get("frozen_eval_sha256")
            == optimized.get("frozen_eval_sha256")
            == comparison.get("frozen_eval_sha256")
        )
        checks["same_generation"] = (
            baseline.get("generation_config")
            == optimized.get("generation_config")
            == comparison.get("generation_config")
        )
        checks["same_model_identity"] = (
            baseline.get("model_identity_sha256")
            == optimized.get("model_identity_sha256")
            == comparison.get("model_identity_sha256")
        )
        checks["prompt_only_ablation"] = (
            prompt.get("change_scope") == "single_rule_prompt_ablation"
            and raw.get("prompt_manifest") == prompt
            and prompt.get("final_test_used_for_selection") is False
        )
        required = {"wrong_tool", "argument_extraction_error", "dead_loop"}
        checks["teacher_failure_categories_present"] = all(
            required <= set(value.get("primary_label_counts", {}))
            for value in (baseline, optimized)
        )
        regressions = comparison.get("regression_cases")
        checks["regressions_reported"] = isinstance(regressions, list) and bool(regressions)
        raw_cases = raw.get("case_results")
        checks["raw_trace_complete"] = isinstance(raw_cases, list) and len(raw_cases) == 40 and all(
            isinstance(row, dict)
            and isinstance(row.get("raw_trace"), dict)
            and isinstance(row["raw_trace"].get("trace"), list)
            and "final_answer" in row["raw_trace"]
            for row in raw_cases
        )
        baseline_failed = int(baseline.get("unique_failed_cases", -1))
        optimized_failed = int(optimized.get("unique_failed_cases", -1))
        recommended = "main" if optimized_failed > baseline_failed else "ablation"
        checks["report_binding"] = all(
            token in report
            for token in (
                f"40 条中 18 条严格通过、{baseline_failed} 条失败",
                "input-fidelity ablation 通过 14 条",
                "最终采用 main Prompt",
                "死循环为 0",
            )
        )
        facts = {
            "baseline_failed_cases": baseline_failed,
            "optimized_failed_cases": optimized_failed,
            "improved_cases": comparison.get("improved_cases", []),
            "regression_cases": regressions or [],
            "recommended_prompt": recommended,
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        checks["inputs_readable"] = False
        facts["error"] = str(error)

    failed = [name for name, passed in checks.items() if not passed]
    receipt = {
        "schema_version": "1.0",
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "facts": facts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
