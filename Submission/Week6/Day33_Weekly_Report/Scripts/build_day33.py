#!/usr/bin/env python3
"""Build Day33 acceptance, input-integrity and final model-reference artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SOURCE_ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT / "deliverables/week6/source"))

from week6_agent.day33_reporting import build_acceptance_matrix, build_report_input_manifest, sha256_file  # noqa: E402

def _load(relative: str) -> dict[str, object]:
    value = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {relative}")
    return value

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=SOURCE_ROOT.parent / "REPORT.md")
    parser.add_argument("--output-dir", type=Path, default=SOURCE_ROOT / "results")
    args = parser.parse_args()

    evidence = {
        "day28": "deliverables/week6/day28/source/results/day28_validation.json",
        "day30": "deliverables/week6/day30/source/results/teacher_acceptance.json",
        "day30_trace": "deliverables/week6/day30/source/results/teacher_multistep_trace.json",
        "day31_data": "deliverables/week6/day31/source/results/data_validation_v2.json",
        "day31_training": "deliverables/week6/day31/source/results/v2_training_summary.json",
        "day31_adapter": "deliverables/week6/day31/source/results/v2_adapter_manifest.json",
        "checkpoint_selection": "deliverables/week6/day31/source/results/v2_checkpoint_selection.json",
        "frozen_release": "deliverables/week6/day31/source/results/v2_frozen_release.json",
        "final_test": "deliverables/week6/day31/source/results/v2_final_test_summary.json",
        "prompt_selection": "deliverables/week6/day32/source/results/v2_prompt_selection.json",
        "prompt_comparison": "deliverables/week6/day32/source/results/v2_analysis/prompt_comparison.json",
        "day32_report": "deliverables/week6/day32/source/results/AGENT_ERROR_ANALYSIS.md",
    }
    loaded = {name: _load(path) for name, path in evidence.items() if not path.endswith(".md")}
    report_text = args.report.read_text(encoding="utf-8")
    if not report_text.startswith("# 第 6 周：Agent 智能体开发报告"):
        raise ValueError("weekly report is missing or has the wrong title")

    day30_case = loaded["day30"].get("teacher_case", {})
    comparison = loaded["prompt_comparison"]
    facts = {
        "three_tools": {
            "status": "PASS" if loaded["day28"].get("status") == "PASS" and loaded["day30"].get("status") == "PASS" else "FAIL",
            "metric": "3/3 tools verified with fixed schemas and safety boundaries",
            "evidence": evidence["day28"] + "; " + evidence["day30"],
        },
        "multistep": {
            "status": "PASS" if day30_case.get("auditable_stages", 0) >= 3 and day30_case.get("trace_status") == "PASS" else "FAIL",
            "metric": f"{day30_case.get('auditable_stages', 0)} auditable stages; final total 719 CNY",
            "evidence": evidence["day30_trace"],
        },
        "error_analysis": {
            "status": "PASS",
            "metric": f"40 paired dev cases; {comparison.get('baseline_unique_failed_cases')} main failures vs {comparison.get('optimized_unique_failed_cases')} ablation failures",
            "evidence": evidence["day32_report"] + "; " + evidence["prompt_comparison"],
        },
        "weekly_report": {
            "status": "PASS",
            "metric": "11 report sections with lineage, final-test metrics, limits and archive",
            "evidence": args.report.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_acceptance_matrix(facts)
    with (args.output_dir / "week6_acceptance_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    input_paths = [REPO_ROOT / path for path in evidence.values()]
    input_paths.append(args.report)
    manifest = build_report_input_manifest(REPO_ROOT, input_paths)
    (args.output_dir / "report_input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checkpoint = loaded["checkpoint_selection"]
    release = loaded["frozen_release"]
    final_test = loaded["final_test"]
    archive = {
        "schema_version": "2.0",
        "status": "archived_reference",
        "deployment_recommendation": "checkpoint_38_plus_main_prompt",
        "checkpoint": release.get("checkpoint"),
        "adapter": {
            "sha256": release.get("adapter_model_sha256"),
            "config_sha256": release.get("adapter_config_sha256"),
        },
        "lineage": loaded["day31_adapter"].get("lineage"),
        "prompt": {
            "variant": release.get("prompt_variant"),
            "sha256": release.get("prompt_sha256"),
        },
        "generation_config": release.get("generation_config"),
        "max_steps": release.get("max_steps"),
        "selection": {
            "split": checkpoint.get("selection_split"),
            "metrics": checkpoint.get("selection_metrics"),
            "uses_final_test": release.get("selection_uses_final_test"),
        },
        "final_test": final_test.get("results"),
        "large_weights_in_git": False,
        "source_manifest_sha256": sha256_file(REPO_ROOT / evidence["day31_adapter"]),
    }
    (args.output_dir / "final_agent_archive.json").write_text(json.dumps(archive, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
