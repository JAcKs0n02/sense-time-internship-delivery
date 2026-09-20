#!/usr/bin/env python3
"""Validate the Week 6 report, acceptance matrix and frozen input manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = SOURCE_ROOT.parents[3]
sys.path.insert(0, str(DEFAULT_REPO / "deliverables/week6/source"))

from week6_agent.day33_reporting import (  # noqa: E402
    ACCEPTANCE_REQUIREMENTS,
    validate_report_inputs,
)


REQUIRED_REPORT_SECTIONS = (
    "## 1. 摘要与老师要求",
    "## 2. 环境与模型血缘",
    "## 3. 三个工具",
    "## 4. ReAct Agent 与单轮调用",
    "## 5. 多步任务",
    "## 6. 工具调用 SFT",
    "## 7. 固定评测与错误模式",
    "## 8. 安全边界与限制",
    "## 9. 老师验收矩阵",
    "## 10. 模型归档与复现",
    "## 11. 结论",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--acceptance-matrix", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument(
        "--model-archive",
        type=Path,
        default=SOURCE_ROOT / "results/final_agent_archive.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    try:
        report = args.report.read_text(encoding="utf-8")
        checks["report_title"] = report.startswith("# 第 6 周：Agent 智能体开发报告")
        missing_sections = [section for section in REQUIRED_REPORT_SECTIONS if section not in report]
        checks["report_sections"] = not missing_sections
        checks["report_no_pending_placeholder"] = "**未开始。**" not in report and "TODO" not in report
        details["missing_report_sections"] = missing_sections
    except OSError as error:
        checks["report_readable"] = False
        details["report_error"] = str(error)

    try:
        with args.acceptance_matrix.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_ids = [item[0] for item in ACCEPTANCE_REQUIREMENTS]
        checks["acceptance_rows"] = [row.get("requirement_id") for row in rows] == expected_ids
        checks["acceptance_statuses"] = all(row.get("status") in {"PASS", "FAIL"} for row in rows)
        checks["acceptance_all_pass"] = all(row.get("status") == "PASS" for row in rows)
        checks["acceptance_evidence"] = all(row.get("metric") and row.get("evidence") for row in rows)
        evidence_paths = [
            item.strip()
            for row in rows
            for item in str(row.get("evidence", "")).split(";")
            if item.strip()
        ]
        checks["acceptance_evidence_paths"] = bool(evidence_paths) and all(
            not Path(item).is_absolute()
            and ".." not in Path(item).parts
            and (args.repo_root / item).is_file()
            for item in evidence_paths
        )
        details["acceptance_status"] = {row.get("requirement_id"): row.get("status") for row in rows}
    except OSError as error:
        checks["acceptance_matrix_readable"] = False
        details["acceptance_error"] = str(error)

    try:
        manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
        manifest_failures = validate_report_inputs(args.repo_root, manifest)
        checks["input_manifest"] = not manifest_failures
        details["manifest_failures"] = manifest_failures
        details["input_count"] = len(manifest.get("files", []))
    except (OSError, json.JSONDecodeError) as error:
        checks["input_manifest_readable"] = False
        details["manifest_error"] = str(error)

    try:
        archive = json.loads(args.model_archive.read_text(encoding="utf-8"))
        checks["model_archive"] = (
            archive.get("status") == "archived_reference"
            and archive.get("deployment_recommendation")
            == "checkpoint_38_plus_main_prompt"
            and archive.get("adapter", {}).get("sha256")
            == "a5e348ad180f8183479d3138b40613ead7163f3b2089c9b271063b64be5cdc5c"
            and archive.get("prompt", {}).get("variant") == "main"
            and archive.get("selection", {}).get("uses_final_test") is False
            and archive.get("final_test", {}).get("raw", {}).get("case_count") == 100
            and archive.get("large_weights_in_git") is False
        )
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        checks["model_archive_readable"] = False
        details["model_archive_error"] = str(error)

    failed = [name for name, passed in checks.items() if not passed]
    receipt = {
        "schema_version": "1.0",
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
