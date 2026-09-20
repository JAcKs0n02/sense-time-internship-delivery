#!/usr/bin/env python3
"""Recompute Day32 failure labels from frozen cases and scored Agent traces."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "deliverables/week6/source"))

from week6_agent.day32_analysis import analyze_evaluation, compare_evaluations  # noqa: E402


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "case_id",
        "passed",
        "primary_label",
        "secondary_labels",
        "expected_tool_sequence",
        "actual_tool_sequence",
        "failed_checks",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: json.dumps(row[name], ensure_ascii=False)
                    if isinstance(row[name], list)
                    else row[name]
                    for name in fields
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        frozen = _load_json(args.frozen)
        baseline = _load_json(args.baseline)
        if not isinstance(frozen, list) or not isinstance(baseline, dict):
            raise ValueError("invalid frozen evaluation or baseline report")
        analysis = analyze_evaluation(frozen, baseline)
        optimized_analysis = None
        comparison = None
        if args.optimized is not None:
            optimized = _load_json(args.optimized)
            if not isinstance(optimized, dict):
                raise ValueError("invalid optimized evaluation report")
            optimized_analysis = analyze_evaluation(frozen, optimized)
            comparison = compare_evaluations(analysis, optimized_analysis)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "baseline_error_analysis.json", analysis)
    _write_csv(args.output_dir / "baseline_error_cases.csv", analysis["case_analysis"])
    if optimized_analysis is not None and comparison is not None:
        _write_json(
            args.output_dir / "optimized_error_analysis.json", optimized_analysis
        )
        _write_csv(
            args.output_dir / "optimized_error_cases.csv",
            optimized_analysis["case_analysis"],
        )
        _write_json(args.output_dir / "prompt_comparison.json", comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
