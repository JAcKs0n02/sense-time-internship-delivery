#!/usr/bin/env python3
"""Validate Day27 reporting integrity without turning a quality failure into PASS."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from build_day27_assets import validate_model_manifest


REQUIRED_REPORT_HEADINGS = tuple(f"## {index}." for index in range(1, 12))
FORBIDDEN_SUBMISSION_PARTS = ("blind_key", "private", "shutdown", "adapter_model.safetensors")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(repo_root: Path) -> list[str]:
    errors: list[str] = []
    week5 = repo_root / "deliverables/week5"
    day27 = week5 / "day27"
    results = day27 / "source/results"

    manifest = json.loads((results / "report_input_manifest.json").read_text())
    for row in manifest["inputs"]:
        path = week5 / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            errors.append(f"report input hash mismatch: {row['path']}")

    report = (day27 / "REPORT.md").read_text(encoding="utf-8")
    for heading in REQUIRED_REPORT_HEADINGS:
        if heading not in report:
            errors.append(f"report heading is missing: {heading}")

    with (results / "week5_acceptance_matrix.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4 or {row["requirement_id"] for row in rows} != {
        "W5-INFERENCE", "W5-ATTENTION", "W5-QUALITY", "W5-REPORT"
    }:
        errors.append("acceptance matrix must contain the four teacher requirements")
    quality = next((row for row in rows if row["requirement_id"] == "W5-QUALITY"), {})
    if quality.get("status") != "FAIL":
        errors.append("quality acceptance must remain FAIL after two v8 development failures")

    model_manifest = json.loads((results / "final_vlm_model_manifest.json").read_text())
    errors.extend(validate_model_manifest(model_manifest))
    if model_manifest.get("status") != "NO_ACCEPTANCE_PASSING_MODEL":
        errors.append("model archive must disclose that no acceptance-passing model exists")

    selection = json.loads(
        (week5 / "day26/source/results/v8/candidate_selection.json").read_text()
    )
    if (
        selection.get("status") != "NO_PASSING_CANDIDATE"
        or len(selection.get("candidates", [])) != 2
        or selection.get("final_evaluation_run") is not False
    ):
        errors.append("v8 terminal selection does not match the preregistered stop rule")

    submission = repo_root / "Submission/Week5/Day27_Weekly_Report"
    required = {
        "README.md",
        "Week5_Multimodal_Practice_Report.md",
        "Week5_Acceptance_Matrix.csv",
        "Final_VLM_Model_Manifest.json",
        "Day26_V8_Candidate_Selection.json",
        "Day27_Validation.json",
    }
    if {path.name for path in submission.iterdir() if path.is_file()} != required:
        errors.append("Day27 teacher projection does not match the frozen minimal file set")
    for path in submission.rglob("*"):
        lowered = path.as_posix().lower()
        if any(part in lowered for part in FORBIDDEN_SUBMISSION_PARTS):
            errors.append(f"forbidden teacher artifact: {path.relative_to(submission)}")
        if path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"teacher artifact is unexpectedly large: {path.relative_to(submission)}")
    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[5]
    errors = validate(repo_root)
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
