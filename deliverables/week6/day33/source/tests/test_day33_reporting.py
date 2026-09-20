from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SOURCE_ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT / "deliverables/week6/source"))

from week6_agent.day33_reporting import (  # noqa: E402
    build_acceptance_matrix,
    build_report_input_manifest,
    validate_report_inputs,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_report_manifest_uses_portable_paths_and_detects_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    first = _write(repo / "evidence/a.json", '{"status":"PASS"}\n')
    second = _write(repo / "evidence/b.csv", "metric,value\nscore,0.9\n")

    manifest = build_report_input_manifest(repo, [first, second])
    assert manifest["schema_version"] == "1.0"
    assert [row["path"] for row in manifest["files"]] == [
        "evidence/a.json",
        "evidence/b.csv",
    ]
    assert manifest["files"][0]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert validate_report_inputs(repo, manifest) == []

    second.write_text("metric,value\nscore,0.8\n", encoding="utf-8")
    assert validate_report_inputs(repo, manifest) == ["hash mismatch: evidence/b.csv"]


def test_acceptance_matrix_has_four_teacher_requirements() -> None:
    facts = {
        "three_tools": {"status": "PASS", "metric": "3/3", "evidence": "a.json"},
        "multistep": {"status": "PASS", "metric": "3 steps", "evidence": "b.json"},
        "error_analysis": {"status": "PASS", "metric": "30 cases", "evidence": "c.md"},
        "weekly_report": {"status": "PASS", "metric": "11 sections", "evidence": "REPORT.md"},
    }
    rows = build_acceptance_matrix(facts)
    assert [row["requirement_id"] for row in rows] == [
        "three_tools",
        "multistep",
        "error_analysis",
        "weekly_report",
    ]
    assert all(row["status"] == "PASS" for row in rows)
    assert all(row["metric"] and row["evidence"] for row in rows)


def test_acceptance_matrix_rejects_missing_or_invalid_status() -> None:
    try:
        build_acceptance_matrix({})
    except ValueError as error:
        assert "missing acceptance fact" in str(error)
    else:
        raise AssertionError("missing acceptance facts must fail")

    facts = {
        key: {"status": "UNKNOWN", "metric": "x", "evidence": "x"}
        for key in ("three_tools", "multistep", "error_analysis", "weekly_report")
    }
    try:
        build_acceptance_matrix(facts)
    except ValueError as error:
        assert "PASS or FAIL" in str(error)
    else:
        raise AssertionError("invalid status must fail")
