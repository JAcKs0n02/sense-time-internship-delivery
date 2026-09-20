from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SOURCE_ROOT / "scripts/validate_day33.py"
sys.path.insert(0, str(SOURCE_ROOT.parents[3] / "deliverables/week6/source"))


def test_validator_cli_exposes_report_matrix_and_manifest_inputs() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--report" in result.stdout
    assert "--acceptance-matrix" in result.stdout
    assert "--input-manifest" in result.stdout
    assert "--output" in result.stdout


def test_validator_fails_closed_when_required_inputs_are_missing(tmp_path: Path) -> None:
    output = tmp_path / "validation.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(tmp_path),
            "--report",
            str(tmp_path / "REPORT.md"),
            "--acceptance-matrix",
            str(tmp_path / "matrix.csv"),
            "--input-manifest",
            str(tmp_path / "manifest.json"),
            "--output",
            str(output),
        ],
        check=False,
    )
    assert result.returncode != 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert receipt["failed_checks"]


def test_v2_local_readiness_binds_required_artifacts_and_metric_namespaces() -> None:
    from week6_agent.day33_reporting import build_local_readiness

    repo_root = SOURCE_ROOT.parents[3]
    receipt = build_local_readiness(
        repo_root, pytest_count=121, pytest_exit_code=0
    )

    assert receipt["status"] == "ready_for_gpu"
    assert receipt["pytest"]["passed"] == 121
    assert receipt["metric_namespaces"] == {
        "raw": "raw_model_metrics",
        "guarded": "guarded_system_metrics",
    }
    assert receipt["gpu_evidence"] == {
        "checkpoint_selection": "pending",
        "training": "pending",
        "final_test": "pending",
    }
    assert {
        "tool_sft_train_v2.json",
        "tool_sft_dev_v2.json",
        "tool_sft_test_v2.json",
        "dataset_manifest_v2.json",
        "knowledge_base_v2.json",
    } <= {Path(row["path"]).name for row in receipt["artifacts"]}


def test_v2_local_readiness_fails_when_artifacts_or_tests_are_missing(
    tmp_path: Path,
) -> None:
    from week6_agent.day33_reporting import build_local_readiness

    with pytest.raises(ValueError, match="required v2 artifact"):
        build_local_readiness(tmp_path, pytest_count=0, pytest_exit_code=1)
