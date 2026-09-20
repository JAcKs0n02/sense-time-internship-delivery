from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SOURCE_ROOT / "scripts/validate_day32.py"


def test_day32_validator_accepts_complete_prompt_ablation(tmp_path: Path) -> None:
    output = tmp_path / "validation.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["checks"]["teacher_failure_categories_present"] is True
    assert receipt["checks"]["prompt_only_ablation"] is True
    assert receipt["checks"]["regressions_reported"] is True
    assert receipt["facts"]["baseline_failed_cases"] == 22
    assert receipt["facts"]["optimized_failed_cases"] == 26
    assert receipt["facts"]["recommended_prompt"] == "main"


def test_day32_validator_help_lists_formal_evidence_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    for flag in ("--baseline", "--optimized", "--comparison", "--report", "--output"):
        assert flag in completed.stdout
