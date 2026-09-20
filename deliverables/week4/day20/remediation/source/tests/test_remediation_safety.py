from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "source/scripts/score_remediation_safety.py"


def load_scorer():
    spec = importlib.util.spec_from_file_location("score_remediation_safety", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harmful_and_benign_metrics_are_independent():
    scorer = load_scorer()
    rows = []
    for index in range(20):
        rows.append({"prompt_type": "harmful", "teacher_refusal": index < 19, "enhanced_safe_completion": index < 15, "actionable_harm": index == 19, "appropriate_help": ""})
    for index in range(10):
        rows.append({"prompt_type": "benign", "teacher_refusal": "", "enhanced_safe_completion": "", "actionable_harm": False, "appropriate_help": index < 9})
    summary = scorer.summarize(rows)
    assert summary["harmful_refusal_rate"] == 0.95
    assert summary["enhanced_safe_completion_rate"] == 0.75
    assert summary["actionable_harm_count"] == 1
    assert summary["benign_appropriate_help_rate"] == 0.9

