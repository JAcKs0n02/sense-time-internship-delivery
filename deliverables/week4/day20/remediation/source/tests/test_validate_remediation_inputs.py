from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "source" / "scripts" / "build_safety_v2_dataset.py"
VALIDATE_SCRIPT = ROOT / "source" / "scripts" / "validate_remediation_inputs.py"
TEACHER = ROOT.parent / "source" / "data" / "safety_prompts.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_generated_inputs_pass_all_gates():
    builder = load(BUILD_SCRIPT, "builder")
    validator = load(VALIDATE_SCRIPT, "validator")
    base = json.loads((ROOT.parents[1] / "day18/source/data/final/week4_preferences_full.json").read_text())
    full, train, validation = builder.build_v2(base, builder.build_supplement(), seed=42)
    dev = builder.build_dev_prompts()
    teacher = json.loads(TEACHER.read_text())

    report = validator.validate_inputs(full, train, validation, dev, teacher)

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["counts"] == {
        "full": 870,
        "train": 783,
        "validation": 87,
        "dev": 35,
        "dev_harmful": 20,
        "dev_benign": 10,
        "dev_business": 5,
    }
    assert report["exact_contamination_count"] == 0
    assert report["near_contamination_count"] == 0


def test_exact_and_near_contamination_are_blocking():
    builder = load(BUILD_SCRIPT, "builder_bad")
    validator = load(VALIDATE_SCRIPT, "validator_bad")
    base = json.loads((ROOT.parents[1] / "day18/source/data/final/week4_preferences_full.json").read_text())
    full, train, validation = builder.build_v2(base, builder.build_supplement(), seed=42)
    dev = builder.build_dev_prompts()
    teacher = json.loads(TEACHER.read_text())
    broken = deepcopy(full)
    broken[0]["conversations"][-1]["value"] = dev[0]["text"]

    report = validator.validate_inputs(broken, train, validation, dev, teacher)

    assert report["valid"] is False
    assert "exact_contamination" in report["errors"]

