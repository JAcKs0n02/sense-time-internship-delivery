from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


DAY17_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = DAY17_ROOT / "source"
SCRIPT_PATH = SOURCE_ROOT / "scripts" / "validate_day17.py"
TAXONOMY_PATH = SOURCE_ROOT / "data" / "preference_taxonomy.json"
EXAMPLES_PATH = SOURCE_ROOT / "examples" / "preference_pair_examples.json"
GUIDE_PATH = SOURCE_ROOT / "docs" / "preference_data_construction_guide.md"

EXPECTED_TYPES = {
    "factuality",
    "safety",
    "completeness",
    "helpfulness",
    "format",
}


def load_validator():
    assert SCRIPT_PATH.is_file(), f"validator is missing: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("validate_day17", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_repository_payloads() -> tuple[dict, dict]:
    assert TAXONOMY_PATH.is_file(), f"taxonomy is missing: {TAXONOMY_PATH}"
    assert EXAMPLES_PATH.is_file(), f"examples are missing: {EXAMPLES_PATH}"
    return (
        json.loads(TAXONOMY_PATH.read_text(encoding="utf-8")),
        json.loads(EXAMPLES_PATH.read_text(encoding="utf-8")),
    )


def test_repository_artifacts_pass_the_day17_contract():
    validator = load_validator()
    taxonomy, examples = load_repository_payloads()

    result = validator.validate_payloads(taxonomy, examples)

    assert result == {
        "valid": True,
        "errors": [],
        "category_counts": {
            "completeness": 2,
            "factuality": 2,
            "format": 2,
            "helpfulness": 2,
            "safety": 2,
        },
        "example_count": 10,
    }
    assert GUIDE_PATH.is_file()


def test_validator_rejects_a_missing_preference_type():
    validator = load_validator()
    taxonomy, examples = load_repository_payloads()
    broken = deepcopy(taxonomy)
    broken["preference_types"] = [
        item for item in broken["preference_types"] if item["id"] != "safety"
    ]

    result = validator.validate_payloads(broken, examples)

    assert result["valid"] is False
    assert "taxonomy.category_set_mismatch" in result["errors"]


def test_validator_rejects_duplicate_example_ids():
    validator = load_validator()
    taxonomy, examples = load_repository_payloads()
    broken = deepcopy(examples)
    broken["examples"][1]["id"] = broken["examples"][0]["id"]

    result = validator.validate_payloads(taxonomy, broken)

    assert result["valid"] is False
    assert "examples.duplicate_id:w4d17-factuality-001" in result["errors"]


def test_validator_rejects_an_identical_preference_pair():
    validator = load_validator()
    taxonomy, examples = load_repository_payloads()
    broken = deepcopy(examples)
    broken["examples"][0]["rejected"]["value"] = broken["examples"][0]["chosen"]["value"]

    result = validator.validate_payloads(taxonomy, broken)

    assert result["valid"] is False
    assert "examples.responses_not_distinct:w4d17-factuality-001" in result["errors"]


def test_validator_rejects_a_pair_that_only_differs_by_outer_whitespace():
    validator = load_validator()
    taxonomy, examples = load_repository_payloads()
    broken = deepcopy(examples)
    chosen = broken["examples"][0]["chosen"]["value"]
    broken["examples"][0]["rejected"]["value"] = f"  {chosen}\n"

    result = validator.validate_payloads(taxonomy, broken)

    assert result["valid"] is False
    assert "examples.responses_not_distinct:w4d17-factuality-001" in result["errors"]


def test_validator_requires_both_safety_behaviors():
    validator = load_validator()
    taxonomy, examples = load_repository_payloads()
    broken = deepcopy(examples)
    for example in broken["examples"]:
        if example["preference_type"] == "safety":
            example["safety_behavior"] = "refuse_harmful"

    result = validator.validate_payloads(taxonomy, broken)

    assert result["valid"] is False
    assert "examples.safety_behavior_coverage_mismatch" in result["errors"]


def test_validator_rejects_unreviewed_or_multidimensional_examples():
    validator = load_validator()
    taxonomy, examples = load_repository_payloads()
    broken = deepcopy(examples)
    broken["examples"][0]["quality_review"]["status"] = "pending"
    broken["examples"][1]["quality_review"]["single_dimension"] = False

    result = validator.validate_payloads(taxonomy, broken)

    assert result["valid"] is False
    assert "examples.review_not_approved:w4d17-factuality-001" in result["errors"]
    assert "examples.not_single_dimension:w4d17-factuality-002" in result["errors"]


def test_cli_writes_the_same_machine_readable_result(tmp_path: Path):
    output_path = tmp_path / "validation.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--taxonomy",
            str(TAXONOMY_PATH),
            "--examples",
            str(EXAMPLES_PATH),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["valid"] is True
    assert result["errors"] == []
    assert result["example_count"] == 10
    assert set(result["category_counts"]) == EXPECTED_TYPES
