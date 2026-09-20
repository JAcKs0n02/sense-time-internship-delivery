from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "source" / "scripts" / "build_safety_v2_dataset.py"
BASE = ROOT.parents[1] / "day18" / "source" / "data" / "final" / "week4_preferences_full.json"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_safety_v2_dataset", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supplement_contract_and_diversity():
    builder = load_builder()
    rows = builder.build_supplement()

    assert len(rows) == 160
    assert len({row["id"] for row in rows}) == 160
    assert {row["safety_behavior"] for row in rows} == {"harmful_refusal", "benign_help"}
    assert sum(row["safety_behavior"] == "harmful_refusal" for row in rows) == 80
    assert sum(row["safety_behavior"] == "benign_help" for row in rows) == 80
    assert len({row["prompt_structure"] for row in rows if row["safety_behavior"] == "harmful_refusal"}) >= 6
    assert len({row["prompt_structure"] for row in rows if row["safety_behavior"] == "benign_help"}) >= 6

    for row in rows:
        assert row["conversations"][-1]["from"] == "human"
        assert row["chosen"]["from"] == row["rejected"]["from"] == "gpt"
        assert row["chosen"]["value"] != row["rejected"]["value"]
        assert row["topic_specific_alternative"] in row["chosen"]["value"]
        assert row["quality_review"]["safe_to_publish"] is True
        assert row["rejected_is_non_operational"] is True


def test_v2_is_870_and_split_is_deterministic_and_stratified():
    builder = load_builder()
    base = json.loads(BASE.read_text(encoding="utf-8"))
    supplement = builder.build_supplement()
    first = builder.build_v2(base, supplement, seed=42)
    second = builder.build_v2(list(reversed(base)), list(reversed(supplement)), seed=42)

    assert len(first[0]) == 870
    assert len(first[1]) + len(first[2]) == 870
    assert len(first[2]) == 87
    assert [row["id"] for row in first[0]] == [row["id"] for row in second[0]]
    assert [row["id"] for row in first[1]] == [row["id"] for row in second[1]]
    assert [row["id"] for row in first[2]] == [row["id"] for row in second[2]]
    assert {row["id"] for row in first[1]}.isdisjoint({row["id"] for row in first[2]})
    for behavior in ("harmful_refusal", "benign_help"):
        assert sum(row.get("safety_behavior") == behavior for row in first[2]) == 8


def test_cli_generation_is_byte_deterministic(tmp_path: Path):
    builder = load_builder()
    first = tmp_path / "first"
    second = tmp_path / "second"
    builder.generate(BASE, first)
    builder.generate(BASE, second)
    names = {
        "safety_supplemental_160.json",
        "week4_preferences_v2.json",
        "week4_dpo_train_v2.json",
        "week4_dpo_validation_v2.json",
        "remediation_dev_prompts.json",
        "dataset_info.json",
    }
    for name in names:
        left = (first / name).read_bytes()
        right = (second / name).read_bytes()
        assert hashlib.sha256(left).digest() == hashlib.sha256(right).digest()

