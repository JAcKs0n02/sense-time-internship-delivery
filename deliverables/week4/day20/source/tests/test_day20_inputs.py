from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_day20_inputs.py"
DAY20 = Path(__file__).parents[2]
SOURCE = DAY20.parents[0] / "day18" / "source" / "data" / "raw" / "evaluation_holdout_prompts.json"


def load_module():
    spec = importlib.util.spec_from_file_location("validate_day20_inputs", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_frozen_inputs_match_source_and_contract():
    module = load_module()
    report = module.validate_inputs(
        SOURCE,
        DAY20 / "configs" / "evaluation_config.json",
        DAY20 / "source" / "data" / "safety_prompts.json",
        DAY20 / "source" / "data" / "business_prompts.json",
    )
    assert report["valid"] is True
    assert report["counts"] == {"safety": 10, "business": 5, "total": 15}


def test_rejects_source_hash_mismatch(tmp_path: Path):
    module = load_module()
    source = tmp_path / "source.json"
    source.write_text(SOURCE.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        module.validate_inputs(
            source,
            DAY20 / "configs" / "evaluation_config.json",
            DAY20 / "source" / "data" / "safety_prompts.json",
            DAY20 / "source" / "data" / "business_prompts.json",
        )


def test_rejects_duplicate_or_wrong_split(tmp_path: Path):
    module = load_module()
    safety = json.loads((DAY20 / "source" / "data" / "safety_prompts.json").read_text())
    safety["prompts"][-1] = safety["prompts"][0]
    altered = tmp_path / "safety.json"
    altered.write_text(json.dumps(safety, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unique|exactly"):
        module.validate_inputs(
            SOURCE,
            DAY20 / "configs" / "evaluation_config.json",
            altered,
            DAY20 / "source" / "data" / "business_prompts.json",
        )


def test_rejects_mutable_generation(tmp_path: Path):
    module = load_module()
    config = json.loads((DAY20 / "configs" / "evaluation_config.json").read_text())
    config["generation_config"]["do_sample"] = True
    altered = tmp_path / "config.json"
    altered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="do_sample"):
        module.validate_inputs(
            SOURCE,
            altered,
            DAY20 / "source" / "data" / "safety_prompts.json",
            DAY20 / "source" / "data" / "business_prompts.json",
        )


def test_remote_preflight_inventory_matches_week3_frozen_file_list():
    inventory = json.loads(
        (DAY20 / "source" / "data" / "sft_model_inventory.json").read_text()
    )
    frozen_lines = (
        DAY20.parents[1]
        / "week3"
        / "day15"
        / "source"
        / "results"
        / "best_merged_model_inventory.csv"
    ).read_text().splitlines()
    expected = {
        name: int(size)
        for name, size in (line.rsplit(",", 1) for line in frozen_lines)
    }
    assert inventory["files"] == expected
    assert inventory["file_count"] == 14
    assert inventory["regular_file_bytes"] == sum(expected.values()) == 15_247_180_180
