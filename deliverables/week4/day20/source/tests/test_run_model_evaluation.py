from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_model_evaluation.py"
DAY20 = Path(__file__).parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location("run_model_evaluation_day20", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_records(alias: str = "sft_only") -> list[dict]:
    prompts = []
    for name in ("safety_prompts.json", "business_prompts.json"):
        prompts.extend(json.loads((DAY20 / "source" / "data" / name).read_text())["prompts"])
    generation = json.loads((DAY20 / "configs" / "evaluation_config.json").read_text())["generation_config"]
    return [
        {
            "schema_version": "1.0",
            "prompt_id": item["id"],
            "prompt_type": item["prompt_type"],
            "prompt_text": item["text"],
            "model_alias": alias,
            "model_dir": "/models/test",
            "messages": [{"role": "user", "content": item["text"]}],
            "rendered_prompt_sha256": "a" * 64,
            "generation_config": generation,
            "response": "test response",
            "input_tokens": 10,
            "output_tokens": 2,
            "elapsed_seconds": 0.1,
            "status": "completed",
        }
        for item in prompts
    ]


def test_accepts_exact_complete_model_run():
    module = load_module()
    config = json.loads((DAY20 / "configs" / "evaluation_config.json").read_text())
    report = module.validate_response_records(make_records(), config, "sft_only")
    assert report["valid"] is True
    assert report["response_count"] == 15


def test_rejects_missing_duplicate_or_rewritten_prompt():
    module = load_module()
    config = json.loads((DAY20 / "configs" / "evaluation_config.json").read_text())
    records = make_records()
    records[-1] = records[0]
    with pytest.raises(ValueError, match="duplicate|exactly"):
        module.validate_response_records(records, config, "sft_only")
    records = make_records()
    records[0]["prompt_text"] = "rewritten"
    with pytest.raises(ValueError, match="prompt text"):
        module.validate_response_records(records, config, "sft_only")


def test_rejects_generation_or_status_drift():
    module = load_module()
    config = json.loads((DAY20 / "configs" / "evaluation_config.json").read_text())
    records = make_records()
    records[0]["generation_config"] = {**records[0]["generation_config"], "do_sample": True}
    with pytest.raises(ValueError, match="generation"):
        module.validate_response_records(records, config, "sft_only")
    records = make_records()
    records[0]["status"] = "failed"
    with pytest.raises(ValueError, match="completed"):
        module.validate_response_records(records, config, "sft_only")
