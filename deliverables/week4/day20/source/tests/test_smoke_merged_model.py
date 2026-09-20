from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "smoke_merged_model.py"


def load_module():
    spec = importlib.util.spec_from_file_location("smoke_merged_model_day20", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_payload_requires_harmless_deterministic_generation():
    module = load_module()
    payload = module.build_payload(
        model_path=Path("/models/dpo-merged"),
        response="可以先澄清影响范围，再按日志时间线排查。",
        input_tokens=12,
        output_tokens=18,
        model_class="Qwen2ForCausalLM",
        tokenizer_class="Qwen2TokenizerFast",
    )
    assert payload["status"] == "PASS"
    assert payload["prompt"] == module.HARMLESS_PROMPT
    assert payload["generation_config"] == {
        "do_sample": False,
        "max_new_tokens": 128,
        "repetition_penalty": 1.0,
        "seed": 42,
    }
    assert payload["local_files_only"] is True
