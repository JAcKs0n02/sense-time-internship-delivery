from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_merged_model.py"


def load_module():
    spec = importlib.util.spec_from_file_location("validate_merged_model_day20", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(tmp_path: Path, *, weight_bytes: int = 64) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x"}))
    (model / "tokenizer.json").write_text("{}")
    (model / "generation_config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"0" * weight_bytes)
    return model


def test_accepts_full_model_contract(tmp_path: Path):
    module = load_module()
    report, files = module.validate_model(make_model(tmp_path), minimum_weight_bytes=32)
    assert report["valid"] is True
    assert report["adapter_weight_present"] is False
    assert len(files) == 5


def test_rejects_adapter_only_output(tmp_path: Path):
    module = load_module()
    model = make_model(tmp_path)
    (model / "model.safetensors").unlink()
    (model / "adapter_model.safetensors").write_bytes(b"0" * 64)
    with pytest.raises(ValueError, match="full model"):
        module.validate_model(model, minimum_weight_bytes=32)


def test_rejects_missing_tokenizer_and_small_weights(tmp_path: Path):
    module = load_module()
    model = make_model(tmp_path, weight_bytes=8)
    (model / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer.json"):
        module.validate_model(model, minimum_weight_bytes=32)
    (model / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="below minimum"):
        module.validate_model(model, minimum_weight_bytes=32)
