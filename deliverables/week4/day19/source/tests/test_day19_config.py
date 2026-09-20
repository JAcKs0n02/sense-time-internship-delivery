from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_day19_config.py"
FORMAL_CONFIG = Path(__file__).parents[2] / "configs" / "qwen25_7b_week4_dpo_qlora.yaml"


def load_module():
    spec = importlib.util.spec_from_file_location("validate_day19_config", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_config(tmp_path: Path) -> dict:
    return {
        "model_name_or_path": "/models/week3-best-merged",
        "ref_model": "/models/week3-best-merged",
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "pref_beta": 0.1,
        "pref_loss": "sigmoid",
        "dataset": "week4_dpo_train",
        "dataset_dir": str(tmp_path),
        "template": "qwen",
        "cutoff_len": 2048,
        "quantization_bit": 4,
        "quantization_method": "bnb",
        "ref_model_quantization_bit": 4,
        "bf16": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5.0e-6,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "save_steps": 80,
        "save_total_limit": 2,
        "report_to": "tensorboard",
        "seed": 42,
        "output_dir": str(tmp_path / "output"),
        "overwrite_output_dir": False,
    }


def test_accepts_frozen_day19_configuration(tmp_path: Path):
    module = load_module()
    errors = module.validate_config(valid_config(tmp_path), require_explicit_ref=True)
    assert errors == []


def test_rejects_wrong_stage_beta_and_reference(tmp_path: Path):
    module = load_module()
    config = valid_config(tmp_path)
    config.update({"stage": "sft", "pref_beta": 0.2, "ref_model": "/models/other"})

    errors = module.validate_config(config, require_explicit_ref=True)

    assert any("stage" in error for error in errors)
    assert any("pref_beta" in error for error in errors)
    assert any("ref_model" in error for error in errors)


def test_rejects_unsafe_or_incompatible_training_settings(tmp_path: Path):
    module = load_module()
    config = valid_config(tmp_path)
    config.update(
        {
            "cutoff_len": 512,
            "quantization_bit": 8,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "overwrite_output_dir": True,
            "logging_steps": 10,
        }
    )

    errors = module.validate_config(config, require_explicit_ref=True)

    for key in (
        "cutoff_len",
        "quantization_bit",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "overwrite_output_dir",
        "logging_steps",
    ):
        assert any(key in error for error in errors), key


def test_cli_returns_nonzero_for_invalid_yaml(tmp_path: Path):
    module = load_module()
    config = valid_config(tmp_path)
    config["pref_beta"] = 1.0
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main([str(config_path), "--require-explicit-ref"]) == 1


def test_repository_formal_config_matches_frozen_contract():
    module = load_module()
    config = module.load_yaml(FORMAL_CONFIG)
    assert module.validate_config(config, require_explicit_ref=True) == []
