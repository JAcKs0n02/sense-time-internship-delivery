#!/usr/bin/env python3
"""Validate the frozen Day19 DPO/QLoRA training contract."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import yaml


FROZEN_VALUES: dict[str, Any] = {
    "stage": "dpo",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "lora_rank": 8,
    "lora_alpha": 16,
    "pref_beta": 0.1,
    "pref_loss": "sigmoid",
    "template": "qwen",
    "cutoff_len": 2048,
    "quantization_bit": 4,
    "quantization_method": "bnb",
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
    "overwrite_output_dir": False,
}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=1e-9, abs_tol=1e-12
        )
    return actual == expected


def validate_config(
    config: dict[str, Any], *, require_explicit_ref: bool = False
) -> list[str]:
    errors: list[str] = []
    for key, expected in FROZEN_VALUES.items():
        actual = config.get(key)
        if not _same(actual, expected):
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")

    for key in ("model_name_or_path", "dataset", "dataset_dir", "output_dir"):
        if not config.get(key):
            errors.append(f"{key}: a non-empty value is required")

    ref_model = config.get("ref_model")
    if require_explicit_ref and not ref_model:
        errors.append("ref_model: explicit reference model is required")
    if require_explicit_ref and config.get("ref_model_quantization_bit") != 4:
        errors.append("ref_model_quantization_bit: explicit reference must use 4-bit quantization")
    if ref_model and ref_model != config.get("model_name_or_path"):
        errors.append("ref_model: must point to the frozen Week3 merged SFT model")

    if "beta" in config:
        errors.append("beta: unsupported key; use LLaMA-Factory pref_beta")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--require-explicit-ref", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_yaml(args.config)
        errors = validate_config(
            config, require_explicit_ref=args.require_explicit_ref
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"INVALID: {exc}")
        return 1
    if errors:
        print("INVALID Day19 configuration:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"VALID: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
