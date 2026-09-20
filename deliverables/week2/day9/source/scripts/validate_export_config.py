#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REQUIRED_KEYS = {
    "model_name_or_path",
    "adapter_name_or_path",
    "template",
    "trust_remote_code",
    "export_dir",
    "export_size",
    "export_device",
    "export_legacy_format",
}
ALLOWED_KEYS = REQUIRED_KEYS
FORBIDDEN_QUANTIZATION_KEYS = {
    "quantization_bit",
    "quantization_method",
    "export_quantization_bit",
    "export_quantization_dataset",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a LLaMA-Factory LoRA export configuration."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def resolved_path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_KEYS - config.keys())
    if missing:
        raise ValueError(f"missing required export keys: {', '.join(missing)}")

    unsupported = sorted(
        config.keys() - ALLOWED_KEYS - FORBIDDEN_QUANTIZATION_KEYS
    )
    if unsupported:
        raise ValueError(f"unsupported export keys: {', '.join(unsupported)}")

    present_quantization = sorted(
        key for key in FORBIDDEN_QUANTIZATION_KEYS if key in config
    )
    if present_quantization:
        raise ValueError(
            "quantization_bit must not be set while merging LoRA adapters: "
            + ", ".join(present_quantization)
        )

    model_dir = resolved_path(config["model_name_or_path"])
    adapter_dir = resolved_path(config["adapter_name_or_path"])
    export_dir = resolved_path(config["export_dir"])
    if not (model_dir / "config.json").is_file():
        raise ValueError(f"base model config.json not found: {model_dir}")
    if not (adapter_dir / "adapter_config.json").is_file():
        raise ValueError(
            f"adapter_config.json not found in adapter: {adapter_dir}"
        )
    if export_dir in {model_dir, adapter_dir}:
        raise ValueError("export_dir must differ from base and adapter paths")
    if export_dir.exists() and any(export_dir.iterdir()):
        raise ValueError(f"export_dir is already non-empty: {export_dir}")

    base_config = json.loads(
        (model_dir / "config.json").read_text(encoding="utf-8")
    )
    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    adapter_base = resolved_path(adapter_config["base_model_name_or_path"])
    if adapter_base != model_dir:
        raise ValueError(
            "adapter base model does not match model_name_or_path: "
            f"{adapter_base} != {model_dir}"
        )
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("adapter peft_type is not LORA")
    if config["template"] != "qwen":
        raise ValueError("template must be qwen for this Qwen2.5 experiment")
    if config["export_device"] not in {"cpu", "auto"}:
        raise ValueError("export_device must be cpu or auto")
    if not isinstance(config["export_size"], int) or config["export_size"] <= 0:
        raise ValueError("export_size must be a positive integer")
    if config["export_legacy_format"] is not False:
        raise ValueError("export_legacy_format must be false")

    return {
        "valid": True,
        "model_dir": str(model_dir),
        "adapter_dir": str(adapter_dir),
        "export_dir": str(export_dir),
        "model_type": base_config.get("model_type"),
        "template": config["template"],
        "export_device": config["export_device"],
        "export_size": config["export_size"],
        "adapter_peft_type": adapter_config.get("peft_type"),
        "adapter_rank": adapter_config.get("r"),
        "adapter_alpha": adapter_config.get("lora_alpha"),
        "quantization_keys_present": [],
    }


def main() -> int:
    args = parse_args()
    try:
        payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("export YAML root must be a mapping")
        report = validate_config(payload)
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
