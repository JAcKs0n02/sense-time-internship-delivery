"""Static validation for Week 7 AWQ and GPTQ export configurations."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .manifests import WEEK4_DPO_IDENTITY


CALIBRATION_PATH = (
    "deliverables/week7/day34/source/data/calibration_128.jsonl"
)
CALIBRATION_SHA256 = (
    "e26e7b96ba8af04b270d2c03c777c4482ddfac7ff081c4da14fce7d4e1a85419"
)
WEEK7_REMOTE_ROOT = "/root/autodl-tmp/qwen25-week7/"


def parse_export_config_text(text: str) -> dict[str, object]:
    """Load JSON-syntax YAML without dependencies, with PyYAML as fallback."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "non-JSON YAML requires PyYAML in the execution environment"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("quantizer config must be a mapping")
    return value


def _require_exact(
    errors: list[str],
    mapping: Mapping[str, object],
    field: str,
    expected: object,
    prefix: str = "",
) -> None:
    if mapping.get(field) != expected:
        errors.append(
            f"{prefix}{field}: expected {expected!r}, received {mapping.get(field)!r}"
        )


def _validate_output_path(errors: list[str], value: object, field: str) -> None:
    if not isinstance(value, str) or not value.startswith(WEEK7_REMOTE_ROOT):
        errors.append(f"{field}: must be inside {WEEK7_REMOTE_ROOT}")


def validate_awq_config(config: Mapping[str, object]) -> list[str]:
    """Validate AutoAWQ settings against the approved Week 7 protocol."""
    errors: list[str] = []
    _require_exact(
        errors,
        config,
        "model_name_or_path",
        WEEK4_DPO_IDENTITY.model_dir,
    )
    _validate_output_path(errors, config.get("output_dir"), "output_dir")

    calibration = config.get("calibration")
    if not isinstance(calibration, Mapping):
        errors.append("calibration: object required")
    else:
        _require_exact(errors, calibration, "path", CALIBRATION_PATH, "calibration.")
        _require_exact(
            errors,
            calibration,
            "sha256",
            CALIBRATION_SHA256,
            "calibration.",
        )
        _require_exact(errors, calibration, "row_count", 128, "calibration.")
        _require_exact(errors, calibration, "max_tokens", 1024, "calibration.")

    quantization = config.get("quantization")
    if not isinstance(quantization, Mapping):
        errors.append("quantization: object required")
    else:
        for field, expected in (
            ("w_bit", 4),
            ("q_group_size", 128),
            ("zero_point", True),
            ("version", "GEMM"),
        ):
            _require_exact(
                errors, quantization, field, expected, "quantization."
            )
    return errors


def validate_gptq_export_config(config: Mapping[str, object]) -> list[str]:
    """Validate LLaMA-Factory 0.9.3's GPTQ-only export contract."""
    errors: list[str] = []
    required = (
        ("model_name_or_path", WEEK4_DPO_IDENTITY.model_dir),
        ("template", "qwen"),
        ("export_quantization_bit", 4),
        ("export_quantization_dataset", CALIBRATION_PATH),
        ("export_quantization_maxlen", 1024),
        ("export_quantization_nsamples", 128),
        ("export_device", "cuda"),
    )
    for field, expected in required:
        _require_exact(errors, config, field, expected)
    _validate_output_path(errors, config.get("export_dir"), "export_dir")
    if "quantization_method" in config:
        errors.append(
            "quantization_method: loading-time field is forbidden in GPTQ export"
        )
    if "quantization_bit" in config:
        errors.append(
            "quantization_bit: on-the-fly loading field is forbidden in export"
        )
    return errors
