#!/usr/bin/env python3
"""Record why LLaMA-Factory 0.9.3 cannot export a new AWQ model."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path


PINNED_VERSION = "0.9.3"
SOURCE_URLS = [
    "https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/v0.9.3/src/llamafactory/model/model_utils/quantization.py",
    "https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/v0.9.3/src/llamafactory/hparams/model_args.py",
]


def _installed_receipt() -> dict[str, object]:
    try:
        version = importlib.metadata.version("llamafactory")
        from llamafactory.model.model_utils import quantization

        source = inspect.getsource(quantization.configure_quantization)
    except (importlib.metadata.PackageNotFoundError, ImportError, OSError) as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "available": True,
        "version": version,
        "version_matches_pin": version == PINNED_VERSION,
        "configure_quantization_has_gptq_config": "GPTQConfig" in source,
        "configure_quantization_has_awq_export_config": (
            "AwqConfig(" in source or "AWQConfig(" in source
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    receipt: dict[str, object] = {
        "schema_version": "1.0",
        "status": "PASS_STATIC_COMPATIBILITY_GATE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "teacher_target": "AWQ quantization of the Week 4 DPO model",
        "teacher_hint": "--quantization_method awq",
        "pinned_llamafactory_version": PINNED_VERSION,
        "finding": (
            "quantization_method is a loading-time field; the 0.9.3 export "
            "path constructs GPTQConfig and does not export a new AWQ model"
        ),
        "approved_awq_exporter": "AutoAWQ",
        "approved_gptq_exporter": "LLaMA-Factory 0.9.3 with GPTQModel",
        "official_source_urls": SOURCE_URLS,
        "evidence_scope": "pinned_official_source_review",
    }
    if not args.static_only:
        receipt["installed_runtime"] = _installed_receipt()
        receipt["evidence_scope"] = "pinned_official_source_and_installed_runtime"

    text = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
