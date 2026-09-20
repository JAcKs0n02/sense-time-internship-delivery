#!/usr/bin/env python3
"""Statically validate quantizer configs or smoke a produced model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from week7_deployment.hashing import portable_repo_path, sha256_file
from week7_deployment.quantization import (
    CALIBRATION_SHA256,
    parse_export_config_text,
    validate_awq_config,
    validate_gptq_export_config,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_AWQ = REPO_ROOT / "deliverables/week7/day34/configs/awq_config.json"
DEFAULT_GPTQ = REPO_ROOT / "deliverables/week7/day35/configs/week4_dpo_gptq.yaml"


def _read_config(method: str, path: Path) -> dict[str, object]:
    if method == "awq":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = parse_export_config_text(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("quantizer config must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("awq", "gptq"), default="gptq")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", default="请用一句话介绍量化部署。")
    args = parser.parse_args()

    config_path = args.config or (DEFAULT_AWQ if args.method == "awq" else DEFAULT_GPTQ)
    config = _read_config(args.method, config_path)
    errors = (
        validate_awq_config(config)
        if args.method == "awq"
        else validate_gptq_export_config(config)
    )
    calibration_path = REPO_ROOT / "deliverables/week7/day34/source/data/calibration_128.jsonl"
    if not calibration_path.is_file() or sha256_file(calibration_path) != CALIBRATION_SHA256:
        errors.append("frozen calibration data is missing or changed")

    receipt: dict[str, object] = {
        "schema_version": "1.0",
        "method": args.method,
        "config": portable_repo_path(config_path, REPO_ROOT),
        "scope": "config_only" if args.validate_config_only else "model_smoke",
        "errors": errors,
    }
    if not args.validate_config_only and not errors:
        if args.model_dir is None:
            errors.append("--model-dir is required for model smoke")
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.model_dir, local_files_only=True, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir,
                local_files_only=True,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
            )
            encoded = tokenizer(args.prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=32, do_sample=False)
            receipt["generated_text"] = tokenizer.decode(
                generated[0][encoded["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            receipt["model_dir"] = args.model_dir.as_posix()
            receipt["load_smoke_passed"] = True
            receipt["generation_smoke_passed"] = True
    receipt["errors"] = errors
    receipt["status"] = "PASS" if not errors else "FAIL"
    text = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
