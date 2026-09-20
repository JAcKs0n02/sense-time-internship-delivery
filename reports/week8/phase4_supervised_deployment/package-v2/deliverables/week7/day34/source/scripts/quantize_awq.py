#!/usr/bin/env python3
"""Export the verified Week 4 DPO model with AutoAWQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from week7_deployment.hashing import sha256_file
from week7_deployment.quantization import validate_awq_config


def _load_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("AWQ config must be a JSON object")
    return value


def _calibration_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            messages = row["messages"]
            texts.append("\n".join(str(message["content"]) for message in messages))
    return texts


def _file_manifest(model_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(model_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    config = _load_config(args.config)
    errors = validate_awq_config(config)
    calibration = config.get("calibration")
    if not isinstance(calibration, dict):
        errors.append("calibration object missing")
        calibration_path = Path("missing")
    else:
        calibration_path = args.repo_root / str(calibration.get("path", ""))
        if not calibration_path.is_file():
            errors.append(f"missing calibration file: {calibration_path}")
        elif sha256_file(calibration_path) != calibration.get("sha256"):
            errors.append("calibration sha256 mismatch")
    if errors:
        raise SystemExit("; ".join(errors))
    if args.validate_config_only:
        print(json.dumps({"status": "PASS", "scope": "config_only"}))
        return 0

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model_path = str(config["model_name_or_path"])
    output_dir = Path(str(config["output_dir"]))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=bool(config.get("trust_remote_code"))
    )
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, trust_remote_code=bool(config.get("trust_remote_code"))
    )
    quant_config = dict(config["quantization"])
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=_calibration_texts(calibration_path),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_quantized(
        output_dir,
        safetensors=bool(config.get("safe_serialization", True)),
    )
    tokenizer.save_pretrained(output_dir)
    files = _file_manifest(output_dir)
    receipt = {
        "schema_version": "1.0",
        "status": "EXPORTED_AWAITING_LOAD_SMOKE",
        "method": "awq",
        "exporter": "AutoAWQ",
        "model_name_or_path": model_path,
        "output_dir": output_dir.as_posix(),
        "calibration_sha256": sha256_file(calibration_path),
        "quantization": quant_config,
        "file_count": len(files),
        "weight_bytes": sum(
            int(record["bytes"])
            for record in files
            if str(record["path"]).endswith((".safetensors", ".bin"))
        ),
        "files": files,
    }
    text = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
