#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and inventory a merged Hugging Face model."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    parser.add_argument("--sha256-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--minimum-weight-bytes",
        type=int,
        default=10_000_000_000,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_full_weight_file(path: Path) -> bool:
    name = path.name
    return (
        name == "model.safetensors"
        or name.startswith("model-") and name.endswith(".safetensors")
        or name == "pytorch_model.bin"
        or name.startswith("pytorch_model-") and name.endswith(".bin")
    )


def validate_model(
    model_dir: Path,
    minimum_weight_bytes: int,
) -> tuple[dict[str, Any], list[Path]]:
    if not model_dir.is_dir():
        raise ValueError(f"merged model directory not found: {model_dir}")
    config_path = model_dir / "config.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if not config_path.is_file():
        raise ValueError("config.json is missing")
    if not tokenizer_config_path.is_file():
        raise ValueError("tokenizer_config.json is missing")

    files = sorted(
        path for path in model_dir.rglob("*") if path.is_file()
    )
    weight_files = [path for path in files if is_full_weight_file(path)]
    if not weight_files:
        raise ValueError("no full model weight files were found")
    weight_bytes = sum(path.stat().st_size for path in weight_files)
    if weight_bytes < minimum_weight_bytes:
        raise ValueError(
            f"full model weights are below minimum: "
            f"{weight_bytes} < {minimum_weight_bytes}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_config = json.loads(
        tokenizer_config_path.read_text(encoding="utf-8")
    )
    report = {
        "valid": True,
        "model_dir": str(model_dir.resolve()),
        "model_type": config.get("model_type"),
        "tokenizer_has_chat_template": bool(
            tokenizer_config.get("chat_template")
        ),
        "file_count": len(files),
        "weight_file_count": len(weight_files),
        "weight_bytes": weight_bytes,
        "total_bytes": sum(path.stat().st_size for path in files),
        "adapter_weight_present": any(
            path.name.startswith("adapter_model") for path in files
        ),
    }
    return report, files


def write_outputs(
    model_dir: Path,
    files: list[Path],
    report: dict[str, Any],
    inventory_output: Path,
    sha256_output: Path,
    report_output: Path,
) -> None:
    inventory_output.parent.mkdir(parents=True, exist_ok=True)
    sha256_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    inventory_lines = [
        f"{path.relative_to(model_dir)}\t{path.stat().st_size}"
        for path in files
    ]
    hash_lines = [
        f"{sha256(path)}  {path.relative_to(model_dir)}" for path in files
    ]
    inventory_output.write_text(
        "\n".join(inventory_lines) + "\n",
        encoding="utf-8",
    )
    sha256_output.write_text(
        "\n".join(hash_lines) + "\n",
        encoding="utf-8",
    )
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        report, files = validate_model(
            args.model_dir,
            args.minimum_weight_bytes,
        )
        write_outputs(
            args.model_dir,
            files,
            report,
            args.inventory_output,
            args.sha256_output,
            args.report_output,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
