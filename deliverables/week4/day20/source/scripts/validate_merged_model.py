#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_full_weight(path: Path) -> bool:
    name = path.name
    return (
        name == "model.safetensors"
        or (name.startswith("model-") and name.endswith(".safetensors"))
        or name == "pytorch_model.bin"
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
    )


def validate_model(model_dir: Path, minimum_weight_bytes: int = 10_000_000_000) -> tuple[dict[str, Any], list[Path]]:
    if not model_dir.is_dir():
        raise ValueError(f"merged model directory not found: {model_dir}")
    required = ("config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json")
    for name in required:
        if not (model_dir / name).is_file():
            raise ValueError(f"{name} is missing")
    files = sorted(path for path in model_dir.rglob("*") if path.is_file())
    full_weights = [path for path in files if is_full_weight(path)]
    if not full_weights:
        raise ValueError("no full model weight files were found")
    weight_bytes = sum(path.stat().st_size for path in full_weights)
    if weight_bytes < minimum_weight_bytes:
        raise ValueError(f"full model weights are below minimum: {weight_bytes} < {minimum_weight_bytes}")
    adapter_present = any(path.name.startswith("adapter_model") for path in files)
    if adapter_present:
        raise ValueError("adapter weights must not remain in a completed merged export")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    if not tokenizer_config.get("chat_template"):
        raise ValueError("tokenizer_config.json must contain a chat_template")
    report = {
        "valid": True,
        "model_dir": str(model_dir.resolve()),
        "model_type": config.get("model_type"),
        "file_count": len(files),
        "weight_file_count": len(full_weights),
        "weight_bytes": weight_bytes,
        "total_bytes": sum(path.stat().st_size for path in files),
        "adapter_weight_present": False,
    }
    return report, files


def build_manifest(model_dir: Path, report: dict[str, Any], files: list[Path]) -> dict[str, Any]:
    entries = [
        {"path": str(path.relative_to(model_dir)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in files
    ]
    digest = hashlib.sha256()
    for item in entries:
        digest.update(f"{item['sha256']}  {item['path']}\n".encode("utf-8"))
    return {**report, "files": entries, "manifest_sha256": digest.hexdigest()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and inventory the Day 20 merged DPO model.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-weight-bytes", type=int, default=10_000_000_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report, files = validate_model(args.model_dir, args.minimum_weight_bytes)
        manifest = build_manifest(args.model_dir, report, files)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
