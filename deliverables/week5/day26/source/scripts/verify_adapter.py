#!/usr/bin/env python3
"""Audit the Day26 LoRA archive and verify that frozen vision weights stay unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_MARKERS = ("visual", "vision", "merger", "projector", "multi_modal")
REQUIRED_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
FROZEN_REFERENCE_SUFFIXES = (
    "visual.patch_embed.proj.weight",
    "visual.merger.mlp.0.weight",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_adapter_keys(keys: Iterable[str]) -> dict[str, Any]:
    names = sorted(keys)
    forbidden = [
        name for name in names if any(marker in name.casefold() for marker in FORBIDDEN_MARKERS)
    ]
    non_lora = [name for name in names if "lora_" not in name.casefold()]
    return {
        "passed": not forbidden and not non_lora and bool(names),
        "tensor_count": len(names),
        "forbidden_keys": forbidden,
        "non_lora_keys": non_lora,
    }


def tensor_sha256(tensor: Any) -> str:
    import torch

    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def frozen_reference_hashes(model: Any) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for suffix in FROZEN_REFERENCE_SUFFIXES:
        matches = [(name, value) for name, value in model.named_parameters() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one tensor ending in {suffix}, got {len(matches)}")
        hashes[suffix] = tensor_sha256(matches[0][1])
    return hashes


def inspect_safetensors(path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return list(handle.keys())


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import Qwen2VLForConditionalGeneration

    adapter_model = args.adapter_dir / "adapter_model.safetensors"
    adapter_config = args.adapter_dir / "adapter_config.json"
    if not adapter_model.is_file() or not adapter_config.is_file():
        raise FileNotFoundError("adapter archive is missing adapter_model.safetensors or adapter_config.json")

    config = json.loads(adapter_config.read_text(encoding="utf-8"))
    targets = set(config.get("target_modules") or [])
    if targets != REQUIRED_TARGETS:
        raise ValueError(f"unexpected adapter targets: {sorted(targets)}")

    keys = inspect_safetensors(adapter_model)
    key_audit = audit_adapter_keys(keys)
    if not key_audit["passed"]:
        raise RuntimeError(f"adapter key audit failed: {key_audit}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    before = frozen_reference_hashes(model)
    wrapped = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    wrapped.eval()
    after = frozen_reference_hashes(wrapped)
    frozen_unchanged = before == after

    files = []
    for path in sorted(args.adapter_dir.iterdir()):
        if path.is_file():
            files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    result = {
        "schema_version": "1.0",
        "completed_at": utc_now(),
        "status": "PASS" if frozen_unchanged else "FAIL",
        "model_dir": str(args.model_dir),
        "adapter_dir": str(args.adapter_dir),
        "target_modules": sorted(targets),
        "key_audit": key_audit,
        "frozen_reference_sha256_before": before,
        "frozen_reference_sha256_after": after,
        "frozen_reference_unchanged": frozen_unchanged,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not frozen_unchanged:
        raise RuntimeError("vision/projector reference tensors changed after loading the adapter")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
