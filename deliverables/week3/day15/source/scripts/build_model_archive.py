#!/usr/bin/env python3
"""Build a small auditable manifest for the remote best-model archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_archive(
    *,
    selected: dict,
    load_validation: dict,
    hashes_path: Path,
    inventory_path: Path,
    total_bytes: int,
) -> dict:
    if load_validation.get("merged_model_load") != "PASS":
        raise ValueError("merged model load validation did not pass")
    hash_lines = [line for line in hashes_path.read_text(encoding="utf-8").splitlines() if line]
    inventory_lines = [
        line for line in inventory_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not hash_lines or len(hash_lines) != len(inventory_lines):
        raise ValueError("model hash and inventory files must be non-empty and aligned")
    return {
        "status": "archived_and_load_validated",
        "selected_run_id": selected["selected_run_id"],
        "adapter_path": selected["adapter_path"],
        "adapter_model_sha256": selected["adapter_model_sha256"],
        "merged_model_path": load_validation["model_path"],
        "merged_model_load": load_validation["merged_model_load"],
        "merged_model_file_count": len(inventory_lines),
        "merged_model_total_bytes": int(total_bytes),
        "merged_model_manifest_path": str(hashes_path),
        "merged_model_manifest_sha256": sha256_file(hashes_path),
        "merged_model_inventory_path": str(inventory_path),
        "merged_model_inventory_sha256": sha256_file(inventory_path),
        "dpo_input_form": "merged_model",
        "opencompass_used_for_selection": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--load-validation", type=Path, required=True)
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--total-bytes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    load_validation = json.loads(args.load_validation.read_text(encoding="utf-8"))
    total_bytes = int(args.total_bytes.read_text(encoding="utf-8").split()[0])
    archive = build_archive(
        selected=selected,
        load_validation=load_validation,
        hashes_path=args.hashes,
        inventory_path=args.inventory,
        total_bytes=total_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(archive, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
