#!/usr/bin/env python3
"""Collect a compact, reviewable Day26 evidence bundle from the GPU workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


TEXT_SUFFIXES = {".csv", ".exit", ".json", ".jsonl", ".log", ".md", ".txt", ".yaml"}
MAX_TEXT_BYTES = 1_000_000
PRIVATE_NAME_FRAGMENTS = ("private_blind_key",)
ADAPTER_FILES = ("README.md", "adapter_config.json", "adapter_model.safetensors")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def collect_evidence(root: Path, adapter_dir: Path) -> dict[str, object]:
    root = root.resolve()
    adapter_dir = adapter_dir.resolve()
    if not root.is_dir() or not adapter_dir.is_dir():
        raise ValueError("root and adapter directory must exist")
    if root not in adapter_dir.parents:
        raise ValueError("adapter directory must be inside the Day26 root")

    text_artifacts: dict[str, dict[str, object]] = {}
    scan_roots = [root / "results", adapter_dir.parent]
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        for path in sorted(scan_root.glob("*")):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            if any(fragment in relative for fragment in PRIVATE_NAME_FRAGMENTS):
                continue
            if path.stat().st_size > MAX_TEXT_BYTES:
                continue
            record = file_record(path)
            record["content"] = path.read_text(encoding="utf-8", errors="replace")
            text_artifacts[relative] = record

    selected_adapter_files: dict[str, dict[str, object]] = {}
    for name in ADAPTER_FILES:
        path = adapter_dir / name
        if path.is_file():
            selected_adapter_files[name] = file_record(path)
    if "adapter_config.json" not in selected_adapter_files:
        raise ValueError("selected adapter is missing adapter_config.json")
    if "adapter_model.safetensors" not in selected_adapter_files:
        raise ValueError("selected adapter is missing adapter_model.safetensors")

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "day26_root": root.as_posix(),
        "selected_adapter_dir": adapter_dir.as_posix(),
        "selected_adapter_files": selected_adapter_files,
        "text_artifacts": text_artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = collect_evidence(args.root, args.adapter_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "text_artifact_count": len(payload["text_artifacts"]),
        "adapter_file_count": len(payload["selected_adapter_files"]),
        "output": args.output.as_posix(),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
