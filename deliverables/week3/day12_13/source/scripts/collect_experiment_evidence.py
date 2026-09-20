#!/usr/bin/env python3
"""Collect small Day 12–13 audit files without copying model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT_ALLOWLIST = {
    "train.log",
    "status.json",
    "gpu_memory.csv",
    "source_config.yaml",
    "runtime_config.yaml",
    "config_diff.json",
    "preflight.json",
    "preflight.txt",
    "environment.txt",
    "adapter_files.sha256",
}
ADAPTER_ALLOWLIST = {"trainer_state.json", "train_results.json"}
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_status(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"status root must be an object: {path}")
    return payload


def attempt_paths(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.glob("attempt-[0-9][0-9][0-9]") if path.is_dir())


def required_names(status: str) -> set[str]:
    base = {
        "status.json",
        "source_config.yaml",
        "runtime_config.yaml",
        "config_diff.json",
        "preflight.txt",
        "environment.txt",
        "adapter_files.sha256",
    }
    if status != "blocked_preflight":
        base.update({"train.log", "gpu_memory.csv"})
    if status == "completed":
        base.update(ADAPTER_ALLOWLIST)
    return base


def collect_evidence(run_root: Path, output_root: Path, run_ids: list[str]) -> list[str]:
    errors: list[str] = []
    if output_root.exists() and any(output_root.iterdir()):
        return [f"output directory is not empty: {output_root}"]
    output_root.mkdir(parents=True, exist_ok=True)

    for run_id in run_ids:
        run_dir = run_root / run_id
        attempts = attempt_paths(run_dir)
        if not attempts:
            errors.append(f"run has no attempts: {run_id}")
            continue
        for attempt in attempts:
            symlinks = [path for path in attempt.rglob("*") if path.is_symlink()]
            if symlinks:
                errors.extend(f"symlink rejected: {path}" for path in symlinks)
                continue
            destination = output_root / run_id / attempt.name
            destination.mkdir(parents=True, exist_ok=True)

            for name in sorted(ROOT_ALLOWLIST):
                source = attempt / name
                if source.is_file():
                    shutil.copy2(source, destination / name)
            adapter = attempt / "adapter"
            for name in sorted(ADAPTER_ALLOWLIST):
                source = adapter / name
                if source.is_file():
                    shutil.copy2(source, destination / name)

            status_path = attempt / "status.json"
            if not status_path.is_file():
                errors.append(f"missing status.json: {run_id}/{attempt.name}")
                continue
            try:
                status = str(load_status(status_path).get("status"))
            except Exception as error:
                errors.append(str(error))
                continue
            missing = sorted(name for name in required_names(status) if not (destination / name).is_file())
            errors.extend(f"missing evidence: {run_id}/{attempt.name}/{name}" for name in missing)

    forbidden = [
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and (path.suffix.lower() in FORBIDDEN_SUFFIXES or "checkpoint-" in path.as_posix())
    ]
    errors.extend(f"forbidden evidence copied: {path}" for path in forbidden)
    return errors


def write_checksum_manifest(output_root: Path, checksum_path: Path) -> None:
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    base = checksum_path.parent
    lines = [
        f"{sha256_file(path)}  {path.relative_to(base)}"
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    ]
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_run_ids(matrix_path: Path) -> list[str]:
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    return [row["run_id"] for row in payload["runs"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the Week 3 raw log evidence folder.")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checksum-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = collect_evidence(args.run_root, args.output_root, load_run_ids(args.matrix))
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    write_checksum_manifest(args.output_root, args.checksum_output)
    print(json.dumps({"valid": True, "run_count": len(load_run_ids(args.matrix))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
