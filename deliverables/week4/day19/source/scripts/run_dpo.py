#!/usr/bin/env python3
"""Launch a Day19 DPO run in a new, immutable attempt directory."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Sequence

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def next_attempt_dir(run_root: Path) -> Path:
    index = 1
    while (run_root / f"attempt_{index:03d}").exists():
        index += 1
    return run_root / f"attempt_{index:03d}"


def create_attempt_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def build_runtime_config(config: dict[str, Any], attempt: Path) -> dict[str, Any]:
    runtime = copy.deepcopy(config)
    runtime["output_dir"] = str(attempt / "trainer_output")
    runtime["run_name"] = f"day19_{attempt.name}"
    runtime["overwrite_output_dir"] = False
    return runtime


def write_yaml_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def environment_snapshot() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "working_directory": os.getcwd(),
        "llamafactory_cli": shutil.which("llamafactory-cli"),
        "nvidia_smi_returncode": gpu.returncode,
        "gpu": gpu.stdout.strip(),
        "gpu_error": gpu.stderr.strip(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = load_yaml(args.config)
    attempt = next_attempt_dir(args.run_root)
    create_attempt_dir(attempt)
    runtime = build_runtime_config(source, attempt)
    runtime_path = attempt / "runtime.yaml"
    write_yaml_exclusive(runtime_path, runtime)
    write_json(attempt / "environment.json", environment_snapshot())

    command = ["llamafactory-cli", "train", str(runtime_path)]
    write_json(
        attempt / "launch.json",
        {"command": command, "source_config": str(args.config.resolve())},
    )
    if args.dry_run:
        print(attempt)
        return 0
    if not shutil.which("llamafactory-cli"):
        write_json(attempt / "status.json", {"status": "failed", "reason": "CLI missing"})
        return 127

    with (attempt / "launch.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    write_json(
        attempt / "status.json",
        {"status": "completed" if result.returncode == 0 else "failed", "returncode": result.returncode},
    )
    print(attempt)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
