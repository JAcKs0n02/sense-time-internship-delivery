#!/usr/bin/env python3
"""Run the frozen OpenCompass CLI plan with isolated, auditable attempts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import subprocess
from typing import NamedTuple


EXPECTED_DATASETS = ["ceval_gen", "cmmlu_gen"]
EXPECTED_LABELS = ["base", "best_sft"]


class RunPlan(NamedTuple):
    label: str
    model_path: str
    work_dir: Path
    command: list[str]


def validate_config(config: dict) -> list[str]:
    errors: list[str] = []
    if config.get("dataset_configs") != EXPECTED_DATASETS:
        errors.append(f"dataset_configs must equal {EXPECTED_DATASETS}")
    models = config.get("models", [])
    labels = [model.get("label") for model in models]
    if labels != EXPECTED_LABELS:
        errors.append(f"model labels must equal {EXPECTED_LABELS}")
    paths = [model.get("path") for model in models]
    if any(not path for path in paths) or len(set(paths)) != 2:
        errors.append("model paths must be present and distinct")
    if config.get("dataset_source") != "ModelScope":
        errors.append("dataset_source must be ModelScope for this frozen run")
    for key in ("opencompass_executable", "max_seq_len", "max_out_len", "batch_size"):
        if not config.get(key):
            errors.append(f"missing config value: {key}")
    return errors


def build_run_plans(config: dict, work_root: Path) -> list[RunPlan]:
    errors = validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    plans: list[RunPlan] = []
    for model in config["models"]:
        label = model["label"]
        work_dir = work_root / label / "attempt-001"
        command = [
            config["opencompass_executable"],
            "--datasets",
            *config["dataset_configs"],
            "--hf-type",
            "chat",
            "--hf-path",
            model["path"],
            "--max-seq-len",
            str(config["max_seq_len"]),
            "--max-out-len",
            str(config["max_out_len"]),
            "--batch-size",
            str(config["batch_size"]),
            "-w",
            str(work_dir),
            "--dump-eval-details",
        ]
        plans.append(RunPlan(label, model["path"], work_dir, command))
    return plans


def load_config(path: Path) -> dict:
    namespace = runpy.run_path(str(path))
    config = namespace.get("CONFIG")
    if not isinstance(config, dict):
        raise ValueError("config module must define a CONFIG dictionary")
    return config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    plans = build_run_plans(config, args.work_root)
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.work_root / "formal_run_plan.json"
    if plan_path.exists():
        raise SystemExit(f"refusing to overwrite existing run plan: {plan_path}")
    for plan in plans:
        if plan.work_dir.exists():
            raise SystemExit(f"refusing to reuse attempt directory: {plan.work_dir}")
    plan_path.write_text(
        json.dumps(
            [
                {
                    "label": plan.label,
                    "model_path": plan.model_path,
                    "work_dir": str(plan.work_dir),
                    "command": plan.command,
                }
                for plan in plans
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_codes: list[int] = []
    for plan in plans:
        log_path = args.log_root / f"opencompass_formal_{plan.label}_attempt-001.log"
        started_at = utc_now()
        environment = os.environ.copy()
        environment["DATASET_SOURCE"] = config["dataset_source"]
        with log_path.open("w", encoding="utf-8") as log_stream:
            completed = subprocess.run(
                plan.command,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        ended_at = utc_now()
        plan.work_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "label": plan.label,
            "model_path": plan.model_path,
            "datasets": config["dataset_configs"],
            "status": "completed" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "started_at": started_at,
            "ended_at": ended_at,
            "command": plan.command,
            "log_path": str(log_path),
        }
        (plan.work_dir / "status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        exit_codes.append(completed.returncode)
    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
