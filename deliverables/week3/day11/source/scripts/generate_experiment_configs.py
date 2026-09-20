#!/usr/bin/env python3
"""Generate the nine isolated Week 3 experiment YAML files."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


RUN_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
GROUPS = {"rank", "learning_rate", "epoch"}


@dataclass(frozen=True)
class ExperimentSpec:
    run_id: str
    group: str
    lora_rank: int
    learning_rate: float
    num_train_epochs: float
    repeated_baseline: bool
    hypothesis: str

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "ExperimentSpec":
        required = {
            "run_id",
            "group",
            "lora_rank",
            "learning_rate",
            "num_train_epochs",
            "repeated_baseline",
            "hypothesis",
        }
        missing = sorted(required - row.keys())
        extra = sorted(row.keys() - required)
        if missing or extra:
            raise ValueError(
                f"invalid manifest row keys; missing={missing}, extra={extra}"
            )

        run_id = row["run_id"]
        group = row["group"]
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"invalid run_id: {run_id!r}")
        if group not in GROUPS:
            raise ValueError(f"invalid group for {run_id}: {group!r}")
        if not isinstance(row["lora_rank"], int) or row["lora_rank"] <= 0:
            raise ValueError(f"invalid lora_rank for {run_id}")
        if not isinstance(row["learning_rate"], (int, float)) or row["learning_rate"] <= 0:
            raise ValueError(f"invalid learning_rate for {run_id}")
        if not isinstance(row["num_train_epochs"], (int, float)) or row["num_train_epochs"] <= 0:
            raise ValueError(f"invalid num_train_epochs for {run_id}")
        if not isinstance(row["repeated_baseline"], bool):
            raise ValueError(f"invalid repeated_baseline for {run_id}")
        if not isinstance(row["hypothesis"], str) or not row["hypothesis"].strip():
            raise ValueError(f"empty hypothesis for {run_id}")

        return cls(
            run_id=run_id,
            group=group,
            lora_rank=row["lora_rank"],
            learning_rate=float(row["learning_rate"]),
            num_train_epochs=float(row["num_train_epochs"]),
            repeated_baseline=row["repeated_baseline"],
            hypothesis=row["hypothesis"].strip(),
        )


def load_specs(path: Path) -> list[ExperimentSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError("manifest root must contain a runs list")
    specs = [ExperimentSpec.from_mapping(row) for row in payload["runs"]]
    run_ids = [spec.run_id for spec in specs]
    duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate run_id values: {duplicates}")
    return specs


def load_base_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("base YAML root must be a mapping")
    return loaded


def generate_configs(
    base_config: dict[str, Any],
    specs: list[ExperimentSpec],
) -> dict[str, dict[str, Any]]:
    generated: dict[str, dict[str, Any]] = {}
    for spec in specs:
        config = copy.deepcopy(base_config)
        config["lora_rank"] = spec.lora_rank
        config["learning_rate"] = spec.learning_rate
        config["num_train_epochs"] = spec.num_train_epochs
        config["run_name"] = f"qwen25-7b-week3-{spec.run_id}"
        config["output_dir"] = (
            f"/root/autodl-tmp/qwen25-week3/runs/{spec.run_id}/adapter"
        )
        generated[spec.run_id] = config
    return generated


def write_configs(configs: dict[str, dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.yaml"))
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing YAML files: "
            + ", ".join(path.name for path in existing)
        )
    for run_id, config in configs.items():
        path = output_dir / f"{run_id}.yaml"
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate isolated YAML files for the Week 3 matrix."
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        specs = load_specs(args.matrix)
        base_config = load_base_config(args.base)
        configs = generate_configs(base_config, specs)
        write_configs(configs, args.output)
    except Exception as error:
        print(f"generation failed: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "generated": len(configs),
                "output_dir": str(args.output),
                "run_ids": list(configs),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
