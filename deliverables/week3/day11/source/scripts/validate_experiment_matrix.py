#!/usr/bin/env python3
"""Validate Week 3 experiment manifest and generated YAML isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


TARGET_FIELD = {
    "rank": "lora_rank",
    "learning_rate": "learning_rate",
    "epoch": "num_train_epochs",
}
METADATA_FIELDS = {"run_name", "output_dir"}
LEARNING_FIELDS = {"lora_rank", "learning_rate", "num_train_epochs"}
EXPECTED_GROUP_VALUES = {
    "rank": {8, 32, 64},
    "learning_rate": {1e-4, 2e-4, 5e-5},
    "epoch": {2.0, 3.0, 5.0},
}
BASELINE_TUPLE = (8, 1e-4, 3.0)
BASELINE_RUN_IDS = ["rank-r8", "lr-1e-4", "epoch-e3"]


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_unique_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.load(file, Loader=UniqueKeyLoader)
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return loaded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError("manifest root must contain a runs list")
    return payload


def expected_config(base: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    config = dict(base)
    config["lora_rank"] = row["lora_rank"]
    config["learning_rate"] = float(row["learning_rate"])
    config["num_train_epochs"] = float(row["num_train_epochs"])
    config["run_name"] = f"qwen25-7b-week3-{row['run_id']}"
    config["output_dir"] = (
        f"/root/autodl-tmp/qwen25-week3/runs/{row['run_id']}/adapter"
    )
    return config


def validate(
    matrix_path: Path,
    base_path: Path,
    config_dir: Path,
) -> dict[str, Any]:
    manifest = load_manifest(matrix_path)
    rows = manifest["runs"]
    base = load_unique_yaml(base_path)
    errors: list[str] = []

    run_ids = [row.get("run_id") for row in rows]
    duplicates = sorted(
        {run_id for run_id in run_ids if run_ids.count(run_id) > 1}
    )
    if duplicates:
        errors.append(f"duplicate run_id values: {duplicates}")

    group_counts = Counter(row.get("group") for row in rows)
    expected_group_counts = Counter({
        "rank": 3,
        "learning_rate": 3,
        "epoch": 3,
    })
    if group_counts != expected_group_counts:
        errors.append(
            f"group counts mismatch: expected {dict(expected_group_counts)}, "
            f"got {dict(group_counts)}"
        )
    if len(rows) != 9:
        errors.append(f"run count mismatch: expected 9, got {len(rows)}")

    for group, field in TARGET_FIELD.items():
        actual_values = {
            float(row[field]) if field != "lora_rank" else row[field]
            for row in rows
            if row.get("group") == group
        }
        if actual_values != EXPECTED_GROUP_VALUES[group]:
            errors.append(
                f"{group} values mismatch: expected "
                f"{sorted(EXPECTED_GROUP_VALUES[group])}, got {sorted(actual_values)}"
            )

    repeated_baselines = [
        row.get("run_id") for row in rows if row.get("repeated_baseline") is True
    ]
    if repeated_baselines != BASELINE_RUN_IDS:
        errors.append(
            f"repeated baseline run IDs mismatch: expected {BASELINE_RUN_IDS}, "
            f"got {repeated_baselines}"
        )

    config_paths = sorted(config_dir.glob("*.yaml"))
    expected_names = {f"{run_id}.yaml" for run_id in run_ids if isinstance(run_id, str)}
    actual_names = {path.name for path in config_paths}
    if actual_names != expected_names:
        errors.append(
            f"config filenames mismatch: expected {sorted(expected_names)}, "
            f"got {sorted(actual_names)}"
        )

    config_hashes: dict[str, str] = {}
    output_dirs: list[str] = []
    run_names: list[str] = []
    for row in rows:
        run_id = row.get("run_id")
        if not isinstance(run_id, str):
            errors.append(f"invalid run_id in row: {row!r}")
            continue
        path = config_dir / f"{run_id}.yaml"
        if not path.exists():
            continue
        try:
            actual = load_unique_yaml(path)
        except Exception as error:
            errors.append(f"{run_id}: {error}")
            continue

        expected = expected_config(base, row)
        all_keys = sorted(set(actual) | set(expected))
        for key in all_keys:
            if actual.get(key) != expected.get(key):
                errors.append(
                    f"{run_id}: {key} expected {expected.get(key)!r}, "
                    f"got {actual.get(key)!r}"
                )

        target = TARGET_FIELD.get(row.get("group"))
        for field in LEARNING_FIELDS - {target}:
            if actual.get(field) != base.get(field):
                errors.append(
                    f"{run_id}: non-target learning field {field} differs "
                    f"from baseline"
                )

        output_dirs.append(str(actual.get("output_dir")))
        run_names.append(str(actual.get("run_name")))
        config_hashes[run_id] = sha256(path)

    if len(output_dirs) != len(set(output_dirs)):
        errors.append("output_dir values must be unique")
    if len(run_names) != len(set(run_names)):
        errors.append("run_name values must be unique")

    baseline_rows = {
        row["run_id"]: row for row in rows if row.get("run_id") in BASELINE_RUN_IDS
    }
    for run_id in BASELINE_RUN_IDS:
        row = baseline_rows.get(run_id)
        if row is None:
            continue
        values = (
            row.get("lora_rank"),
            float(row.get("learning_rate")),
            float(row.get("num_train_epochs")),
        )
        if values != BASELINE_TUPLE:
            errors.append(
                f"{run_id}: repeated baseline expected {BASELINE_TUPLE}, got {values}"
            )

    combinations = {
        (
            row.get("lora_rank"),
            float(row.get("learning_rate")),
            float(row.get("num_train_epochs")),
        )
        for row in rows
    }

    return {
        "valid": not errors,
        "run_count": len(rows),
        "group_counts": dict(group_counts),
        "unique_hyperparameter_combinations": len(combinations),
        "repeated_baseline_run_ids": repeated_baselines,
        "allowed_learning_fields": sorted(LEARNING_FIELDS),
        "allowed_metadata_fields": sorted(METADATA_FIELDS),
        "matrix_sha256": sha256(matrix_path),
        "base_config_sha256": sha256(base_path),
        "config_sha256": config_hashes,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Week 3 single-variable experiment matrix."
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate(args.matrix, args.base, args.config_dir)
    except Exception as error:
        report = {"valid": False, "errors": [str(error)]}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["valid"]:
        print("validation failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
