#!/usr/bin/env python3
"""Validate the annotated Day 8 LLaMA-Factory training configuration."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """A YAML loader that rejects duplicate mapping keys."""


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


ACTIVE_KEY_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(?:\s|$)")
ANNOTATION_MARKERS = ("作用", "当前值", "调整影响")


def _load_unique_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.load(file, Loader=UniqueKeyLoader)
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a YAML mapping")
    return loaded


def _annotation_report(path: Path) -> tuple[list[str], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    active_keys: list[str] = []
    missing: list[str] = []

    for index, line in enumerate(lines):
        match = ACTIVE_KEY_PATTERN.match(line)
        if not match:
            continue
        key = match.group(1)
        active_keys.append(key)

        comments: list[str] = []
        cursor = index - 1
        while cursor >= 0 and lines[cursor].lstrip().startswith("#"):
            comments.append(lines[cursor].lstrip()[1:].strip())
            cursor -= 1
        comment_text = "\n".join(reversed(comments))
        absent_markers = [
            marker for marker in ANNOTATION_MARKERS if marker not in comment_text
        ]
        if absent_markers:
            missing.append(f"{key} (missing: {', '.join(absent_markers)})")

    return active_keys, missing


def _validate_values(
    config: dict[str, Any],
    dataset_info: dict[str, Any],
    expected_count: int,
) -> list[str]:
    errors: list[str] = []
    expected_values = {
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "template": "qwen",
        "cutoff_len": 2048,
        "max_samples": expected_count,
        "logging_steps": 1,
        "plot_loss": True,
        "report_to": "tensorboard",
        "seed": 42,
    }
    for key, expected in expected_values.items():
        if config.get(key) != expected:
            errors.append(
                f"{key}: expected {expected!r}, got {config.get(key)!r}"
            )

    required_keys = (
        "model_name_or_path",
        "lora_rank",
        "lora_alpha",
        "learning_rate",
        "dataset",
        "dataset_dir",
        "output_dir",
        "run_name",
    )
    for key in required_keys:
        if key not in config:
            errors.append(f"missing required key: {key}")

    dataset_name = config.get("dataset")
    if not isinstance(dataset_name, str) or dataset_name not in dataset_info:
        errors.append(
            f"dataset {dataset_name!r} is not registered in dataset_info.json"
        )
    else:
        registration = dataset_info[dataset_name]
        expected_columns = {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        }
        if registration.get("columns") != expected_columns:
            errors.append(
                "dataset column mapping must be "
                "instruction/input/output -> prompt/query/response"
            )

    return errors


def validate(
    config_path: Path,
    dataset_info_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    config = _load_unique_yaml(config_path)
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_info, dict):
        raise ValueError("dataset_info.json root must be an object")

    active_keys, missing_annotations = _annotation_report(config_path)
    errors = _validate_values(config, dataset_info, expected_count)
    if missing_annotations:
        errors.append(
            "annotation requirements not met for: "
            + "; ".join(missing_annotations)
        )

    return {
        "valid": not errors,
        "config_path": str(config_path),
        "dataset_info_path": str(dataset_info_path),
        "expected_count": expected_count,
        "active_parameter_count": len(active_keys),
        "all_parameters_fully_annotated": not missing_annotations,
        "missing_annotations": missing_annotations,
        "dataset_name": config.get("dataset"),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Day 8 annotated LLaMA-Factory YAML."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate(
            config_path=args.config,
            dataset_info_path=args.dataset_info,
            expected_count=args.expected_count,
        )
    except Exception as error:
        report = {
            "valid": False,
            "errors": [str(error)],
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"validation failed: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")

    if not report["valid"]:
        print(
            "validation failed: " + "; ".join(report["errors"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
