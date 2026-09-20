#!/usr/bin/env python3
"""Derive a two-model, one-subject smoke config from a CLI-expanded config."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path


def build_smoke_payload(source: dict, *, base_path: str, best_path: str) -> dict:
    datasets = source.get("datasets", [])
    models = source.get("models", [])
    if not datasets or not models:
        raise ValueError("expanded config must contain datasets and models")
    payload = deepcopy(source)
    payload.pop("work_dir", None)
    payload["datasets"] = [deepcopy(datasets[0])]
    base_model = deepcopy(models[0])
    best_model = deepcopy(models[0])
    base_model.update(abbr="base", path=base_path, peft_path=None)
    best_model.update(abbr="best_sft", path=best_path, peft_path=None)
    payload["models"] = [base_model, best_model]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--best-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    from mmengine import Config

    args = parse_args()
    source = Config.fromfile(args.source_config, format_python_code=False).to_dict()
    payload = build_smoke_payload(
        source,
        base_path=args.base_path,
        best_path=args.best_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Config(payload).dump(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
