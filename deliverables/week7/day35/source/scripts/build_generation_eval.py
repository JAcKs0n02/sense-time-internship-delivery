#!/usr/bin/env python3
"""Freeze the Week 7 twenty-prompt generation benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from week7_deployment.datasets import freeze_generation_prompts
from week7_deployment.hashing import sha256_file


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety-source", type=Path, required=True)
    parser.add_argument("--business-source", type=Path, required=True)
    parser.add_argument("--general-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = freeze_generation_prompts(
        _read_object(args.safety_source),
        _read_object(args.business_source),
        _read_object(args.general_source),
    )
    payload = {
        "schema_version": "1.0",
        "protocol": "week7_generation_eval_v1",
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
            "seed": 42,
        },
        "sources": [
            {"path": path.as_posix(), "sha256": sha256_file(path)}
            for path in (
                args.safety_source,
                args.business_source,
                args.general_source,
            )
        ],
        "prompts": prompts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"row_count": len(prompts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
