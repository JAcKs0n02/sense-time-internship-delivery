#!/usr/bin/env python3
"""Independently load a merged Transformers model from local files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def success_payload(
    *, model_path: Path, model_class: str, tokenizer_class: str
) -> dict:
    return {
        "status": "PASS",
        "merged_model_load": "PASS",
        "model_path": str(model_path),
        "model_class": model_class,
        "tokenizer_class": tokenizer_class,
        "local_files_only": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto", choices=("auto", "cpu"))
    return parser.parse_args()


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    path = args.model_path.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        device_map=args.device_map,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    payload = success_payload(
        model_path=path,
        model_class=model.__class__.__name__,
        tokenizer_class=tokenizer.__class__.__name__,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
