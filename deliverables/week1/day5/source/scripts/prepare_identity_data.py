#!/usr/bin/env python3
"""Resolve the two documented placeholders in LLaMA-Factory identity data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(template_path: Path, output_path: Path, name: str, author: str) -> dict:
    rows = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("identity template must be a non-empty JSON list")
    original_placeholders = 0
    resolved = []
    for index, row in enumerate(rows):
        if set(row) != {"instruction", "input", "output"}:
            raise ValueError(f"unexpected keys in row {index}: {sorted(row)}")
        rendered = dict(row)
        original_placeholders += rendered["output"].count("{{name}}")
        original_placeholders += rendered["output"].count("{{author}}")
        rendered["output"] = rendered["output"].replace("{{name}}", name)
        rendered["output"] = rendered["output"].replace("{{author}}", author)
        resolved.append(rendered)
    if original_placeholders == 0:
        raise ValueError("identity template contains no expected placeholders")
    unresolved = sum(
        "{{name}}" in row["output"] or "{{author}}" in row["output"]
        for row in resolved
    )
    if unresolved:
        raise ValueError(f"{unresolved} rows still contain placeholders")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "samples": len(resolved),
        "name": name,
        "author": author,
        "placeholder_occurrences_replaced": original_placeholders,
        "template_sha256": sha256(template_path),
        "resolved_sha256": sha256(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--metadata", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metadata = prepare(args.template, args.output, args.name, args.author)
    rendered = json.dumps(metadata, ensure_ascii=False, indent=2)
    print(rendered)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(rendered + "\n", encoding="utf-8")
