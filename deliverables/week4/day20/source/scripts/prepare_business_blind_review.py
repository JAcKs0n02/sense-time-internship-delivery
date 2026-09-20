#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any


EXPECTED_IDS = [f"w4-eval-business-{index:02d}" for index in range(1, 6)]
MODEL_ALIASES = ("sft_only", "sft_dpo")


def build_blind_rows(records: list[dict[str, Any]], seed: int = 42) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    business = [row for row in records if row.get("prompt_type") == "business"]
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in business:
        key = (str(row.get("prompt_id")), str(row.get("model_alias")))
        if key in lookup:
            raise ValueError("duplicate business model-prompt pair")
        lookup[key] = row
    expected = {(prompt_id, alias) for prompt_id in EXPECTED_IDS for alias in MODEL_ALIASES}
    if set(lookup) != expected:
        raise ValueError("business responses must contain exactly both models for five prompts")
    rng = random.Random(seed)
    blind_rows: list[dict[str, str]] = []
    mapping: list[dict[str, str]] = []
    for prompt_id in EXPECTED_IDS:
        aliases = list(MODEL_ALIASES)
        rng.shuffle(aliases)
        for label, alias in zip(("A", "B"), aliases):
            row = lookup[(prompt_id, alias)]
            if row.get("status") != "completed" or not str(row.get("response", "")).strip():
                raise ValueError(f"{prompt_id}/{alias} is not a completed nonempty response")
            blind_rows.append({"prompt_id": prompt_id, "prompt_text": str(row["prompt_text"]), "candidate_label": label, "response": str(row["response"])})
            mapping.append({"prompt_id": prompt_id, "candidate_label": label, "model_alias": alias})
    return blind_rows, mapping


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the deterministic blinded Day 20 business review.")
    parser.add_argument("--responses", type=Path, nargs=2, required=True)
    parser.add_argument("--blind-output", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        records = [row for path in args.responses for row in read_jsonl(path)]
        blind_rows, mapping = build_blind_rows(records, args.seed)
        write_csv(args.blind_output, blind_rows)
        args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
        args.mapping_output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = {"valid": True, "seed": args.seed, "candidate_count": len(blind_rows), "blinded_sha256": hashlib.sha256(args.blind_output.read_bytes()).hexdigest(), "identity_fields_excluded": ["model_alias", "model_dir"]}
        args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
