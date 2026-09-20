#!/usr/bin/env python3
"""Build the frozen train-only Day26 v7 repair continuation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


REPAIR_REPEATS = {"natural_scene": 3, "ui": 3, "ocr": 1}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_source_indices(internal_rows: list[dict]) -> list[int]:
    indices: list[int] = []
    for index, row in enumerate(internal_rows):
        if row.get("split") != "train":
            raise ValueError("repair input must contain train rows only")
        repeats = REPAIR_REPEATS.get(row.get("category"), 0)
        indices.extend([index] * repeats)
    return indices


def build_repair_dataset(
    public_rows: list[dict], internal_rows: list[dict], *, seed: int
) -> list[dict]:
    if len(public_rows) != len(internal_rows) or len(public_rows) != 200:
        raise ValueError("repair input must be the aligned 200-row v7 training export")
    indices = repair_source_indices(internal_rows)
    random.Random(seed).shuffle(indices)
    weighted = [public_rows[index] for index in indices]
    if len(weighted) != 280:
        raise ValueError("repair continuation set must contain exactly 280 rows")
    if any(
        "dev" in image or "final" in image
        for row in weighted
        for image in row.get("images", [])
    ):
        raise ValueError("held-out media cannot enter the repair continuation set")
    return weighted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2608)
    args = parser.parse_args()

    public_rows = json.loads(args.public.read_text(encoding="utf-8"))
    internal_rows = json.loads(args.internal.read_text(encoding="utf-8"))
    weighted = build_repair_dataset(public_rows, internal_rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(weighted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = Counter(
        internal_rows[index]["category"] for index in repair_source_indices(internal_rows)
    )
    manifest = {
        "schema_version": "day26-v7-repair-dataset-1",
        "status": "PRE_REGISTERED",
        "purpose": "repair natural-scene and UI regressions without held-out reuse",
        "seed": args.seed,
        "parent_public_sha256": sha256(args.public),
        "parent_internal_sha256": sha256(args.internal),
        "row_count": len(weighted),
        "category_counts": dict(sorted(counts.items())),
        "source_splits": ["train"],
        "output_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
