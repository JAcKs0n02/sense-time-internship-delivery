#!/usr/bin/env python3
"""Freeze the Day 14 selection evidence before OpenCompass is executed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contains_opencompass_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            "opencompass" in str(key).lower() or contains_opencompass_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_opencompass_key(item) for item in value)
    return False


def parse_adapter_weight_hash(path: Path) -> str:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split(maxsplit=1)
        if len(parts) == 2 and Path(parts[1].strip()).name == "adapter_model.safetensors":
            return parts[0]
    raise ValueError("adapter manifest has no adapter_model.safetensors entry")


def freeze_selected_model_input(
    *,
    best_model_path: Path,
    human_scores_path: Path,
    reviewer_paths: Iterable[Path],
    adapter_manifest_path: Path,
    remote_root: str,
    selection_commit: str,
    selection_timestamp: str,
) -> dict:
    best = json.loads(best_model_path.read_text(encoding="utf-8"))
    if contains_opencompass_key(best):
        raise ValueError("Day 14 selection input must not contain OpenCompass results")

    run_id = best.get("model_id") or best.get("run_id")
    if not run_id:
        raise ValueError("best model record has neither model_id nor run_id")
    adapter_path = str(best.get("adapter_path", "")).strip()
    if not adapter_path:
        raise ValueError("best model record has no adapter_path")

    remote_adapter = Path(remote_root) / adapter_path
    reviewers = list(reviewer_paths)
    if len(reviewers) != 2:
        raise ValueError("exactly two reviewer score files are required")

    return {
        "status": "frozen_before_opencompass",
        "selected_run_id": run_id,
        "adapter_path": str(remote_adapter),
        "adapter_model_sha256": parse_adapter_weight_hash(adapter_manifest_path),
        "human_weighted_total": best.get("weighted_total"),
        "selection_source": "Day 14 two-rater weighted human score",
        "selection_commit": selection_commit,
        "selection_timestamp": selection_timestamp,
        "opencompass_used_for_selection": False,
        "source_sha256": {
            "best_model_json": sha256_file(best_model_path),
            "human_scores_csv": sha256_file(human_scores_path),
            "reviewer_score_files": [sha256_file(path) for path in reviewers],
            "adapter_manifest": sha256_file(adapter_manifest_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-model", type=Path, required=True)
    parser.add_argument("--human-scores", type=Path, required=True)
    parser.add_argument("--reviewer", type=Path, action="append", required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--selection-commit", required=True)
    parser.add_argument("--selection-timestamp", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = freeze_selected_model_input(
        best_model_path=args.best_model,
        human_scores_path=args.human_scores,
        reviewer_paths=args.reviewer,
        adapter_manifest_path=args.adapter_manifest,
        remote_root=args.remote_root,
        selection_commit=args.selection_commit,
        selection_timestamp=args.selection_timestamp,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
