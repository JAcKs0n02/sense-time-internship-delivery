#!/usr/bin/env python3
"""Normalize completed OpenCompass subject results into four benchmark rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


MODEL_LABELS = ("base", "best_sft")
DATASET_PREFIXES = {"ceval-": "ceval", "cmmlu-": "cmmlu"}


def dataset_for(filename: str) -> str | None:
    return next(
        (dataset for prefix, dataset in DATASET_PREFIXES.items() if filename.startswith(prefix)),
        None,
    )


def collect_model_results(formal_root: Path, label: str) -> list[dict]:
    attempt = formal_root / label / "attempt-001"
    status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed" or status.get("exit_code") != 0:
        raise ValueError(f"formal OpenCompass run is not completed: {label}")
    result_dirs = [path for path in attempt.glob("*/results/*") if path.is_dir()]
    if len(result_dirs) != 1:
        raise ValueError(f"expected one result directory for {label}, found {len(result_dirs)}")
    result_dir = result_dirs[0]
    by_dataset: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for path in sorted(result_dir.glob("*.json")):
        dataset = dataset_for(path.name)
        if dataset is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        accuracy = float(payload["accuracy"])
        question_count = len(payload.get("details", {}))
        if question_count <= 0 or not 0.0 <= accuracy <= 100.0:
            raise ValueError(f"invalid subject result: {path}")
        by_dataset[dataset].append((accuracy, question_count))
    rows: list[dict] = []
    for dataset in ("ceval", "cmmlu"):
        subjects = by_dataset.get(dataset, [])
        if not subjects:
            raise ValueError(f"missing {dataset} results for {label}")
        total_questions = sum(count for _, count in subjects)
        weighted = sum(score * count for score, count in subjects) / total_questions
        rows.append(
            {
                "model": label,
                "dataset": dataset,
                "score": round(weighted, 6),
                "macro_subject_score": round(mean(score for score, _ in subjects), 6),
                "subject_count": len(subjects),
                "question_count": total_questions,
                "status": "completed",
                "aggregation": "sample_weighted_accuracy",
                "result_dir": str(result_dir),
            }
        )
    return rows


def collect_formal_results(formal_root: Path) -> list[dict]:
    rows: list[dict] = []
    for label in MODEL_LABELS:
        rows.extend(collect_model_results(formal_root, label))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_formal_results(args.formal_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"results": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
