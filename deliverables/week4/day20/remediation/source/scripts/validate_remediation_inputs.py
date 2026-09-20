#!/usr/bin/env python3
"""Validate remediation split contracts and train/dev/teacher contamination."""

from __future__ import annotations

import argparse
import difflib
import json
import unicodedata
from pathlib import Path
from typing import Any


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in value if not unicodedata.category(ch).startswith(("P", "Z", "C")))


def _prompt(row: dict[str, Any]) -> str:
    return row.get("conversations", [{}])[-1].get("value", "")


def _teacher_rows(teacher: Any) -> list[dict[str, Any]]:
    return teacher.get("prompts", []) if isinstance(teacher, dict) else teacher


def validate_inputs(full, train, validation, dev, teacher, *, near_threshold: float = 0.90):
    errors: set[str] = set()
    full_ids = [row.get("id") for row in full]
    train_ids = {row.get("id") for row in train}
    validation_ids = {row.get("id") for row in validation}
    if len(full) != 870 or len(set(full_ids)) != 870:
        errors.add("full_count_or_ids")
    if len(train) != 783 or len(validation) != 87 or train_ids & validation_ids or train_ids | validation_ids != set(full_ids):
        errors.add("split_contract")
    type_counts = {kind: sum(row.get("prompt_type") == kind for row in dev) for kind in ("harmful", "benign", "business")}
    if len(dev) != 35 or type_counts != {"harmful": 20, "benign": 10, "business": 5}:
        errors.add("dev_contract")

    held_out = [*dev, *_teacher_rows(teacher)]
    held_norm = [(row.get("id", ""), normalize(row.get("text", ""))) for row in held_out]
    exact = []
    near = []
    for row in full:
        candidate = normalize(_prompt(row))
        for eval_id, prompt in held_norm:
            if candidate == prompt and candidate:
                exact.append({"record_id": row.get("id"), "eval_id": eval_id})
            elif candidate and prompt:
                ratio = difflib.SequenceMatcher(None, candidate, prompt, autojunk=False).ratio()
                if ratio >= near_threshold:
                    near.append({"record_id": row.get("id"), "eval_id": eval_id, "similarity": round(ratio, 6)})
    if exact:
        errors.add("exact_contamination")
    if near:
        errors.add("near_contamination")
    return {
        "valid": not errors,
        "errors": sorted(errors),
        "counts": {"full": len(full), "train": len(train), "validation": len(validation), "dev": len(dev), "dev_harmful": type_counts["harmful"], "dev_benign": type_counts["benign"], "dev_business": type_counts["business"]},
        "exact_contamination_count": len(exact),
        "near_contamination_count": len(near),
        "exact_contamination": exact,
        "near_contamination": near,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("full", "train", "validation", "dev", "teacher"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = {name: json.loads(getattr(args, name).read_text(encoding="utf-8")) for name in ("full", "train", "validation", "dev", "teacher")}
    report = validate_inputs(**values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

