#!/usr/bin/env python3
"""GPU-side model inventory, tokenizer length, and ranking-schema preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    expected = json.loads(args.inventory.read_text(encoding="utf-8"))["files"]
    actual = {str(path.relative_to(args.model)): path.stat().st_size for path in sorted(args.model.rglob("*")) if path.is_file()}
    if actual != expected:
        raise SystemExit("SFT inventory mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    lengths: list[int] = []
    counts: dict[str, int] = {}
    for label, path in (("train", args.train), ("validation", args.validation)):
        rows = json.loads(path.read_text(encoding="utf-8")); counts[label] = len(rows)
        for row in rows:
            if row["conversations"][-1]["from"] != "human" or row["chosen"]["from"] != "gpt" or row["rejected"]["from"] != "gpt":
                raise SystemExit(f"ranking schema mismatch: {row.get('id')}")
            history = [{"role": "user" if item["from"] == "human" else "assistant", "content": item["value"]} for item in row["conversations"]]
            for side in ("chosen", "rejected"):
                messages = [*history, {"role": "assistant", "content": row[side]["value"]}]
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                lengths.append(len(tokenizer(rendered, add_special_tokens=False).input_ids))
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"], text=True).strip()
    payload: dict[str, Any] = {
        "valid": counts == {"train": 783, "validation": 87} and max(lengths) <= 2048,
        "gpu": gpu,
        "sft_model_path": str(args.model),
        "sft_file_count": len(actual),
        "sft_total_bytes": sum(actual.values()),
        "sft_manifest_sha256_from_day19": "e4512f7dd0f85b2f1d8154d123aec176295f557046142aab406a030c8e76e971",
        "train_sha256": sha256(args.train),
        "validation_sha256": sha256(args.validation),
        "counts": counts,
        "encoded_sequence_count": len(lengths),
        "max_encoded_tokens": max(lengths),
        "over_cutoff_2048": sum(length > 2048 for length in lengths),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
