#!/usr/bin/env python3
"""Run the pinned LLaMA-Factory registry/parser/alignment smoke and sign its receipt."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from day31_harness import (
    LLAMAFACTORY_REVISION,
    LLAMAFACTORY_VERSION,
    canonical_json,
    official_profile_spec,
    sha256_file,
)


DAY31 = Path(__file__).resolve().parents[1]


def align_registered_datasets(
    temp: Path,
    get_dataset_list: Callable[[list[str], str], Any],
    align_dataset: Callable[[Any, Any, Any, Any], Any],
    load_dataset: Callable[..., Any],
    dataset_names: list[str] | None = None,
) -> list[tuple[int, str]]:
    """Execute the v0.9.3 registry, loader, and converter APIs over frozen data."""
    data_args = SimpleNamespace(
        media_dir=str(temp),
        streaming=False,
        preprocessing_num_workers=1,
        overwrite_cache=True,
    )
    training_args = SimpleNamespace(local_process_index=0)
    names = dataset_names or ["week6_tool_sft_train", "week6_tool_sft_eval_frozen"]
    attrs = get_dataset_list(names, str(temp))
    projections: list[tuple[int, str]] = []
    for attr in attrs:
        dataset = load_dataset("json", data_files=str(temp / attr.dataset_name), split="train")
        aligned = align_dataset(dataset, attr, data_args, training_args)
        rows = list(aligned)
        projection = [
            {"prompt": row.get("_prompt"), "response": row.get("_response"), "tools": row.get("_tools")}
            for row in rows
        ]
        if not rows or any(not item["prompt"] or not item["response"] or not item["tools"] for item in projection):
            raise ValueError("official aligned projection empty")
        projections.append((len(rows), hashlib.sha256(canonical_json(projection).encode()).hexdigest()))
    return projections


def _module_identity(module: Any) -> dict[str, str]:
    path = Path(inspect.getfile(module)).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DAY31 / "data")
    parser.add_argument("--dataset-info", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("legacy", "v2"), default="legacy")
    args = parser.parse_args()
    try:
        if importlib.metadata.version("llamafactory") != LLAMAFACTORY_VERSION:
            raise ValueError("LLaMA-Factory version mismatch")
        root = Path(inspect.getfile(importlib.import_module("llamafactory"))).resolve().parent
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if revision != LLAMAFACTORY_REVISION:
            raise ValueError("LLaMA-Factory git revision mismatch")
        parser_module = importlib.import_module("llamafactory.data.parser")
        converter_module = importlib.import_module("llamafactory.data.converter")
        loader_module = importlib.import_module("llamafactory.data.loader")
        datasets_module = importlib.import_module("datasets")
        spec = official_profile_spec(args.profile)
        dataset_info = args.dataset_info or (
            DAY31.parent / "configs" / spec["dataset_info_name"]
        )
        train_path = args.data_dir / spec["file_names"][0]
        eval_path = args.data_dir / spec["file_names"][1]
        train = json.loads(train_path.read_text(encoding="utf-8"))
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            shutil.copy2(dataset_info, temp / "dataset_info.json")
            shutil.copy2(train_path, temp / spec["file_names"][0])
            shutil.copy2(eval_path, temp / spec["file_names"][1])
            projections = align_registered_datasets(
                temp,
                parser_module.get_dataset_list,
                converter_module.align_dataset,
                datasets_module.load_dataset,
                dataset_names=spec["registry_names"],
            )
        if [count for count, _ in projections] != spec["row_counts"]:
            raise ValueError("official aligned row counts mismatch")
        receipt: dict[str, Any] = {
            "command": sys.argv,
            "profile": args.profile,
            "version": LLAMAFACTORY_VERSION,
            "git_revision": revision,
            "exit_code": 0,
            "train_aligned_rows": spec["row_counts"][0],
            "eval_aligned_rows": spec["row_counts"][1],
            "aligned_projection_sha256": {"train": projections[0][1], "eval_frozen": projections[1][1]},
            "dataset_sha256": {
                "train": hashlib.sha256(canonical_json(train).encode()).hexdigest(),
                "eval_frozen": hashlib.sha256(canonical_json(evaluation).encode()).hexdigest(),
            },
            "dataset_file_sha256": {
                "train": sha256_file(train_path),
                "eval_frozen": sha256_file(eval_path),
                "dataset_info": sha256_file(dataset_info),
            },
            "runner_script_sha256": sha256_file(Path(__file__)),
            "official_module_hashes": {
                module.__name__: _module_identity(module)
                for module in (parser_module, converter_module, loader_module)
            },
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt).encode()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    except Exception as error:
        print(f"official parser smoke failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
