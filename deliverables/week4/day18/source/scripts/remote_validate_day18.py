#!/usr/bin/env python3
"""Run Day 18 tokenizer statistics and LLaMA-Factory schema smoke on AutoDL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def describe_lengths(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("lengths_must_not_be_empty")
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
    }


def validate_sharegpt_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        errors: set[str] = set()
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            errors.add("conversations_invalid")
        else:
            for turn, message in enumerate(conversations):
                expected = "human" if turn % 2 == 0 else "gpt"
                if (
                    not isinstance(message, dict)
                    or message.get("from") != expected
                    or not isinstance(message.get("value"), str)
                    or not message["value"].strip()
                ):
                    errors.add("conversation_role_order")
                    break
            if len(conversations) % 2 == 0:
                errors.add("conversation_role_order")

        pair: dict[str, dict[str, Any]] = {}
        for label in ("chosen", "rejected"):
            message = record.get(label)
            if (
                not isinstance(message, dict)
                or message.get("from") != "gpt"
                or not isinstance(message.get("value"), str)
                or not message["value"].strip()
            ):
                errors.add(f"{label}_invalid")
            else:
                pair[label] = message
        if pair.get("chosen", {}).get("value") == pair.get("rejected", {}).get("value"):
            errors.add("responses_not_distinct")

        if errors:
            failures.append(
                {
                    "id": record.get("id", f"index-{index}"),
                    "errors": sorted(errors),
                }
            )
    return failures


def _role_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    role_map = {"human": "user", "gpt": "assistant"}
    return [
        {"role": role_map[message["from"]], "content": message["value"]}
        for message in record["conversations"]
    ]


def collect_length_statistics(
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    cutoff_len: int,
) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    chosen_total_lengths: list[int] = []
    rejected_total_lengths: list[int] = []
    over_cutoff_ids: set[str] = set()

    for record in records:
        prompt = _role_messages(record)
        chosen = {"role": "assistant", "content": record["chosen"]["value"]}
        rejected = {"role": "assistant", "content": record["rejected"]["value"]}
        prompt_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )
        chosen_ids = tokenizer.encode(record["chosen"]["value"], add_special_tokens=False)
        rejected_ids = tokenizer.encode(record["rejected"]["value"], add_special_tokens=False)
        chosen_total = tokenizer.apply_chat_template(
            prompt + [chosen], tokenize=True, add_generation_prompt=False
        )
        rejected_total = tokenizer.apply_chat_template(
            prompt + [rejected], tokenize=True, add_generation_prompt=False
        )
        prompt_lengths.append(len(prompt_ids))
        chosen_lengths.append(len(chosen_ids))
        rejected_lengths.append(len(rejected_ids))
        chosen_total_lengths.append(len(chosen_total))
        rejected_total_lengths.append(len(rejected_total))
        if max(len(chosen_total), len(rejected_total)) > cutoff_len:
            over_cutoff_ids.add(record["id"])

    return {
        "status": "fail" if over_cutoff_ids else "pass",
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "cutoff_len": cutoff_len,
        "record_count": len(records),
        "prompt": describe_lengths(prompt_lengths),
        "chosen_response": describe_lengths(chosen_lengths),
        "rejected_response": describe_lengths(rejected_lengths),
        "chosen_total": describe_lengths(chosen_total_lengths),
        "rejected_total": describe_lengths(rejected_total_lengths),
        "records_over_cutoff": len(over_cutoff_ids),
        "records_over_cutoff_ids": sorted(over_cutoff_ids),
    }


def run_llamafactory_smoke(
    *,
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    dataset_name: str,
    cutoff_len: int,
) -> dict[str, Any]:
    import llamafactory
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    args = {
        "model_name_or_path": str(model_path),
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "dataset": dataset_name,
        "dataset_dir": str(data_dir),
        "template": "qwen",
        "cutoff_len": cutoff_len,
        "max_samples": 8,
        "preprocessing_num_workers": 1,
        "overwrite_cache": True,
        "output_dir": str(output_dir / "smoke_unused_output"),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 1.0,
        "learning_rate": 1e-5,
        "report_to": "none",
    }
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="rm",
        **tokenizer_module,
    )
    dataset = dataset_module["train_dataset"]
    if len(dataset) != 8:
        raise AssertionError(f"expected 8 smoke samples, got {len(dataset)}")
    required_columns = {
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_labels",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_labels",
    }
    missing = sorted(required_columns - set(dataset.column_names))
    if missing:
        raise AssertionError(f"missing tokenized columns: {missing}")
    first = dataset[0]
    if not first["chosen_input_ids"] or not first["rejected_input_ids"]:
        raise AssertionError("empty tokenized ranking pair")

    return {
        "status": "pass",
        "llamafactory_version": getattr(llamafactory, "__version__", "unknown"),
        "finetuning_stage": finetuning_args.stage,
        "loader_stage": "rm",
        "dataset_name": dataset_name,
        "requested_smoke_samples": 8,
        "loaded_smoke_samples": len(dataset),
        "columns": sorted(dataset.column_names),
        "first_chosen_tokens": len(first["chosen_input_ids"]),
        "first_rejected_tokens": len(first["rejected_input_ids"]),
        "template": data_args.template,
        "cutoff_len": data_args.cutoff_len,
        "model_path": str(model_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff-len", type=int, default=2048)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "week4_dpo_train.json"
    validation_path = args.data_dir / "week4_dpo_validation.json"
    records = json.loads(train_path.read_text(encoding="utf-8"))
    records += json.loads(validation_path.read_text(encoding="utf-8"))
    failures = validate_sharegpt_records(records)
    environment = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "instance_id": "78fb4ea487-e700080a",
        "model_path": str(args.model_path),
        "model_path_exists": args.model_path.is_dir(),
        "data_files": {
            train_path.name: {"count": len(json.loads(train_path.read_text(encoding="utf-8"))), "sha256": sha256(train_path)},
            validation_path.name: {"count": len(json.loads(validation_path.read_text(encoding="utf-8"))), "sha256": sha256(validation_path)},
            "dataset_info.json": {"sha256": sha256(args.data_dir / "dataset_info.json")},
        },
        "nvidia_smi": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        "structural_failure_count": len(failures),
        "structural_failures": failures,
    }
    write_json(args.output_dir / "remote_environment.json", environment)
    if failures or not args.model_path.is_dir():
        return 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    lengths = collect_length_statistics(
        tokenizer,
        records,
        cutoff_len=args.cutoff_len,
    )
    write_json(args.output_dir / "length_statistics.json", lengths)
    if lengths["status"] != "pass":
        return 1

    try:
        smoke = run_llamafactory_smoke(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            dataset_name="week4_dpo_train",
            cutoff_len=args.cutoff_len,
        )
    except Exception as error:
        smoke = {
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output_dir / "llamafactory_schema_smoke.json", smoke)
        return 1

    write_json(args.output_dir / "llamafactory_schema_smoke.json", smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
