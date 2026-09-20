#!/usr/bin/env python3
"""Validate Day 5 text artifacts without requiring CUDA or model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_RECORD_KEYS = {
    "id",
    "messages",
    "chat_template_text",
    "generation",
    "input_tokens",
    "output_tokens",
    "elapsed_seconds",
    "device",
    "dtype",
    "response",
    "score",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records, f"{path} contains no records"
    ids: set[str] = set()
    for record in records:
        missing = REQUIRED_RECORD_KEYS - record.keys()
        assert not missing, f"{path}: missing keys {sorted(missing)}"
        assert record["id"] not in ids, f"{path}: duplicate id {record['id']}"
        ids.add(record["id"])
        assert record["response"].strip(), f"{path}: empty response for {record['id']}"
        assert record["input_tokens"] > 0
        assert record["output_tokens"] > 0
        assert math.isfinite(float(record["elapsed_seconds"]))
    return records


def compare_runs(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> None:
    assert [r["id"] for r in before] == [r["id"] for r in after]
    after_by_id = {record["id"]: record for record in after}
    for old in before:
        new = after_by_id[old["id"]]
        assert old["messages"] == new["messages"], f"messages differ: {old['id']}"
        assert old["generation"] == new["generation"], f"generation differs: {old['id']}"
        if "chat_template_text" in old and "chat_template_text" in new:
            assert old["chat_template_text"] == new["chat_template_text"], (
                f"chat template differs: {old['id']}"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_run_metrics_in_log(log: str, metrics: dict[str, Any]) -> None:
    expected_fragments = (
        f"Num examples = {metrics['samples']}",
        f"Num Epochs = {int(metrics['configured_epochs'])}",
        f"Total optimization steps = {metrics['optimization_steps']}",
        f"'train_runtime': {metrics['train_runtime_seconds']}",
        f"'train_loss': {metrics['train_loss']}",
        f"'loss': {metrics['last_step_loss']}",
        "training_exit_code=0",
    )
    for fragment in expected_fragments:
        assert fragment in log, f"training log missing metric: {fragment}"


def validate(root: Path, expected_count: int = 8) -> dict[str, Any]:
    required = [
        "configs/qwen25_7b_identity_qlora.yaml",
        "configs/qwen25_7b_identity_qlora_run2.yaml",
        "source/prompts/identity_prompts.json",
        "source/results/day5_preflight.txt",
        "source/results/day5_train.txt",
        "source/results/day5_train_run2.txt",
        "source/results/day5_training_metrics.json",
        "source/results/day5_adapter_inventory.txt",
        "source/results/day5_adapter_inventory_run2.txt",
        "source/results/identity_before_training.jsonl",
        "source/results/identity_after_run1.jsonl",
        "source/results/identity_after_run2.jsonl",
        "source/results/identity_after_training.jsonl",
        "source/results/identity_evaluation.md",
        "source/results/day5_identity_comparison_summary.txt",
        "source/results/day5_tensorboard_page_text.txt",
        "evidence/training_complete.png",
        "evidence/tensorboard_loss.png",
    ]
    paths = {relative: root / relative for relative in required}
    for relative, path in paths.items():
        assert path.is_file() and path.stat().st_size > 0, f"missing: {relative}"

    before = read_jsonl(paths["source/results/identity_before_training.jsonl"])
    run1 = read_jsonl(paths["source/results/identity_after_run1.jsonl"])
    run2 = read_jsonl(paths["source/results/identity_after_run2.jsonl"])
    after = read_jsonl(paths["source/results/identity_after_training.jsonl"])
    assert len(before) == expected_count
    assert len(run1) == expected_count
    assert len(run2) == expected_count
    assert len(after) == expected_count
    compare_runs(before, run1)
    compare_runs(before, run2)
    compare_runs(before, after)
    assert all(record.get("adapter_loaded") is False for record in before)
    assert all(record.get("adapter_loaded") is True for record in run1)
    assert all(record.get("adapter_loaded") is True for record in run2)
    assert all(record.get("adapter_loaded") is True for record in after)
    assert paths["source/results/identity_after_run2.jsonl"].read_bytes() == paths[
        "source/results/identity_after_training.jsonl"
    ].read_bytes()

    training_logs: dict[str, str] = {}
    for run, relative in (
        ("run1", "source/results/day5_train.txt"),
        ("run2", "source/results/day5_train_run2.txt"),
    ):
        train_log = paths[relative].read_text(encoding="utf-8")
        training_logs[run] = train_log
        assert "training_exit_code=0" in train_log
        for forbidden in ("CUDA out of memory", "Traceback (most recent call last)"):
            assert forbidden not in train_log

    metrics = json.loads(
        paths["source/results/day5_training_metrics.json"].read_text(encoding="utf-8")
    )
    assert metrics["training_exit_code"] == 0
    assert metrics["epochs"] >= 3
    assert metrics["samples"] > 0
    assert math.isfinite(float(metrics["train_loss"]))
    assert metrics["adapter_config_present"] is True
    assert metrics["adapter_weights_present"] is True
    assert metrics["tensorboard_event_present"] is True
    assert metrics["selected_run"] == "run2"
    assert metrics["runs"]["run1"]["training_exit_code"] == 0
    assert metrics["runs"]["run2"]["training_exit_code"] == 0
    assert metrics["runs"]["run1"]["configured_epochs"] == 3.0
    assert metrics["runs"]["run2"]["configured_epochs"] == 5.0
    assert metrics["single_variable_change"]["field"] == "num_train_epochs"
    assert_run_metrics_in_log(training_logs["run1"], metrics["runs"]["run1"])
    assert_run_metrics_in_log(training_logs["run2"], metrics["runs"]["run2"])

    page_text = paths["source/results/day5_tensorboard_page_text.txt"].read_text(
        encoding="utf-8"
    )
    for expected in ("identity-qlora-run1", "identity-qlora-run2", "/loss"):
        assert expected in page_text

    before_passed = sum(r["score"]["passed"] for r in before)
    run1_passed = sum(r["score"]["passed"] for r in run1)
    run2_passed = sum(r["score"]["passed"] for r in run2)
    assert (before_passed, run1_passed, run2_passed) == (0, 6, 7)

    return {
        "status": "PASS",
        "identity_records_before": len(before),
        "identity_records_each_run": expected_count,
        "before_passed": before_passed,
        "run1_passed": run1_passed,
        "run2_passed": run2_passed,
        "selected_run": metrics["selected_run"],
        "training_exit_code": metrics["training_exit_code"],
        "train_loss": metrics["train_loss"],
        "validated_files": len(paths),
        "file_sha256": {relative: sha256(path) for relative, path in paths.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = validate(args.root, args.expected_count)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
