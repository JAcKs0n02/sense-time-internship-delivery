#!/usr/bin/env python3
"""Validate and summarize the nine Week 3 training attempts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSpec:
    run_id: str
    group: str
    lora_rank: int
    learning_rate: float
    num_train_epochs: float


@dataclass(frozen=True)
class LossSummary:
    status: str
    final_loss: float | None
    last_100_loss_mean: float | None
    loss_count: int


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    attempt_id: str | None
    group: str
    lora_rank: int
    learning_rate: float
    num_train_epochs: float
    status: str
    final_loss: float | None
    last_100_loss_mean: float | None
    train_runtime_seconds: float | None
    train_runtime_hhmmss: str | None
    wall_clock_runtime_seconds: float | None
    wall_clock_runtime_hhmmss: str | None
    sampled_peak_gpu_memory_mib: int | None
    optimizer_steps: int | None
    expected_optimizer_steps: int
    adapter_size_bytes: int | None
    adapter_path: str | None
    source_config_sha256: str | None
    runtime_config_sha256: str | None
    exit_code: int | None


CSV_FIELDS = tuple(RunSummary.__dataclass_fields__)
BASELINE_RUN_IDS = ("rank-r8", "lr-1e-4", "epoch-e3")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seconds_hhmmss(value: float | None) -> str | None:
    if value is None or not math.isfinite(value) or value < 0:
        return None
    total = int(round(value))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def summarize_log_history(rows: list[dict[str, Any]]) -> LossSummary:
    losses: list[float] = []
    for row in rows:
        if "loss" not in row:
            continue
        try:
            loss = float(row["loss"])
        except (TypeError, ValueError):
            return LossSummary("invalid_nonfinite_loss", None, None, len(losses) + 1)
        if not math.isfinite(loss):
            return LossSummary("invalid_nonfinite_loss", None, None, len(losses) + 1)
        losses.append(loss)
    if not losses:
        return LossSummary("invalid_missing_loss", None, None, 0)
    window = losses[-100:]
    return LossSummary(
        "valid",
        losses[-1],
        sum(window) / len(window),
        len(losses),
    )


def expected_optimizer_steps(
    spec: ExperimentSpec,
    training_records: int = 4999,
    effective_batch: int = 4,
) -> int:
    epochs = float(spec.num_train_epochs)
    if not epochs.is_integer():
        raise ValueError(f"Week 3 epochs must be integral, got {epochs}")
    return (training_records // effective_batch) * int(epochs)


def sampled_peak_memory(path: Path) -> int | None:
    if not path.is_file():
        return None
    values: list[int] = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) < 2:
                continue
            raw = row[1].strip().removesuffix(" MiB").strip()
            try:
                values.append(int(float(raw)))
            except ValueError:
                continue
    return max(values) if values else None


def verify_checksum_manifest(manifest_path: Path, base_dir: Path) -> bool:
    if not manifest_path.is_file():
        return False
    lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return False
    for line in lines:
        try:
            expected, relative = line.split(None, 1)
        except ValueError:
            return False
        candidate = (base_dir / relative.strip()).resolve()
        if not candidate.is_relative_to(base_dir.resolve()) or not candidate.is_file():
            return False
        if sha256_file(candidate) != expected:
            return False
    return True


def empty_summary(
    run_id: str,
    attempt_id: str | None,
    spec: ExperimentSpec,
    status: str,
    *,
    exit_code: int | None = None,
    wall_clock: float | None = None,
    source_hash: str | None = None,
    runtime_hash: str | None = None,
) -> RunSummary:
    expected = expected_optimizer_steps(spec)
    return RunSummary(
        run_id=run_id,
        attempt_id=attempt_id,
        group=spec.group,
        lora_rank=spec.lora_rank,
        learning_rate=spec.learning_rate,
        num_train_epochs=spec.num_train_epochs,
        status=status,
        final_loss=None,
        last_100_loss_mean=None,
        train_runtime_seconds=None,
        train_runtime_hhmmss=None,
        wall_clock_runtime_seconds=wall_clock,
        wall_clock_runtime_hhmmss=seconds_hhmmss(wall_clock),
        sampled_peak_gpu_memory_mib=None,
        optimizer_steps=None,
        expected_optimizer_steps=expected,
        adapter_size_bytes=None,
        adapter_path=None,
        source_config_sha256=source_hash,
        runtime_config_sha256=runtime_hash,
        exit_code=exit_code,
    )


def summarize_run(run_id: str, attempt_dir: Path, spec: ExperimentSpec) -> RunSummary:
    expected = expected_optimizer_steps(spec)
    status_path = attempt_dir / "status.json"
    if not status_path.is_file():
        return empty_summary(run_id, attempt_dir.name, spec, "invalid_missing_status")
    try:
        status_payload = load_json(status_path)
    except Exception:
        return empty_summary(run_id, attempt_dir.name, spec, "invalid_missing_status")

    exit_code = status_payload.get("exit_code")
    wall_clock_raw = status_payload.get("wall_clock_runtime_seconds")
    wall_clock = float(wall_clock_raw) if wall_clock_raw is not None else None
    source_path = attempt_dir / "source_config.yaml"
    runtime_path = attempt_dir / "runtime_config.yaml"
    source_hash = sha256_file(source_path) if source_path.is_file() else None
    runtime_hash = sha256_file(runtime_path) if runtime_path.is_file() else None
    preliminary = str(status_payload.get("status", "invalid_missing_status"))

    if (
        source_hash is None
        or runtime_hash is None
        or status_payload.get("source_config_sha256") != source_hash
        or status_payload.get("runtime_config_sha256") != runtime_hash
    ):
        return empty_summary(
            run_id,
            attempt_dir.name,
            spec,
            "invalid_config_hash",
            exit_code=exit_code,
            wall_clock=wall_clock,
            source_hash=source_hash,
            runtime_hash=runtime_hash,
        )
    if preliminary == "completed" and status_payload.get("failure_reason") not in (None, ""):
        return empty_summary(
            run_id,
            attempt_dir.name,
            spec,
            "invalid_failure_reason",
            exit_code=exit_code,
            wall_clock=wall_clock,
            source_hash=source_hash,
            runtime_hash=runtime_hash,
        )

    if exit_code != 0 or preliminary != "completed":
        reported = "failed" if exit_code not in (None, 0) else preliminary
        return empty_summary(
            run_id,
            attempt_dir.name,
            spec,
            reported,
            exit_code=exit_code,
            wall_clock=wall_clock,
            source_hash=source_hash,
            runtime_hash=runtime_hash,
        )

    adapter = attempt_dir / "adapter"
    trainer_state_path = adapter / "trainer_state.json"
    train_results_path = adapter / "train_results.json"
    try:
        trainer_state = load_json(trainer_state_path)
        train_results = load_json(train_results_path)
    except Exception:
        return empty_summary(
            run_id,
            attempt_dir.name,
            spec,
            "invalid_missing_evidence",
            exit_code=exit_code,
            wall_clock=wall_clock,
            source_hash=source_hash,
            runtime_hash=runtime_hash,
        )

    loss_summary = summarize_log_history(trainer_state.get("log_history", []))
    if loss_summary.status != "valid":
        invalid = empty_summary(
            run_id,
            attempt_dir.name,
            spec,
            loss_summary.status,
            exit_code=exit_code,
            wall_clock=wall_clock,
            source_hash=source_hash,
            runtime_hash=runtime_hash,
        )
        return invalid

    global_step = int(trainer_state.get("global_step", -1))
    max_steps = int(trainer_state.get("max_steps", -1))
    if global_step != expected or max_steps != expected:
        return empty_summary(
            run_id,
            attempt_dir.name,
            spec,
            "invalid_incomplete_steps",
            exit_code=exit_code,
            wall_clock=wall_clock,
            source_hash=source_hash,
            runtime_hash=runtime_hash,
        )

    adapter_config = adapter / "adapter_config.json"
    weights = [
        path
        for pattern in ("adapter_model.safetensors", "adapter_model.bin")
        for path in adapter.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]
    if not adapter_config.is_file() or adapter_config.stat().st_size == 0 or not weights:
        return empty_summary(
            run_id,
            attempt_dir.name,
            spec,
            "invalid_adapter",
            exit_code=exit_code,
            wall_clock=wall_clock,
            source_hash=source_hash,
            runtime_hash=runtime_hash,
        )
    if not verify_checksum_manifest(attempt_dir / "adapter_files.sha256", adapter):
        return empty_summary(
            run_id,
            attempt_dir.name,
            spec,
            "invalid_adapter_hash",
            exit_code=exit_code,
            wall_clock=wall_clock,
            source_hash=source_hash,
            runtime_hash=runtime_hash,
        )

    train_runtime_raw = train_results.get("train_runtime")
    train_runtime = float(train_runtime_raw) if train_runtime_raw is not None else None
    if train_runtime is None or not math.isfinite(train_runtime) or train_runtime < 0:
        return empty_summary(
            run_id,
            attempt_dir.name,
            spec,
            "invalid_runtime",
            exit_code=exit_code,
            wall_clock=wall_clock,
            source_hash=source_hash,
            runtime_hash=runtime_hash,
        )

    return RunSummary(
        run_id=run_id,
        attempt_id=attempt_dir.name,
        group=spec.group,
        lora_rank=spec.lora_rank,
        learning_rate=spec.learning_rate,
        num_train_epochs=spec.num_train_epochs,
        status="completed",
        final_loss=loss_summary.final_loss,
        last_100_loss_mean=loss_summary.last_100_loss_mean,
        train_runtime_seconds=train_runtime,
        train_runtime_hhmmss=seconds_hhmmss(train_runtime),
        wall_clock_runtime_seconds=wall_clock,
        wall_clock_runtime_hhmmss=seconds_hhmmss(wall_clock),
        sampled_peak_gpu_memory_mib=sampled_peak_memory(attempt_dir / "gpu_memory.csv"),
        optimizer_steps=global_step,
        expected_optimizer_steps=expected,
        adapter_size_bytes=sum(path.stat().st_size for path in weights),
        adapter_path=str(adapter),
        source_config_sha256=source_hash,
        runtime_config_sha256=runtime_hash,
        exit_code=exit_code,
    )


def load_specs(matrix_path: Path) -> list[ExperimentSpec]:
    payload = load_json(matrix_path)
    return [
        ExperimentSpec(
            run_id=row["run_id"],
            group=row["group"],
            lora_rank=int(row["lora_rank"]),
            learning_rate=float(row["learning_rate"]),
            num_train_epochs=float(row["num_train_epochs"]),
        )
        for row in payload["runs"]
    ]


def latest_attempt(run_dir: Path) -> Path | None:
    attempts = sorted(path for path in run_dir.glob("attempt-[0-9][0-9][0-9]") if path.is_dir())
    return attempts[-1] if attempts else None


def summarize_matrix(matrix_path: Path, run_root: Path) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for spec in load_specs(matrix_path):
        attempt = latest_attempt(run_root / spec.run_id)
        if attempt is None:
            summaries.append(empty_summary(spec.run_id, None, spec, "not_started"))
        else:
            summaries.append(summarize_run(spec.run_id, attempt, spec))
    return summaries


def variability_payload(summaries: list[RunSummary]) -> dict[str, Any]:
    rows = [row for row in summaries if row.run_id in BASELINE_RUN_IDS and row.status == "completed"]
    result: dict[str, Any] = {
        "run_ids": list(BASELINE_RUN_IDS),
        "successful_run_ids": [row.run_id for row in rows],
        "complete": len(rows) == len(BASELINE_RUN_IDS),
        "metrics": {},
    }
    for field in (
        "final_loss",
        "train_runtime_seconds",
        "wall_clock_runtime_seconds",
        "sampled_peak_gpu_memory_mib",
    ):
        raw = {row.run_id: getattr(row, field) for row in rows}
        values = [float(value) for value in raw.values() if value is not None]
        result["metrics"][field] = {
            "raw": raw,
            "mean": statistics.mean(values) if values else None,
            "standard_deviation": statistics.stdev(values) if len(values) >= 2 else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }
    return result


def write_outputs(
    summaries: list[RunSummary],
    csv_path: Path,
    json_path: Path,
    baseline_path: Path,
    adapters_path: Path,
) -> None:
    rows = [asdict(summary) for summary in summaries]
    for path in (csv_path, json_path, baseline_path, adapters_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    baseline_path.write_text(
        json.dumps(variability_payload(summaries), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    adapters = [
        {
            "run_id": row.run_id,
            "attempt_id": row.attempt_id,
            "adapter_path": row.adapter_path,
            "status": row.status,
            "adapter_files_sha256": str(
                Path(row.adapter_path).parent / "adapter_files.sha256"
            ),
        }
        for row in summaries
        if row.status == "completed" and row.adapter_path is not None
    ]
    adapters_path.write_text(
        json.dumps(adapters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Week 3 training attempts.")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--adapters-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summaries = summarize_matrix(args.matrix, args.run_root)
        write_outputs(summaries, args.csv, args.json, args.baseline_output, args.adapters_output)
    except Exception as error:
        print(f"summary failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps([asdict(row) for row in summaries], ensure_ascii=False, indent=2))
    return 0 if all(row.status == "completed" for row in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
