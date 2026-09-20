#!/usr/bin/env python3
"""Run the frozen Week 3 QLoRA matrix sequentially with auditable state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml


MIN_FREE_GPU_MIB = 20_000
MIN_FREE_DISK_GIB = 15.0
ATTEMPT_PATTERN = re.compile(r"attempt-(\d{3})$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_frozen_inputs(
    matrix_path: Path,
    base_config_path: Path,
    config_dir: Path,
    validation_path: Path,
) -> list[str]:
    errors: list[str] = []
    try:
        validation = load_json(validation_path)
        matrix = load_json(matrix_path)
    except Exception as error:
        return [str(error)]

    if validation.get("valid") is not True:
        errors.append("Day 11 validation is not valid")
    if validation.get("matrix_sha256") != sha256_file(matrix_path):
        errors.append("matrix hash mismatch")
    if validation.get("base_config_sha256") != sha256_file(base_config_path):
        errors.append("base config hash mismatch")

    runs = matrix.get("runs")
    if not isinstance(runs, list):
        errors.append("matrix runs must be a list")
        return errors
    expected_hashes = validation.get("config_sha256")
    if not isinstance(expected_hashes, dict):
        errors.append("Day 11 validation has no config_sha256 mapping")
        return errors

    expected_ids = []
    for row in runs:
        run_id = row.get("run_id") if isinstance(row, dict) else None
        if not isinstance(run_id, str):
            errors.append("matrix contains an invalid run_id")
            continue
        expected_ids.append(run_id)
        config_path = config_dir / f"{run_id}.yaml"
        if not config_path.is_file():
            errors.append(f"config missing: {run_id}")
        elif expected_hashes.get(run_id) != sha256_file(config_path):
            errors.append(f"config hash mismatch: {run_id}")

    unexpected = sorted(
        path.stem for path in config_dir.glob("*.yaml") if path.stem not in expected_ids
    )
    if unexpected:
        errors.append(f"unexpected configs: {unexpected}")
    if set(expected_hashes) != set(expected_ids):
        errors.append("validation config IDs do not match matrix run IDs")
    return errors


def build_runtime_config(
    source_config: dict[str, Any],
    run_id: str,
    attempt_id: str,
    attempt_dir: Path,
    resume_checkpoint: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    runtime = dict(source_config)
    runtime["output_dir"] = str(attempt_dir / "adapter")
    source_run_name = str(source_config.get("run_name", run_id))
    runtime["run_name"] = f"{source_run_name}-{attempt_id}"

    if resume_checkpoint is not None:
        resolved_checkpoint = resume_checkpoint.resolve()
        run_dir = attempt_dir.parent.resolve()
        if not resolved_checkpoint.is_dir():
            raise ValueError(f"resume checkpoint does not exist: {resume_checkpoint}")
        if not resolved_checkpoint.is_relative_to(run_dir):
            raise ValueError("resume checkpoint must belong to the same run")
        runtime["resume_from_checkpoint"] = str(resolved_checkpoint)

    differences = {
        key: {"source": source_config.get(key), "runtime": runtime.get(key)}
        for key in sorted(set(source_config) | set(runtime))
        if source_config.get(key) != runtime.get(key)
    }
    allowed = {"output_dir", "run_name"}
    if resume_checkpoint is not None:
        allowed.add("resume_from_checkpoint")
    forbidden = set(differences) - allowed
    if forbidden:
        raise ValueError(f"runtime config changes forbidden fields: {sorted(forbidden)}")
    return runtime, differences


def attempt_directories(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    return sorted(
        (path for path in run_dir.iterdir() if path.is_dir() and ATTEMPT_PATTERN.fullmatch(path.name)),
        key=lambda path: int(ATTEMPT_PATTERN.fullmatch(path.name).group(1)),  # type: ignore[union-attr]
    )


def next_attempt_dir(run_dir: Path) -> Path:
    attempts = attempt_directories(run_dir)
    number = 1 if not attempts else int(attempts[-1].name.removeprefix("attempt-")) + 1
    return run_dir / f"attempt-{number:03d}"


def default_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def latest_status(run_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    attempts = attempt_directories(run_dir)
    if not attempts:
        return None, None
    attempt = attempts[-1]
    status_path = attempt / "status.json"
    if not status_path.is_file():
        return attempt, {"status": "interrupted", "failure_reason": "missing status.json"}
    try:
        return attempt, load_json(status_path)
    except Exception as error:
        return attempt, {"status": "interrupted", "failure_reason": str(error)}


def decide_action(
    run_dir: Path,
    allow_retry: bool,
    process_alive: Callable[[int], bool] = default_process_alive,
) -> str:
    _attempt, status = latest_status(run_dir)
    if status is None:
        return "start_new"
    state = status.get("status")
    if state == "completed":
        return "skip_completed"
    if state == "running":
        pid = status.get("pid")
        if isinstance(pid, int) and process_alive(pid):
            return "skip_running"
        return "mark_interrupted"
    if allow_retry:
        return "start_retry"
    return "skip_failed"


def preflight_errors(
    gpu_process_count: int,
    free_gpu_mib: int,
    free_disk_gib: float,
) -> list[str]:
    errors: list[str] = []
    if gpu_process_count:
        errors.append("gpu busy")
    if free_gpu_mib < MIN_FREE_GPU_MIB:
        errors.append(f"free GPU memory below {MIN_FREE_GPU_MIB} MiB")
    if free_disk_gib < MIN_FREE_DISK_GIB:
        errors.append(f"free disk below {MIN_FREE_DISK_GIB:g} GiB")
    return errors


def read_matrix_status(matrix_path: Path, run_root: Path) -> list[dict[str, Any]]:
    matrix = load_json(matrix_path)
    rows: list[dict[str, Any]] = []
    for spec in matrix.get("runs", []):
        run_id = spec["run_id"]
        attempt, status = latest_status(run_root / run_id)
        rows.append(
            {
                "run_id": run_id,
                "attempt_id": attempt.name if attempt else None,
                "status": status.get("status", "interrupted") if status else "not_started",
            }
        )
    return rows


def run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        return subprocess.CompletedProcess(command, 127, stdout="", stderr=str(error))


def query_preflight(runtime: dict[str, Any], disk_path: Path) -> tuple[list[str], dict[str, Any]]:
    gpu = run_capture(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    processes = run_capture(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    full_smi = run_capture(["nvidia-smi"])
    if gpu.returncode != 0 or processes.returncode != 0:
        return ["nvidia-smi query failed"], {
            "gpu_query": gpu.stderr,
            "process_query": processes.stderr,
            "nvidia_smi": full_smi.stdout + full_smi.stderr,
        }

    free_gpu_values = [int(line.strip()) for line in gpu.stdout.splitlines() if line.strip().isdigit()]
    process_ids = [line.strip() for line in processes.stdout.splitlines() if line.strip().isdigit()]
    disk = shutil.disk_usage(disk_path)
    free_disk_gib = disk.free / (1024**3)
    free_gpu_mib = free_gpu_values[0] if free_gpu_values else 0
    errors = preflight_errors(len(process_ids), free_gpu_mib, free_disk_gib)

    model_path = Path(str(runtime.get("model_name_or_path", "")))
    dataset_dir = Path(str(runtime.get("dataset_dir", "")))
    if not model_path.is_dir():
        errors.append(f"model directory missing: {model_path}")
    if not dataset_dir.is_dir():
        errors.append(f"dataset directory missing: {dataset_dir}")
    elif not (dataset_dir / "dataset_info.json").is_file():
        errors.append(f"dataset_info.json missing: {dataset_dir}")

    details = {
        "checked_at": utc_now(),
        "free_gpu_memory_mib": free_gpu_mib,
        "gpu_compute_pids": process_ids,
        "free_disk_gib": free_disk_gib,
        "disk_path": str(disk_path),
        "nvidia_smi": full_smi.stdout + full_smi.stderr,
        "errors": errors,
    }
    return errors, details


def environment_snapshot() -> str:
    commands = [
        [sys.executable, "--version"],
        ["llamafactory-cli", "version"],
        ["nvidia-smi"],
        ["df", "-h", "/root/autodl-tmp"],
    ]
    chunks: list[str] = []
    for command in commands:
        result = run_capture(command)
        chunks.append(f"$ {' '.join(command)}\n{result.stdout}{result.stderr}")
    return "\n".join(chunks)


def write_adapter_manifest(adapter_dir: Path, output: Path) -> None:
    lines: list[str] = []
    if adapter_dir.is_dir():
        for path in sorted(adapter_dir.iterdir()):
            if path.is_file():
                lines.append(f"{sha256_file(path)}  {path.name}")
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_attempt(
    run_id: str,
    source_config_path: Path,
    attempt_dir: Path,
    disk_path: Path,
    resume_checkpoint: Path | None = None,
    timeout_seconds: float = 12 * 3600,
) -> dict[str, Any]:
    if attempt_dir.exists() and any(attempt_dir.iterdir()):
        raise ValueError(f"refusing non-empty attempt directory: {attempt_dir}")
    attempt_dir.mkdir(parents=True, exist_ok=True)

    source_copy = attempt_dir / "source_config.yaml"
    shutil.copy2(source_config_path, source_copy)
    source = load_yaml(source_copy)
    runtime, differences = build_runtime_config(
        source,
        run_id=run_id,
        attempt_id=attempt_dir.name,
        attempt_dir=attempt_dir,
        resume_checkpoint=resume_checkpoint,
    )
    runtime_path = attempt_dir / "runtime_config.yaml"
    runtime_path.write_text(
        yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    write_json_atomic(attempt_dir / "config_diff.json", differences)
    (attempt_dir / "environment.txt").write_text(environment_snapshot(), encoding="utf-8")

    errors, preflight = query_preflight(runtime, disk_path)
    write_json_atomic(attempt_dir / "preflight.json", preflight)
    (attempt_dir / "preflight.txt").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    base_status = {
        "run_id": run_id,
        "attempt_id": attempt_dir.name,
        "source_config_path": str(source_copy),
        "source_config_sha256": sha256_file(source_copy),
        "runtime_config_path": str(runtime_path),
        "runtime_config_sha256": sha256_file(runtime_path),
        "started_at": utc_now(),
    }
    if errors:
        status = {
            **base_status,
            "status": "blocked_preflight",
            "exit_code": None,
            "failure_reason": "; ".join(errors),
            "ended_at": utc_now(),
            "wall_clock_runtime_seconds": 0.0,
        }
        write_json_atomic(attempt_dir / "status.json", status)
        return status

    gpu_log = (attempt_dir / "gpu_memory.csv").open("w", encoding="utf-8")
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-l",
            "1",
        ],
        stdout=gpu_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    command = ["llamafactory-cli", "train", str(runtime_path)]
    start = time.monotonic()
    training: subprocess.Popen[str] | None = None
    exit_code: int | None = None
    failure_reason: str | None = None
    try:
        with (attempt_dir / "train.log").open("w", encoding="utf-8", buffering=1) as train_log:
            training = subprocess.Popen(
                command,
                stdout=train_log,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            write_json_atomic(
                attempt_dir / "status.json",
                {
                    **base_status,
                    "status": "running",
                    "pid": training.pid,
                    "sampler_pid": sampler.pid,
                    "command": command,
                },
            )
            exit_code = training.wait(timeout=timeout_seconds)
            if exit_code != 0:
                failure_reason = f"training exited with code {exit_code}"
    except subprocess.TimeoutExpired:
        failure_reason = f"hard timeout after {timeout_seconds:g} seconds"
        if training is not None and training.poll() is None:
            training.send_signal(signal.SIGTERM)
            try:
                exit_code = training.wait(timeout=30)
            except subprocess.TimeoutExpired:
                training.kill()
                exit_code = training.wait(timeout=10)
    except KeyboardInterrupt as error:
        failure_reason = f"{type(error).__name__}: {error}"
        if training is not None and training.poll() is None:
            training.send_signal(signal.SIGTERM)
            try:
                exit_code = training.wait(timeout=30)
            except subprocess.TimeoutExpired:
                training.kill()
                exit_code = training.wait(timeout=10)
        raise
    except Exception as error:
        failure_reason = f"{type(error).__name__}: {error}"
        if training is not None and training.poll() is None:
            training.send_signal(signal.SIGTERM)
            try:
                exit_code = training.wait(timeout=30)
            except subprocess.TimeoutExpired:
                training.kill()
                exit_code = training.wait(timeout=10)
    finally:
        duration = time.monotonic() - start
        terminate_process(sampler)
        gpu_log.close()
        adapter_dir = Path(runtime["output_dir"])
        write_adapter_manifest(adapter_dir, attempt_dir / "adapter_files.sha256")
        status = {
            **base_status,
            "status": "completed" if exit_code == 0 and failure_reason is None else "failed",
            "pid": training.pid if training is not None else None,
            "sampler_pid": sampler.pid,
            "command": command,
            "exit_code": exit_code,
            "failure_reason": failure_reason,
            "ended_at": utc_now(),
            "wall_clock_runtime_seconds": duration,
            "adapter_path": str(adapter_dir),
        }
        write_json_atomic(attempt_dir / "status.json", status)
    return status


def mark_interrupted(attempt_dir: Path) -> None:
    status_path = attempt_dir / "status.json"
    status = load_json(status_path) if status_path.is_file() else {}
    status.update(
        {
            "status": "interrupted",
            "ended_at": utc_now(),
            "failure_reason": "recorded process is no longer alive",
        }
    )
    write_json_atomic(status_path, status)


def parse_resume_values(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--resume-checkpoint must use RUN_ID=PATH")
        run_id, path = value.split("=", 1)
        parsed[run_id] = Path(path)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Week 3 experiment matrix sequentially.")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--disk-path", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--retry-run", action="append", default=[])
    parser.add_argument("--resume-checkpoint", action="append", default=[])
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status_only:
        print(json.dumps(read_matrix_status(args.matrix, args.run_root), ensure_ascii=False, indent=2))
        return 0
    if args.validation is None or args.base_config is None or args.config_dir is None:
        print("--validation, --base-config and --config-dir are required for training", file=sys.stderr)
        return 2

    errors = verify_frozen_inputs(args.matrix, args.base_config, args.config_dir, args.validation)
    if errors:
        print(json.dumps({"status": "blocked_frozen_input", "errors": errors}, indent=2), file=sys.stderr)
        return 1

    matrix = load_json(args.matrix)
    retries = set(args.retry_run)
    resumes = parse_resume_values(args.resume_checkpoint)
    known_ids = {row["run_id"] for row in matrix["runs"]}
    unknown = (retries | set(resumes)) - known_ids
    if unknown:
        print(f"unknown retry/resume run IDs: {sorted(unknown)}", file=sys.stderr)
        return 2

    args.run_root.mkdir(parents=True, exist_ok=True)
    for row in matrix["runs"]:
        run_id = row["run_id"]
        run_dir = args.run_root / run_id
        action = decide_action(run_dir, allow_retry=run_id in retries)
        if action == "mark_interrupted":
            attempt, _status = latest_status(run_dir)
            if attempt is not None:
                mark_interrupted(attempt)
            action = "start_retry" if run_id in retries else "skip_failed"
        if action in {"skip_completed", "skip_running", "skip_failed"}:
            print(json.dumps({"run_id": run_id, "action": action}))
            continue

        attempt_dir = next_attempt_dir(run_dir)
        print(json.dumps({"run_id": run_id, "action": action, "attempt": attempt_dir.name}))
        status = run_attempt(
            run_id,
            args.config_dir / f"{run_id}.yaml",
            attempt_dir,
            args.disk_path,
            resume_checkpoint=resumes.get(run_id),
            timeout_seconds=args.timeout_hours * 3600,
        )
        print(json.dumps(status, ensure_ascii=False))
        if status["status"] == "blocked_preflight":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
