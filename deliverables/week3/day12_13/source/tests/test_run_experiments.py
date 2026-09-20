import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


DAY12_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = DAY12_ROOT / "source" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_experiments  # noqa: E402
from run_experiments import (  # noqa: E402
    build_runtime_config,
    decide_action,
    next_attempt_dir,
    preflight_errors,
    read_matrix_status,
    run_capture,
    run_attempt,
    verify_frozen_inputs,
    write_json_atomic,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen_bundle(root: Path) -> tuple[Path, Path, Path, Path]:
    matrix = root / "experiment_matrix.json"
    matrix.write_text(
        json.dumps({"runs": [{"run_id": "rank-r8"}]}) + "\n",
        encoding="utf-8",
    )
    base = root / "base.yaml"
    base.write_text("seed: 42\noutput_dir: /frozen/base\nrun_name: base\n", encoding="utf-8")
    config_dir = root / "configs"
    config_dir.mkdir()
    config = config_dir / "rank-r8.yaml"
    config.write_text(
        "seed: 42\noutput_dir: /frozen/rank-r8/adapter\nrun_name: rank-r8\n",
        encoding="utf-8",
    )
    validation = root / "day11_validation.json"
    validation.write_text(
        json.dumps(
            {
                "valid": True,
                "matrix_sha256": sha256(matrix),
                "base_config_sha256": sha256(base),
                "config_sha256": {"rank-r8": sha256(config)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return matrix, base, config_dir, validation


def write_status(attempt_dir: Path, status: str, *, pid: int | None = None) -> None:
    attempt_dir.mkdir(parents=True)
    payload = {"run_id": attempt_dir.parent.name, "attempt_id": attempt_dir.name, "status": status}
    if pid is not None:
        payload["pid"] = pid
    (attempt_dir / "status.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_verify_frozen_inputs_accepts_exact_day11_hashes(tmp_path: Path) -> None:
    matrix, base, config_dir, validation = write_frozen_bundle(tmp_path)

    assert verify_frozen_inputs(matrix, base, config_dir, validation) == []


def test_verify_frozen_inputs_blocks_modified_config(tmp_path: Path) -> None:
    matrix, base, config_dir, validation = write_frozen_bundle(tmp_path)
    (config_dir / "rank-r8.yaml").write_text("seed: 7\n", encoding="utf-8")

    assert verify_frozen_inputs(matrix, base, config_dir, validation) == [
        "config hash mismatch: rank-r8"
    ]


def test_build_runtime_config_changes_only_attempt_metadata(tmp_path: Path) -> None:
    source = {
        "model_name_or_path": "/models/qwen",
        "dataset": "week2_clean_alpaca",
        "seed": 42,
        "lora_rank": 8,
        "learning_rate": 1e-4,
        "num_train_epochs": 3.0,
        "output_dir": "/frozen/rank-r8/adapter",
        "run_name": "qwen25-7b-week3-rank-r8",
        "resume_from_checkpoint": None,
    }
    attempt = tmp_path / "rank-r8" / "attempt-001"

    runtime, differences = build_runtime_config(
        source,
        run_id="rank-r8",
        attempt_id="attempt-001",
        attempt_dir=attempt,
    )

    assert set(differences) == {"output_dir", "run_name"}
    assert runtime["output_dir"] == str(attempt / "adapter")
    assert runtime["run_name"] == "qwen25-7b-week3-rank-r8-attempt-001"
    for key in set(source) - {"output_dir", "run_name"}:
        assert runtime[key] == source[key]


def test_build_runtime_config_only_allows_explicit_same_run_resume(tmp_path: Path) -> None:
    source = {
        "output_dir": "/frozen/rank-r8/adapter",
        "run_name": "rank-r8",
        "resume_from_checkpoint": None,
    }
    run_dir = tmp_path / "rank-r8"
    checkpoint = run_dir / "attempt-001" / "adapter" / "checkpoint-500"
    checkpoint.mkdir(parents=True)

    runtime, differences = build_runtime_config(
        source,
        run_id="rank-r8",
        attempt_id="attempt-002",
        attempt_dir=run_dir / "attempt-002",
        resume_checkpoint=checkpoint,
    )

    assert set(differences) == {"output_dir", "run_name", "resume_from_checkpoint"}
    assert runtime["resume_from_checkpoint"] == str(checkpoint)


def test_build_runtime_config_rejects_checkpoint_from_another_run(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rank-r32" / "attempt-001" / "adapter" / "checkpoint-500"
    checkpoint.mkdir(parents=True)

    try:
        build_runtime_config(
            {"output_dir": "/frozen", "run_name": "rank-r8", "resume_from_checkpoint": None},
            run_id="rank-r8",
            attempt_id="attempt-002",
            attempt_dir=tmp_path / "rank-r8" / "attempt-002",
            resume_checkpoint=checkpoint,
        )
    except ValueError as error:
        assert "same run" in str(error)
    else:
        raise AssertionError("cross-run checkpoint was accepted")


def test_next_attempt_directory_is_zero_padded(tmp_path: Path) -> None:
    run_dir = tmp_path / "rank-r8"
    write_status(run_dir / "attempt-001", "failed")

    assert next_attempt_dir(run_dir) == run_dir / "attempt-002"


def test_completed_run_is_never_overwritten(tmp_path: Path) -> None:
    run_dir = tmp_path / "rank-r8"
    write_status(run_dir / "attempt-001", "completed")

    assert decide_action(run_dir, allow_retry=False) == "skip_completed"


def test_failed_run_is_not_automatically_retried(tmp_path: Path) -> None:
    run_dir = tmp_path / "rank-r8"
    write_status(run_dir / "attempt-001", "failed")

    assert decide_action(run_dir, allow_retry=False) == "skip_failed"
    assert decide_action(run_dir, allow_retry=True) == "start_retry"


def test_stale_running_attempt_is_marked_interrupted(tmp_path: Path) -> None:
    run_dir = tmp_path / "rank-r8"
    write_status(run_dir / "attempt-001", "running", pid=99999999)

    assert decide_action(run_dir, allow_retry=False, process_alive=lambda _pid: False) == "mark_interrupted"


def test_preflight_resource_gates() -> None:
    assert preflight_errors(0, 24000, 50.0) == []
    assert preflight_errors(1, 24000, 50.0) == ["gpu busy"]
    assert preflight_errors(0, 19999, 14.9) == [
        "free GPU memory below 20000 MiB",
        "free disk below 15 GiB",
    ]


def test_atomic_status_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    output = tmp_path / "status.json"

    write_json_atomic(output, {"status": "completed"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "completed"}
    assert not list(tmp_path.glob("*.tmp"))


def test_status_report_includes_all_manifest_runs(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps({"runs": [{"run_id": "rank-r8"}, {"run_id": "rank-r32"}]}) + "\n",
        encoding="utf-8",
    )
    write_status(tmp_path / "runs" / "rank-r8" / "attempt-001", "completed")

    assert read_matrix_status(matrix, tmp_path / "runs") == [
        {"run_id": "rank-r8", "attempt_id": "attempt-001", "status": "completed"},
        {"run_id": "rank-r32", "attempt_id": None, "status": "not_started"},
    ]


def test_cli_status_only_does_not_require_gpu_arguments(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"runs": [{"run_id": "rank-r8"}]}) + "\n", encoding="utf-8")
    script = SCRIPTS / "run_experiments.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            str(matrix),
            "--run-root",
            str(tmp_path / "runs"),
            "--status-only",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPTS)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "not_started"' in result.stdout


def test_launch_error_is_recorded_without_raising(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "rank-r8.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "model_name_or_path": "/model",
                "dataset_dir": "/data",
                "output_dir": "/frozen",
                "run_name": "rank-r8",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class FakeSampler:
        pid = 101

        def poll(self):
            return 0

    calls = 0

    def fake_popen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeSampler()
        raise OSError("launcher unavailable")

    monkeypatch.setattr(run_experiments, "environment_snapshot", lambda: "environment")
    monkeypatch.setattr(run_experiments, "query_preflight", lambda *_args: ([], {"errors": []}))
    monkeypatch.setattr(run_experiments.subprocess, "Popen", fake_popen)

    status = run_attempt(
        "rank-r8",
        source,
        tmp_path / "runs" / "rank-r8" / "attempt-001",
        tmp_path,
    )

    assert status["status"] == "failed"
    assert status["exit_code"] is None
    assert "launcher unavailable" in status["failure_reason"]


def test_hard_timeout_terminates_training_and_records_failure(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "rank-r8.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "model_name_or_path": "/model",
                "dataset_dir": "/data",
                "output_dir": "/frozen",
                "run_name": "rank-r8",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class FakeSampler:
        pid = 101

        def poll(self):
            return 0

    class FakeTraining:
        pid = 202
        terminated = False

        def wait(self, timeout=None):
            if not self.terminated:
                raise subprocess.TimeoutExpired("train", timeout)
            return -15

        def poll(self):
            return None if not self.terminated else -15

        def send_signal(self, _signal):
            self.terminated = True

        def kill(self):
            self.terminated = True

    training = FakeTraining()
    processes = iter([FakeSampler(), training])
    monkeypatch.setattr(run_experiments, "environment_snapshot", lambda: "environment")
    monkeypatch.setattr(run_experiments, "query_preflight", lambda *_args: ([], {"errors": []}))
    monkeypatch.setattr(run_experiments.subprocess, "Popen", lambda *_args, **_kwargs: next(processes))

    status = run_attempt(
        "rank-r8",
        source,
        tmp_path / "runs" / "rank-r8" / "attempt-001",
        tmp_path,
        timeout_seconds=1,
    )

    assert training.terminated is True
    assert status["status"] == "failed"
    assert "hard timeout" in status["failure_reason"]


def test_missing_preflight_command_returns_auditable_exit_code() -> None:
    result = run_capture(["definitely-not-a-real-command-day12"])

    assert result.returncode == 127
    assert "No such file" in result.stderr
