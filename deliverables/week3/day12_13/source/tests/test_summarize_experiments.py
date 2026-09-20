import json
import hashlib
import math
import sys
from pathlib import Path

import yaml


DAY12_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = DAY12_ROOT / "source" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_experiments import (  # noqa: E402
    ExperimentSpec,
    expected_optimizer_steps,
    summarize_log_history,
    summarize_run,
)


def baseline_spec(epochs: float = 3.0) -> ExperimentSpec:
    return ExperimentSpec(
        run_id="rank-r8",
        group="rank",
        lora_rank=8,
        learning_rate=1e-4,
        num_train_epochs=epochs,
    )


def write_run_fixture(
    root: Path,
    *,
    exit_code: int = 0,
    global_step: int = 3747,
    max_steps: int = 3747,
    losses: list[float] | None = None,
    train_runtime: float = 100.0,
    wall_clock_runtime: float = 112.5,
    adapter_weights: bool = True,
) -> Path:
    attempt = root / "rank-r8" / "attempt-001"
    adapter = attempt / "adapter"
    adapter.mkdir(parents=True)
    losses = losses if losses is not None else [1.2, 1.1, 1.0]
    history = [
        {"step": index + 1, "loss": value, "epoch": (index + 1) / len(losses)}
        for index, value in enumerate(losses)
    ]
    (adapter / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step, "max_steps": max_steps, "log_history": history}, allow_nan=True),
        encoding="utf-8",
    )
    (adapter / "train_results.json").write_text(
        json.dumps({"train_runtime": train_runtime, "train_loss": 1.1}), encoding="utf-8"
    )
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    if adapter_weights:
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        adapter_hash = hashlib.sha256(b"adapter").hexdigest()
        (attempt / "adapter_files.sha256").write_text(
            f"{adapter_hash}  adapter_model.safetensors\n", encoding="utf-8"
        )
    else:
        (attempt / "adapter_files.sha256").write_text("", encoding="utf-8")
    source = attempt / "source_config.yaml"
    runtime = attempt / "runtime_config.yaml"
    source.write_text("seed: 42\n", encoding="utf-8")
    runtime.write_text(
        yaml.safe_dump({"output_dir": str(adapter), "seed": 42}, sort_keys=False),
        encoding="utf-8",
    )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    runtime_hash = hashlib.sha256(runtime.read_bytes()).hexdigest()
    (attempt / "status.json").write_text(
        json.dumps(
            {
                "run_id": "rank-r8",
                "attempt_id": "attempt-001",
                "status": "completed",
                "exit_code": exit_code,
                "wall_clock_runtime_seconds": wall_clock_runtime,
                "source_config_sha256": source_hash,
                "runtime_config_sha256": runtime_hash,
                "failure_reason": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (attempt / "gpu_memory.csv").write_text(
        "2026/08/04 10:00:00.000, 8000, 90\n2026/08/04 10:00:01.000, 9100, 95\n",
        encoding="utf-8",
    )
    return attempt


def test_final_loss_is_last_logging_step_and_mean_uses_last_100() -> None:
    rows = [{"step": index, "loss": float(index)} for index in range(1, 102)]

    summary = summarize_log_history(rows)

    assert summary.status == "valid"
    assert summary.final_loss == 101.0
    assert summary.last_100_loss_mean == 51.5


def test_any_nonfinite_loss_invalidates_history() -> None:
    rows = [{"step": 1, "loss": math.inf}, {"step": 2, "loss": 1.0}]

    summary = summarize_log_history(rows)

    assert summary.status == "invalid_nonfinite_loss"
    assert summary.final_loss is None


def test_expected_optimizer_steps_are_frozen_for_three_epoch_values() -> None:
    assert expected_optimizer_steps(baseline_spec(2.0), training_records=4999, effective_batch=4) == 2498
    assert expected_optimizer_steps(baseline_spec(3.0), training_records=4999, effective_batch=4) == 3747
    assert expected_optimizer_steps(baseline_spec(5.0), training_records=4999, effective_batch=4) == 6245


def test_valid_run_reports_teacher_and_auxiliary_metrics(tmp_path: Path) -> None:
    attempt = write_run_fixture(tmp_path)

    summary = summarize_run("rank-r8", attempt, baseline_spec())

    assert summary.status == "completed"
    assert summary.final_loss == 1.0
    assert summary.train_runtime_seconds == 100.0
    assert summary.wall_clock_runtime_seconds == 112.5
    assert summary.sampled_peak_gpu_memory_mib == 9100
    assert summary.optimizer_steps == 3747
    assert summary.expected_optimizer_steps == 3747
    assert summary.adapter_size_bytes == len(b"adapter")


def test_exit_zero_with_incomplete_steps_is_invalid(tmp_path: Path) -> None:
    attempt = write_run_fixture(tmp_path, global_step=3000, max_steps=3747)

    summary = summarize_run("rank-r8", attempt, baseline_spec())

    assert summary.status == "invalid_incomplete_steps"


def test_nonfinite_loss_invalidates_run_even_if_last_loss_is_finite(tmp_path: Path) -> None:
    attempt = write_run_fixture(tmp_path, losses=[math.nan, 1.0])

    summary = summarize_run("rank-r8", attempt, baseline_spec())

    assert summary.status == "invalid_nonfinite_loss"


def test_missing_adapter_weight_invalidates_run(tmp_path: Path) -> None:
    attempt = write_run_fixture(tmp_path, adapter_weights=False)

    summary = summarize_run("rank-r8", attempt, baseline_spec())

    assert summary.status == "invalid_adapter"


def test_nonzero_exit_code_remains_failed(tmp_path: Path) -> None:
    attempt = write_run_fixture(tmp_path, exit_code=1)

    summary = summarize_run("rank-r8", attempt, baseline_spec())

    assert summary.status == "failed"


def test_modified_runtime_config_invalidates_completed_run(tmp_path: Path) -> None:
    attempt = write_run_fixture(tmp_path)
    (attempt / "runtime_config.yaml").write_text("seed: 7\n", encoding="utf-8")

    summary = summarize_run("rank-r8", attempt, baseline_spec())

    assert summary.status == "invalid_config_hash"


def test_modified_adapter_weight_invalidates_completed_run(tmp_path: Path) -> None:
    attempt = write_run_fixture(tmp_path)
    (attempt / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")

    summary = summarize_run("rank-r8", attempt, baseline_spec())

    assert summary.status == "invalid_adapter_hash"


def test_completed_status_with_failure_reason_is_invalid(tmp_path: Path) -> None:
    attempt = write_run_fixture(tmp_path)
    status_path = attempt / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["failure_reason"] = "unexpected warning promoted to failure"
    status_path.write_text(json.dumps(status) + "\n", encoding="utf-8")

    summary = summarize_run("rank-r8", attempt, baseline_spec())

    assert summary.status == "invalid_failure_reason"
