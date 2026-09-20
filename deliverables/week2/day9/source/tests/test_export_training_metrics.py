import csv
import json
import math
from pathlib import Path
import subprocess
import sys


DAY9_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = DAY9_ROOT / "source" / "scripts" / "export_training_metrics.py"


def run_export(
    tmp_path: Path,
    history: list[dict],
    *,
    expected_steps: int,
) -> subprocess.CompletedProcess[str]:
    trainer_state = tmp_path / "trainer_state.json"
    trainer_state.write_text(
        json.dumps({"log_history": history}),
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trainer-state",
            str(trainer_state),
            "--csv-output",
            str(tmp_path / "train_loss_by_step.csv"),
            "--summary-output",
            str(tmp_path / "loss_summary.json"),
            "--expected-steps",
            str(expected_steps),
            "--moving-average-window",
            "2",
        ],
        capture_output=True,
        text=True,
    )


def test_exports_every_optimizer_step_and_preserves_available_fields(
    tmp_path: Path,
) -> None:
    history = [
        {
            "step": 1,
            "epoch": 0.1,
            "loss": 3.0,
            "learning_rate": 0.0001,
            "grad_norm": 1.2,
        },
        {
            "step": 2,
            "epoch": 0.2,
            "loss": 2.5,
            "learning_rate": 0.00009,
        },
        {
            "step": 3,
            "epoch": 0.3,
            "loss": 2.0,
            "learning_rate": 0.00008,
            "grad_norm": 0.9,
        },
        {
            "step": 4,
            "epoch": 0.4,
            "loss": 1.5,
            "learning_rate": 0.00007,
            "grad_norm": 0.8,
        },
        {"step": 4, "train_runtime": 12.3},
    ]

    result = run_export(tmp_path, history, expected_steps=4)

    assert result.returncode == 0, result.stdout + result.stderr
    with (tmp_path / "train_loss_by_step.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "step",
        "epoch",
        "loss",
        "learning_rate",
        "grad_norm",
    ]
    assert [int(row["step"]) for row in rows] == [1, 2, 3, 4]
    assert rows[1]["grad_norm"] == ""

    summary = json.loads(
        (tmp_path / "loss_summary.json").read_text(encoding="utf-8")
    )
    assert summary["loss_record_count"] == 4
    assert summary["step_sequence_complete"] is True
    assert summary["all_losses_finite"] is True
    assert summary["first_loss"] == 3.0
    assert summary["last_loss"] == 1.5
    assert summary["first_decile_mean"] == 3.0
    assert summary["last_decile_mean"] == 1.5
    assert summary["moving_average_window"] == 2
    assert summary["moving_average_last"] == 1.75
    assert summary["overall_downward_trend"] is True


def test_rejects_missing_optimizer_step(tmp_path: Path) -> None:
    history = [
        {"step": 1, "loss": 3.0},
        {"step": 3, "loss": 2.0},
    ]

    result = run_export(tmp_path, history, expected_steps=3)

    assert result.returncode != 0
    assert "expected contiguous steps 1..3" in result.stderr


def test_rejects_nonfinite_loss(tmp_path: Path) -> None:
    history = [
        {"step": 1, "loss": 3.0},
        {"step": 2, "loss": math.inf},
    ]

    result = run_export(tmp_path, history, expected_steps=2)

    assert result.returncode != 0
    assert "non-finite loss" in result.stderr
