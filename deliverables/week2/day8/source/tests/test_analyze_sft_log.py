import csv
import json
import subprocess
import sys
from pathlib import Path


DAY8_ROOT = Path(__file__).resolve().parents[2]
ANALYZER = DAY8_ROOT / "source" / "scripts" / "analyze_sft_log.py"


def test_analyzer_extracts_every_loss_and_reports_downward_trend(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "trainer_log.jsonl"
    records = [
        {"step": 1, "loss": 3.0, "learning_rate": 0.0001, "epoch": 0.1},
        {"step": 2, "loss": 2.8, "lr": 0.00009, "epoch": 0.2},
        {"step": 3, "loss": 2.4, "learning_rate": 0.00008, "epoch": 0.3},
        {"step": 4, "loss": 2.1, "learning_rate": 0.00007, "epoch": 0.4},
    ]
    log_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    csv_path = tmp_path / "loss_by_step.csv"
    summary_path = tmp_path / "loss_summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--trainer-log",
            str(log_path),
            "--csv-output",
            str(csv_path),
            "--summary-output",
            str(summary_path),
            "--expected-steps",
            "4",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["loss_record_count"] == 4
    assert summary["step_sequence_complete"] is True
    assert summary["all_losses_finite"] is True
    assert summary["linear_slope"] < 0
    assert summary["last_window_mean"] < summary["first_window_mean"]
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["step"]) for row in rows] == [1, 2, 3, 4]
    assert float(rows[1]["learning_rate"]) == 0.00009


def test_analyzer_rejects_nonfinite_loss(tmp_path: Path) -> None:
    log_path = tmp_path / "trainer_log.jsonl"
    log_path.write_text(
        '{"step": 1, "loss": 2.0}\n{"step": 2, "loss": NaN}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--trainer-log",
            str(log_path),
            "--csv-output",
            str(tmp_path / "loss.csv"),
            "--summary-output",
            str(tmp_path / "summary.json"),
            "--expected-steps",
            "2",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "non-finite" in (result.stdout + result.stderr).lower()
