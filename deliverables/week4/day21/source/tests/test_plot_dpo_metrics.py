from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_dpo_metrics.py"
REPO = Path(__file__).parents[5]
SOURCE = REPO / "Submission" / "Week4" / "Day19_DPO_Training" / "DPO_Metrics_By_Step.csv"
SUMMARY = REPO / "Submission" / "Week4" / "Day19_DPO_Training" / "DPO_Training_Summary.json"


def load_module():
    spec = importlib.util.spec_from_file_location("plot_dpo_metrics", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalizes_exact_40_step_metric_contract():
    module = load_module()
    rows = module.load_and_normalize(SOURCE)
    assert len(rows) == 40
    assert [row["step"] for row in rows] == list(range(1, 41))
    assert rows[-1]["epoch"] == pytest.approx(0.4086845466155811)
    assert rows[-1]["chosen_ma20"] == pytest.approx(0.03400641945190728)
    assert rows[-1]["rejected_ma20"] == pytest.approx(0.004416526667773724)
    assert rows[-1]["margin_ma20"] == pytest.approx(0.029589892784133555)
    assert all(row["margin"] == pytest.approx(row["chosen"] - row["rejected"], abs=1e-6) for row in rows)
    assert module.sha256(SOURCE) == "55c41c3c99494bcbb5c9c3eda78c156a46c67ac9fe27fe7addf16b53a9418e13"


def test_rejects_step_gaps_and_non_finite_metrics(tmp_path: Path):
    module = load_module()
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    gap = tmp_path / "gap.csv"
    with gap.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[1:])
    with pytest.raises(ValueError, match="steps"):
        module.load_and_normalize(gap)
    rows[0]["rewards/chosen"] = "nan"
    invalid = tmp_path / "nan.csv"
    with invalid.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="finite"):
        module.load_and_normalize(invalid)


def test_renders_four_panel_png_and_manifest(tmp_path: Path):
    module = load_module()
    rows = module.load_and_normalize(SOURCE)
    validation = module.load_validation(SUMMARY)
    assert [row["step"] for row in validation] == [20, 40]
    output = tmp_path / "curves.png"
    module.render_plot(rows, validation, output, beta=0.1, moving_average_window=20)
    assert output.stat().st_size > 100_000
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width >= 1800
        assert image.height >= 1200


def test_accepts_a_completed_40_step_corrective_run(tmp_path: Path):
    module = load_module()
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))[:40]
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(source_rows)
    rows = module.load_and_normalize(metrics)
    assert len(rows) == 40
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary["evaluation_history"] = summary["evaluation_history"]
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    validation = module.load_validation(summary_path)
    assert [row["step"] for row in validation] == [20, 40]
    output = tmp_path / "corrective.png"
    module.render_plot(rows, validation, output, beta=0.1, moving_average_window=20)
    with Image.open(output) as image:
        assert image.width >= 1800 and image.height >= 1200
