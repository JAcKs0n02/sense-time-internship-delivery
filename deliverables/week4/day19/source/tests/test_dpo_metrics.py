from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_dpo_metrics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_dpo_metrics", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_state(path: Path, history: list[dict]) -> None:
    path.write_text(json.dumps({"log_history": history}), encoding="utf-8")


def test_extracts_training_reward_metrics_and_ignores_eval_rows(tmp_path: Path):
    module = load_module()
    state = tmp_path / "trainer_state.json"
    write_state(
        state,
        [
            {
                "step": 1,
                "epoch": 0.1,
                "loss": 0.71,
                "rewards/chosen": -0.8,
                "rewards/rejected": -0.6,
                "rewards/margins": -0.2,
                "rewards/accuracies": 0.25,
                "logps/chosen": -12.0,
                "logps/rejected": -11.0,
                "learning_rate": 5e-6,
                "grad_norm": 1.2,
            },
            {"step": 1, "eval_loss": 0.69},
            {
                "step": 2,
                "epoch": 0.2,
                "loss": 0.66,
                "rewards/chosen": -0.3,
                "rewards/rejected": -0.9,
                "rewards/margins": 0.6,
                "rewards/accuracies": 0.75,
                "logps/chosen": -10.0,
                "logps/rejected": -13.0,
                "learning_rate": 4e-6,
                "grad_norm": 0.9,
            },
        ],
    )

    rows = module.extract_rows(state)

    assert [row["step"] for row in rows] == [1, 2]
    assert rows[-1]["rewards/chosen"] == pytest.approx(-0.3)
    assert rows[-1]["rewards/rejected"] == pytest.approx(-0.9)


def test_summary_reports_direction_and_margin_improvement(tmp_path: Path):
    module = load_module()
    rows = [
        {"step": 1, "rewards/chosen": -1.0, "rewards/rejected": -0.5, "rewards/margins": -0.5},
        {"step": 2, "rewards/chosen": -0.2, "rewards/rejected": -1.1, "rewards/margins": 0.9},
    ]

    summary = module.summarize(rows)

    assert summary["logged_steps"] == 2
    assert summary["chosen_reward_delta"] == pytest.approx(0.8)
    assert summary["rejected_reward_delta"] == pytest.approx(-0.6)
    assert summary["margin_delta"] == pytest.approx(1.4)
    assert summary["teacher_trend_observed"] is True


def test_rejects_nonfinite_or_incomplete_reward_rows(tmp_path: Path):
    module = load_module()
    state = tmp_path / "trainer_state.json"
    write_state(
        state,
        [
            {
                "step": 1,
                "loss": 0.5,
                "rewards/chosen": float("nan"),
                "rewards/rejected": -0.4,
                "rewards/margins": 0.1,
            }
        ],
    )

    with pytest.raises(ValueError, match="finite"):
        module.extract_rows(state)


def test_empty_history_cannot_be_summarized():
    module = load_module()
    with pytest.raises(ValueError, match="No DPO reward metrics"):
        module.summarize([])


def test_adds_trailing_moving_averages_without_replacing_raw_values():
    module = load_module()
    rows = [
        {"rewards/chosen": 1.0, "rewards/rejected": -1.0, "rewards/margins": 2.0},
        {"rewards/chosen": 3.0, "rewards/rejected": -3.0, "rewards/margins": 6.0},
        {"rewards/chosen": 5.0, "rewards/rejected": -5.0, "rewards/margins": 10.0},
    ]

    enriched = module.add_moving_averages(rows, window=2)

    assert enriched[-1]["rewards/chosen"] == 5.0
    assert enriched[-1]["rewards/chosen_ma"] == pytest.approx(4.0)
    assert enriched[-1]["rewards/rejected_ma"] == pytest.approx(-4.0)
    assert enriched[-1]["rewards/margins_ma"] == pytest.approx(8.0)
    assert "rewards/chosen_ma" not in rows[-1]


def test_summary_compares_non_overlapping_reward_windows():
    module = load_module()
    rows = [
        {"step": i + 1, "rewards/chosen": value, "rewards/rejected": -value, "rewards/margins": 2 * value}
        for i, value in enumerate([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    ]

    summary = module.summarize(rows, trend_window=3)

    assert summary["trend_window"] == 3
    assert summary["chosen_reward_window_delta"] == pytest.approx(1.0)
    assert summary["rejected_reward_window_delta"] == pytest.approx(-1.0)
    assert summary["teacher_trend_observed"] is True
