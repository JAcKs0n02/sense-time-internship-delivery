import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_opencompass.py"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_opencompass", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_score_table_requires_and_orders_the_four_formal_results():
    module = load_module()
    rows = [
        {"model": "best_sft", "dataset": "cmmlu", "score": 63.5, "status": "completed"},
        {"model": "base", "dataset": "ceval", "score": 71.0, "status": "completed"},
        {"model": "best_sft", "dataset": "ceval", "score": 69.5, "status": "completed"},
        {"model": "base", "dataset": "cmmlu", "score": 65.0, "status": "completed"},
    ]

    table = module.build_score_table(rows)

    assert [(row["model"], row["dataset"]) for row in table] == [
        ("base", "ceval"),
        ("base", "cmmlu"),
        ("best_sft", "ceval"),
        ("best_sft", "cmmlu"),
    ]
    assert table[0]["delta_vs_base"] == 0.0
    assert table[1]["delta_vs_base"] == 0.0
    assert table[2]["delta_vs_base"] == -1.5
    assert table[3]["delta_vs_base"] == -1.5


def test_build_score_table_rejects_missing_model_dataset_pair():
    module = load_module()
    rows = [
        {"model": "base", "dataset": "ceval", "score": 71.0, "status": "completed"},
        {"model": "base", "dataset": "cmmlu", "score": 65.0, "status": "completed"},
        {"model": "best_sft", "dataset": "ceval", "score": 69.5, "status": "completed"},
    ]

    with pytest.raises(ValueError, match="missing formal result"):
        module.build_score_table(rows)


def test_build_score_table_rejects_failed_or_out_of_range_scores():
    module = load_module()
    rows = [
        {"model": "base", "dataset": "ceval", "score": 71.0, "status": "completed"},
        {"model": "base", "dataset": "cmmlu", "score": 65.0, "status": "completed"},
        {"model": "best_sft", "dataset": "ceval", "score": 101.0, "status": "completed"},
        {"model": "best_sft", "dataset": "cmmlu", "score": 63.5, "status": "failed"},
    ]

    with pytest.raises(ValueError):
        module.build_score_table(rows)
