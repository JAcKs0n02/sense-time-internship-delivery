import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "freeze_selected_model_input.py"


def load_module():
    spec = importlib.util.spec_from_file_location("freeze_selected_model_input", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_freeze_accepts_actual_day14_model_id_schema_and_adds_audit_hashes(tmp_path):
    module = load_module()
    best = write_json(
        tmp_path / "best_model.json",
        {
            "status": "selected_from_two_human_raters",
            "model_id": "epoch-e5",
            "adapter_path": "runs/epoch-e5/attempt-001/adapter",
            "weighted_total": 3.78875,
            "automatic_metrics_used_for_ranking": False,
        },
    )
    human = (tmp_path / "human_scores.csv")
    human.write_text("model_id,weighted_total\nepoch-e5,3.78875\n", encoding="utf-8")
    reviewer_1 = (tmp_path / "reviewer_1.csv")
    reviewer_1.write_text("answer_id,accuracy\na,5\n", encoding="utf-8")
    reviewer_2 = (tmp_path / "reviewer_2.csv")
    reviewer_2.write_text("answer_id,accuracy\na,4\n", encoding="utf-8")
    adapter_manifest = tmp_path / "adapter_files.sha256"
    adapter_manifest.write_text(
        "cd2822fc5b160095524aa0203b1e2bab8e71e7897535cf093f4e8807c540a75a  adapter_model.safetensors\n",
        encoding="utf-8",
    )

    frozen = module.freeze_selected_model_input(
        best_model_path=best,
        human_scores_path=human,
        reviewer_paths=[reviewer_1, reviewer_2],
        adapter_manifest_path=adapter_manifest,
        remote_root="/root/autodl-tmp/qwen25-week3",
        selection_commit="c65d1d720041c19b32be5ca39e171142cad85157",
        selection_timestamp="2026-08-06T21:09:14+08:00",
    )

    assert frozen["selected_run_id"] == "epoch-e5"
    assert frozen["adapter_path"] == "/root/autodl-tmp/qwen25-week3/runs/epoch-e5/attempt-001/adapter"
    assert frozen["adapter_model_sha256"] == "cd2822fc5b160095524aa0203b1e2bab8e71e7897535cf093f4e8807c540a75a"
    assert frozen["selection_source"] == "Day 14 two-rater weighted human score"
    assert frozen["opencompass_used_for_selection"] is False
    assert frozen["source_sha256"]["best_model_json"]
    assert frozen["source_sha256"]["human_scores_csv"]
    assert len(frozen["source_sha256"]["reviewer_score_files"]) == 2


def test_freeze_rejects_selection_file_contaminated_with_opencompass_result(tmp_path):
    module = load_module()
    best = write_json(
        tmp_path / "best_model.json",
        {
            "model_id": "epoch-e5",
            "adapter_path": "runs/epoch-e5/attempt-001/adapter",
            "opencompass_score": 80.0,
        },
    )
    human = tmp_path / "human_scores.csv"
    human.write_text("model_id,weighted_total\nepoch-e5,3.78875\n", encoding="utf-8")
    reviewer = tmp_path / "reviewer.csv"
    reviewer.write_text("answer_id,accuracy\na,5\n", encoding="utf-8")
    adapter_manifest = tmp_path / "adapter_files.sha256"
    adapter_manifest.write_text(
        "cd2822fc5b160095524aa0203b1e2bab8e71e7897535cf093f4e8807c540a75a  adapter_model.safetensors\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="OpenCompass"):
        module.freeze_selected_model_input(
            best_model_path=best,
            human_scores_path=human,
            reviewer_paths=[reviewer, reviewer],
            adapter_manifest_path=adapter_manifest,
            remote_root="/root/autodl-tmp/qwen25-week3",
            selection_commit="c65d1d7",
            selection_timestamp="2026-08-06T21:09:14+08:00",
        )


def test_freeze_requires_adapter_weight_hash(tmp_path):
    module = load_module()
    best = write_json(
        tmp_path / "best_model.json",
        {"model_id": "epoch-e5", "adapter_path": "runs/epoch-e5/attempt-001/adapter"},
    )
    human = tmp_path / "human_scores.csv"
    human.write_text("model_id,weighted_total\nepoch-e5,3.78875\n", encoding="utf-8")
    reviewer = tmp_path / "reviewer.csv"
    reviewer.write_text("answer_id,accuracy\na,5\n", encoding="utf-8")
    adapter_manifest = tmp_path / "adapter_files.sha256"
    adapter_manifest.write_text("abc  adapter_config.json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        module.freeze_selected_model_input(
            best_model_path=best,
            human_scores_path=human,
            reviewer_paths=[reviewer, reviewer],
            adapter_manifest_path=adapter_manifest,
            remote_root="/root/autodl-tmp/qwen25-week3",
            selection_commit="c65d1d7",
            selection_timestamp="2026-08-06T21:09:14+08:00",
        )
