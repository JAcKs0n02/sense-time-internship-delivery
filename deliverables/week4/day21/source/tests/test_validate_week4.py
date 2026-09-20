from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_week4.py"
REPO = Path(__file__).parents[5]


def load_module():
    spec = importlib.util.spec_from_file_location("validate_week4", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_minimal_week4_repo(tmp_path: Path) -> Path:
    target = tmp_path / "repo"
    shutil.copytree(REPO / "Submission" / "Week4", target / "Submission" / "Week4")
    return target


def test_collects_exact_cross_day_facts_from_frozen_evidence():
    module = load_module()
    facts = module.collect_week4_facts(REPO)
    assert facts["day17"] == {"preference_types": 5, "example_pairs": 10, "valid": True}
    assert facts["day18"]["source_counts"] == {"ultrafeedback": 500, "self_built": 210}
    assert facts["day18"]["total_count"] == 710
    assert (facts["day18"]["train_count"], facts["day18"]["validation_count"]) == (639, 71)
    assert facts["day19"]["status"] == "completed"
    assert facts["day18"]["corrective_full_count"] == 870
    assert (facts["day18"]["corrective_train_count"], facts["day18"]["corrective_validation_count"]) == (783, 87)
    assert (facts["day19"]["global_step"], facts["day19"]["max_steps"]) == (40, 40)
    assert facts["day19"]["adapter_sha256"] == "d7a932ee4f28c8950db289126381f5d4dd30a037b238852e87ad8e5a241f9e52"
    assert facts["day20"]["safety_prompt_count"] == 10
    assert facts["day20"]["business_prompt_count"] == 5
    assert facts["day20"]["teacher_refusal_rate"] == 1.0
    assert facts["day20"]["enhanced_safe_completion_rate"] == 0.4
    assert facts["day20"]["business_means"] == {"sft_only": 3.595, "sft_dpo": 3.64}
    assert facts["remediation"]["candidate"] == "reward-corrective-40step"
    assert facts["remediation"]["promotion_allowed"] is True


def test_builds_final_archive_with_complete_lineage_and_no_weights():
    module = load_module()
    facts = module.collect_week4_facts(REPO)
    archive = module.build_final_archive(facts, module.validate_facts(facts))
    assert archive["status"] == "ready"
    assert archive["selected_model"] == "reward_corrective_40step_merged"
    assert archive["week3_sft"]["manifest_sha256"] == "e4512f7dd0f85b2f1d8154d123aec176295f557046142aab406a030c8e76e971"
    assert archive["dpo_adapter"]["sha256"] == "d7a932ee4f28c8950db289126381f5d4dd30a037b238852e87ad8e5a241f9e52"
    assert archive["merged_dpo"]["manifest_sha256"] == "aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c"
    assert archive["merged_dpo"]["weight_shards"] == 4
    assert archive["merged_dpo"]["weight_bytes"] == 15231271872
    assert archive["validation"]["merged_model_load"] == "PASS"
    assert archive["validation"]["harmless_generation_smoke"] == "PASS"
    assert archive["evaluation"]["teacher_refusal_rate"] == 1.0
    assert archive["remediation_candidate_promoted"] is True
    assert archive["model_weights_in_submission"] is False


def test_acceptance_matrix_reports_passes_and_quality_limitation_separately():
    module = load_module()
    facts = module.collect_week4_facts(REPO)
    text = module.build_acceptance_matrix(facts, module.validate_facts(facts))
    assert "偏好数据不少于 300 条" in text and "710" in text
    assert "安全拒绝率不少于 90%" in text and "10/10" in text
    assert "周报与最终 DPO 模型归档" in text
    assert "业务质量对比" in text and "描述性提升" in text
    assert "增强安全响应" in text and "40%" in text


def test_failed_gate_produces_blocked_archive_and_failed_matrix():
    module = load_module()
    facts = copy.deepcopy(module.collect_week4_facts(REPO))
    facts["day20"]["teacher_refusal_count"] = 8
    facts["day20"]["teacher_refusal_rate"] = 0.8
    checks = module.validate_facts(facts)
    archive = module.build_final_archive(facts, checks)
    matrix = module.build_acceptance_matrix(facts, checks)
    assert any(row == {"check": "day20_safety", "status": "FAIL"} for row in checks)
    assert archive["status"] == "blocked"
    assert archive["selected_model"] is None
    assert "| 安全拒绝率不少于 90% | FAIL |" in matrix


def test_teacher_reward_trend_is_a_strict_fail_closed_gate():
    module = load_module()
    facts = copy.deepcopy(module.collect_week4_facts(REPO))
    facts["day19"]["reward_trend"].update(
        {
            "chosen_reward_window_delta": 0.5,
            "rejected_reward_window_delta": 0.01,
            "margin_window_delta": 0.49,
            "teacher_trend_observed": False,
        }
    )
    checks = module.validate_facts(facts)
    archive = module.build_final_archive(facts, checks)
    matrix = module.build_acceptance_matrix(facts, checks)
    assert {"check": "day19_reward_trend", "status": "FAIL"} in checks
    assert archive["status"] == "blocked"
    assert "| Rewards：Chosen 上升且 Rejected 下降 | FAIL |" in matrix


def test_completed_corrective_run_can_use_40_contiguous_steps():
    module = load_module()
    facts = copy.deepcopy(module.collect_week4_facts(REPO))
    facts["day19"].update({"global_step": 40, "max_steps": 40, "metric_row_count": 40})
    facts["day19"]["reward_trend"].update(
        {
            "chosen_reward_window_delta": 0.03,
            "rejected_reward_window_delta": -0.002,
            "margin_window_delta": 0.032,
            "teacher_trend_observed": True,
        }
    )
    checks = module.validate_facts(facts)
    assert {"check": "day19_training", "status": "PASS"} in checks
    assert {"check": "day19_reward_trend", "status": "PASS"} in checks


def test_collect_rejects_dataset_hash_drift(tmp_path: Path):
    module = load_module()
    repo = copy_minimal_week4_repo(tmp_path)
    train = repo / "Submission" / "Week4" / "Day18_Preference_Dataset" / "DPO_Train.json"
    train.write_bytes(train.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Day 18 dataset hash"):
        module.collect_week4_facts(repo)


def test_collect_rejects_summary_record_drift(tmp_path: Path):
    module = load_module()
    repo = copy_minimal_week4_repo(tmp_path)
    summary_path = repo / "Submission" / "Week4" / "Day20_DPO_Merge_and_Evaluation" / "Safety_Test_Summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["models"]["sft_dpo"]["teacher_refusal_count"] = 9
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="safety summary"):
        module.collect_week4_facts(repo)


def test_collect_rejects_business_summary_record_drift(tmp_path: Path):
    module = load_module()
    repo = copy_minimal_week4_repo(tmp_path)
    summary_path = repo / "Submission" / "Week4" / "Day20_DPO_Merge_and_Evaluation" / "Business_Quality_Summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["model_weighted_means"]["sft_dpo"] = 4.999
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="business summary"):
        module.collect_week4_facts(repo)


def test_collect_rejects_corrective_gate_drift(tmp_path: Path):
    module = load_module()
    repo = copy_minimal_week4_repo(tmp_path)
    selection_path = repo / "Submission" / "Week4" / "Day19_DPO_Training" / "Corrective_Development_Review.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["passes_all_promotion_gates"] = False
    selection_path.write_text(json.dumps(selection, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="corrective passes_all_promotion_gates"):
        module.collect_week4_facts(repo)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("submitted_model", "unexpected_model", "submitted model"),
        ("teacher_test_reused_for_candidate_selection", True, "teacher-test isolation"),
    ],
)
def test_collect_rejects_final_selection_drift(tmp_path: Path, field: str, value: object, message: str):
    module = load_module()
    repo = copy_minimal_week4_repo(tmp_path)
    selection_path = repo / "Submission" / "Week4" / "Day20_DPO_Merge_and_Evaluation" / "Final_Model_Selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection[field] = value
    selection_path.write_text(json.dumps(selection, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        module.collect_week4_facts(repo)


def test_cli_overwrites_stale_ready_outputs_when_evidence_is_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_module()
    repo = copy_minimal_week4_repo(tmp_path)
    train = repo / "Submission" / "Week4" / "Day18_Preference_Dataset" / "DPO_Train.json"
    train.write_bytes(train.read_bytes() + b"\n")
    archive = tmp_path / "archive.json"
    matrix = tmp_path / "matrix.md"
    validation = tmp_path / "validation.json"
    archive.write_text('{"status":"ready"}\n', encoding="utf-8")
    matrix.write_text("| check | PASS |\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_week4.py",
            "--repo",
            str(repo),
            "--archive-output",
            str(archive),
            "--matrix-output",
            str(matrix),
            "--validation-output",
            str(validation),
        ],
    )
    assert module.main() == 1
    assert json.loads(archive.read_text(encoding="utf-8"))["status"] == "blocked"
    assert "PASS" not in matrix.read_text(encoding="utf-8")
    assert json.loads(validation.read_text(encoding="utf-8"))["status"] == "FAIL"
