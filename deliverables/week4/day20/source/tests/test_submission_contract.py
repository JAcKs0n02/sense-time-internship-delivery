from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).parents[5]
SUBMISSION = REPO / "Submission" / "Week4" / "Day20_DPO_Merge_and_Evaluation"
WEEK4_README = REPO / "Submission" / "Week4" / "README.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_day20_teacher_submission_has_required_tables_and_honest_results():
    required = {
        "README.md",
        "Safety_Test_Records.csv",
        "Safety_Test_Summary.json",
        "Business_Quality_Comparison.csv",
        "Business_Quality_Summary.json",
        "Evaluation_Config.json",
        "Evaluation_Prompts.json",
        "Merged_Model_Manifest.json",
        "Merged_Model_Smoke.json",
        "Final_Model_Selection.json",
    }
    assert {path.name for path in SUBMISSION.iterdir() if path.is_file()} == required
    assert len(read_csv(SUBMISSION / "Safety_Test_Records.csv")) == 10
    assert len(read_csv(SUBMISSION / "Business_Quality_Comparison.csv")) == 5

    safety = json.loads((SUBMISSION / "Safety_Test_Summary.json").read_text())
    business = json.loads((SUBMISSION / "Business_Quality_Summary.json").read_text())
    selection = json.loads((SUBMISSION / "Final_Model_Selection.json").read_text())
    dpo = safety["models"]["sft_dpo"]
    assert safety["schema_version"] == "2.0"
    assert safety["teacher_success_definition"] == "score >= 1"
    assert dpo["denominator"] == 10
    assert dpo["teacher_refusal_count"] == 10
    assert dpo["teacher_refusal_rate"] == 1.0
    assert dpo["enhanced_safe_completion_count"] == 4
    assert dpo["enhanced_safe_completion_rate"] == 0.4
    assert dpo["passes_90_percent_gate"] is True
    readme = (SUBMISSION / "README.md").read_text(encoding="utf-8")
    assert "安全拒绝率：`10/10 = 100%`" in readme
    assert "增强安全响应率：`4/10 = 40%`" in readme
    assert "安全拒绝率为 `4/10" not in readme
    assert business["model_prompt_counts"] == {"sft_dpo": 5, "sft_only": 5}
    assert business["weighted_mean_delta_sft_dpo_minus_sft_only"] == 0.045
    assert business["model_weighted_means"] == {"sft_dpo": 3.64, "sft_only": 3.595}
    assert selection["submitted_model"] == "reward_corrective_40step_merged"
    assert selection["prior_safety_remediation_candidates_evaluated"] == 3
    assert selection["corrective_candidate_promoted"] is True
    assert selection["teacher_test_reused_for_candidate_selection"] is False
    assert selection["submitted_model_teacher_refusal_rate"] == dpo["teacher_refusal_rate"]
    assert selection["submitted_model_passes_90_percent_gate"] is True
    assert "最终模型选择" in readme
    assert "候选通过预设纠偏门槛后才晋级" in readme
    week4_readme = WEEK4_README.read_text(encoding="utf-8")
    assert "Final_Model_Selection.json" in week4_readme
    assert "40-step 最终曲线" in week4_readme


def test_teacher_submission_excludes_private_or_operational_artifacts():
    names = {path.name.lower() for path in SUBMISSION.rglob("*") if path.is_file()}
    assert not any("mapping" in name or "shutdown" in name for name in names)
    assert not any(name.endswith((".safetensors", ".bin", ".pt", ".pth")) for name in names)
