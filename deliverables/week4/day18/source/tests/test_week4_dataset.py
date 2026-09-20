from __future__ import annotations

import importlib.util
import json
import csv
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


DAY18_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = DAY18_ROOT / "source"
SCRIPT_PATH = SOURCE_ROOT / "scripts" / "build_week4_dataset.py"
ULTRAFEEDBACK_PATH = SOURCE_ROOT / "data" / "interim" / "ultrafeedback_500.json"
SELF_BUILT_PATH = SOURCE_ROOT / "data" / "raw" / "self_built_preferences.json"
HOLDOUT_PATH = SOURCE_ROOT / "data" / "raw" / "evaluation_holdout_prompts.json"


def load_builder():
    assert SCRIPT_PATH.is_file(), f"dataset builder is missing: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("build_week4_dataset", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_record(record_id: str, prompt: str, source_kind: str = "self_built") -> dict:
    record = {
        "id": record_id,
        "conversations": [{"from": "human", "value": prompt}],
        "chosen": {"from": "gpt", "value": "更准确且满足要求的回答。"},
        "rejected": {"from": "gpt", "value": "相关但遗漏关键要求的回答。"},
    }
    if source_kind == "ultrafeedback":
        record["metadata"] = {
            "source_kind": "ultrafeedback",
            "preference_type": "mixed_external",
            "upstream_source": "fixture",
            "source_id": record_id,
        }
    else:
        record.update(
            {
                "preference_type": "factuality",
                "scene_kind": "business",
                "construction_reason": "chosen 给出条件和证据，rejected 遗漏关键边界。",
                "source": {"type": "self_built", "source_id": record_id},
                "quality_review": {
                    "single_dimension": True,
                    "safe_to_publish": True,
                    "status": "approved",
                    "reviewer_kind": "codex_review",
                },
            }
        )
    return record


def test_normalize_prompt_handles_unicode_whitespace_and_punctuation():
    builder = load_builder()

    assert builder.normalize_prompt(" ＡＰＩ， 版本\n控制！ ") == "api版本控制"


def test_validate_preference_record_rejects_outer_whitespace_fake_difference():
    builder = load_builder()
    record = sample_record("sb-001", "怎样验证备份？")
    record["rejected"]["value"] = f"  {record['chosen']['value']}\n"

    errors = builder.validate_preference_record(record)

    assert errors == ["responses_not_distinct"]


def test_stratified_split_is_exact_stable_and_disjoint():
    builder = load_builder()
    records = [
        sample_record(f"uf-{index:02d}", f"开源问题 {index}", "ultrafeedback")
        for index in range(10)
    ] + [
        sample_record(f"sb-{index:02d}", f"自建问题 {index}")
        for index in range(10)
    ]

    first_train, first_validation = builder.stratified_split(
        records,
        seed=42,
        validation_ratio=0.1,
    )
    second_train, second_validation = builder.stratified_split(
        list(reversed(records)),
        seed=42,
        validation_ratio=0.1,
    )

    assert len(first_train) == 18
    assert len(first_validation) == 2
    assert [record["id"] for record in first_train] == [
        record["id"] for record in second_train
    ]
    assert [record["id"] for record in first_validation] == [
        record["id"] for record in second_validation
    ]
    assert {record["id"] for record in first_train}.isdisjoint(
        {record["id"] for record in first_validation}
    )


def test_duplicate_and_holdout_contamination_are_blocking():
    builder = load_builder()
    first = sample_record("sb-001", "怎样验证备份？")
    duplicate = sample_record("sb-002", " 怎样验证备份！ ")
    records = [first, duplicate]
    holdout = {"prompts": [{"id": "eval-01", "text": "怎样验证备份？"}]}

    duplicate_groups = builder.find_exact_duplicate_groups(records)
    contamination = builder.find_eval_contamination(records, holdout)

    assert duplicate_groups == [["sb-001", "sb-002"]]
    assert contamination == [
        {"record_id": "sb-001", "eval_id": "eval-01", "match_type": "exact"},
        {"record_id": "sb-002", "eval_id": "eval-01", "match_type": "exact"},
    ]


def test_near_duplicate_audit_flags_small_prompt_rewrites_without_auto_deleting():
    builder = load_builder()
    records = [
        sample_record("sb-001", "请说明数据库备份恢复验证步骤"),
        sample_record("sb-002", "请说明数据库备份的恢复验证步骤"),
        sample_record("sb-003", "如何规划一次周末徒步？"),
    ]

    candidates = builder.find_near_duplicate_candidates(records, threshold=0.8)

    assert [(item["left_id"], item["right_id"]) for item in candidates] == [
        ("sb-001", "sb-002")
    ]
    assert candidates[0]["decision"] == "review_required"
    assert 0.8 <= candidates[0]["similarity"] < 1.0


def test_near_holdout_contamination_is_detected() -> None:
    builder = load_builder()
    records = [sample_record("sb-001", "请说明数据库备份恢复验证步骤")]
    holdout = {
        "prompts": [
            {"id": "eval-01", "text": "请说明数据库备份的恢复验证步骤"},
            {"id": "eval-02", "text": "如何规划一次周末徒步？"},
        ]
    }

    assert builder.find_near_eval_contamination(records, holdout, threshold=0.8) == [
        {
            "record_id": "sb-001",
            "eval_id": "eval-01",
            "match_type": "high_similarity",
            "similarity": 0.965517,
        }
    ]


def test_repository_inputs_build_the_frozen_710_record_dataset():
    builder = load_builder()
    ultrafeedback = json.loads(ULTRAFEEDBACK_PATH.read_text(encoding="utf-8"))
    self_built = json.loads(SELF_BUILT_PATH.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))

    result = builder.build_dataset(ultrafeedback, self_built, holdout, seed=42)

    assert result["validation"] == {
        "valid": True,
        "errors": [],
        "total_count": 710,
        "ultrafeedback_count": 500,
        "self_built_count": 210,
        "train_count": 639,
        "validation_count": 71,
        "exact_duplicate_group_count": 0,
        "eval_contamination_count": 0,
        "eval_near_contamination_count": 0,
    }
    assert len(result["full"]) == 710
    assert len(result["train"]) == 639
    assert len(result["validation_records"]) == 71
    assert "review_required" not in {
        item["decision"] for item in result["near_duplicate_candidates"]
    }
    assert {
        item["decision"] for item in result["near_duplicate_candidates"]
    } <= {"retained_distinct_topic", "retained_distinct_preference_dimension"}


def test_repository_build_fails_if_an_eval_prompt_is_injected():
    builder = load_builder()
    ultrafeedback = json.loads(ULTRAFEEDBACK_PATH.read_text(encoding="utf-8"))
    self_built = json.loads(SELF_BUILT_PATH.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    broken = deepcopy(self_built)
    broken[0]["conversations"][-1]["value"] = holdout["prompts"][0]["text"]

    result = builder.build_dataset(ultrafeedback, broken, holdout, seed=42)

    assert result["validation"]["valid"] is False
    assert "eval_contamination" in result["validation"]["errors"]


def test_cli_writes_complete_deterministic_day18_artifacts(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    expected_files = {
        "week4_preferences_full.json",
        "week4_dpo_train.json",
        "week4_dpo_validation.json",
        "preference_stats.json",
        "preference_split.csv",
        "duplicate_audit.json",
        "day18_validation.json",
        "evaluation_holdout_manifest.json",
        "dataset_info_week4.json",
    }

    for output_dir in (first, second):
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--ultrafeedback",
                str(ULTRAFEEDBACK_PATH),
                "--self-built",
                str(SELF_BUILT_PATH),
                "--holdout",
                str(HOLDOUT_PATH),
                "--output-dir",
                str(output_dir),
                "--seed",
                "42",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert {path.name for path in output_dir.iterdir()} == expected_files

    for name in expected_files:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    validation = json.loads((first / "day18_validation.json").read_text(encoding="utf-8"))
    assert validation["valid"] is True
    assert b"\r\n" not in (first / "preference_split.csv").read_bytes()
    with (first / "preference_split.csv").open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 710
