from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


DAY25_ROOT = Path(__file__).resolve().parents[2]
WEEK5_ROOT = DAY25_ROOT.parent
REPO_ROOT = DAY25_ROOT.parents[2]
SCRIPTS = DAY25_ROOT / "source" / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    assert path.is_file(), f"missing production script: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upstream_paths() -> tuple[Path, Path, Path]:
    manifest = WEEK5_ROOT / "day22" / "source" / "manifests" / "image_manifest.csv"
    ground_truth = WEEK5_ROOT / "day22" / "source" / "manifests" / "image_ground_truth.json"
    config = WEEK5_ROOT / "day23" / "configs" / "generation_config.json"
    return manifest, ground_truth, config


def test_builder_freezes_exactly_ten_cases_with_two_per_image(tmp_path: Path) -> None:
    builder = load_script("build_day25_cases")
    manifest, ground_truth, config = upstream_paths()

    result = builder.build_artifacts(manifest, ground_truth, config, tmp_path)
    payload = json.loads((tmp_path / "source/data/hallucination_cases.json").read_text())
    records = payload["records"]

    assert result["record_count"] == 10
    assert payload["status"] == "FROZEN_BEFORE_INFERENCE"
    assert [row["execution_index"] for row in records] == list(range(1, 11))
    assert len({row["case_id"] for row in records}) == 10
    assert Counter(row["image_id"] for row in records) == {
        "w5-table-01": 2,
        "w5-scene-01": 2,
        "w5-logo-01": 2,
        "w5-formula-01": 2,
        "w5-ui-01": 2,
    }


def test_builder_covers_all_preregistered_error_categories(tmp_path: Path) -> None:
    builder = load_script("build_day25_cases")
    manifest, ground_truth, config = upstream_paths()
    builder.build_artifacts(manifest, ground_truth, config, tmp_path)
    records = json.loads(
        (tmp_path / "source/data/hallucination_cases.json").read_text()
    )["records"]

    assert {row["error_category"] for row in records} == {
        "nonexistent_object",
        "wrong_text_or_number",
        "wrong_attribute",
        "wrong_spatial_relation",
        "wrong_action_or_intent",
    }
    for row in records:
        assert row["question"]
        assert row["fake_answer"]
        assert row["ground_truth"]
        assert row["expected_behavior"]
        assert row["fake_answer"] in row["prompt"]
        assert row["question"] in row["prompt"]
        assert "假答案" not in row["prompt"]


def test_builder_preserves_upstream_config_and_image_hashes(tmp_path: Path) -> None:
    builder = load_script("build_day25_cases")
    manifest, ground_truth, config = upstream_paths()
    result = builder.build_artifacts(manifest, ground_truth, config, tmp_path)
    copied_config = tmp_path / "configs/generation_config.json"
    cases_path = tmp_path / "source/data/hallucination_cases.json"
    freeze_path = tmp_path / "source/data/test_case_manifest.json"
    payload = json.loads(cases_path.read_text())
    freeze = json.loads(freeze_path.read_text())

    assert copied_config.read_bytes() == config.read_bytes()
    assert {row["generation_config_sha256"] for row in payload["records"]} == {
        sha256(config)
    }
    assert result["cases_sha256"] == sha256(cases_path)
    assert freeze["cases_sha256"] == sha256(cases_path)
    assert freeze["generation_config_sha256"] == sha256(config)
    assert freeze["record_count"] == 10


def test_runner_builds_one_image_and_one_frozen_prompt_message(tmp_path: Path) -> None:
    runner = load_script("run_hallucination_test")
    image = tmp_path / "example image.png"
    image.write_bytes(b"placeholder")

    messages = runner.build_messages(image, "冻结后的提示")

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.resolve().as_uri()},
                {"type": "text", "text": "冻结后的提示"},
            ],
        }
    ]


def test_runner_rejects_any_case_set_that_is_not_the_frozen_ten() -> None:
    runner = load_script("run_hallucination_test")
    records = [
        {
            "case_id": f"D25-{index:02d}",
            "execution_index": index,
            "image_id": image_id,
            "prompt": "x",
            "question": "q",
            "fake_answer": "f",
            "ground_truth": "g",
            "expected_behavior": "e",
            "error_category": "nonexistent_object",
            "image_relative_path": "images/x.png",
            "image_sha256": "a" * 64,
            "generation_config_sha256": "b" * 64,
            "model_revision": "r",
        }
        for index, image_id in enumerate(
            [
                "w5-table-01",
                "w5-table-01",
                "w5-scene-01",
                "w5-scene-01",
                "w5-logo-01",
                "w5-logo-01",
                "w5-formula-01",
                "w5-formula-01",
                "w5-ui-01",
                "w5-ui-01",
            ],
            start=1,
        )
    ]
    runner.validate_formal_cases(records)

    with pytest.raises(ValueError, match="ten"):
        runner.validate_formal_cases(records[:-1])


def sample_records() -> tuple[list[dict], list[dict], list[dict]]:
    cases = [
        {
            "case_id": "D25-01",
            "execution_index": 1,
            "image_id": "w5-scene-01",
            "error_category": "nonexistent_object",
            "ground_truth": "图中没有汽车。",
            "fake_answer": "汽车在湖边。",
        },
        {
            "case_id": "D25-02",
            "execution_index": 2,
            "image_id": "w5-logo-01",
            "error_category": "wrong_text_or_number",
            "ground_truth": "图中没有文字。",
            "fake_answer": "中央写着WIKI。",
        },
    ]
    raw = [
        {
            **cases[0],
            "status": "PASS",
            "raw_response": "是的，红色汽车停在湖边。",
            "output_tokens": 12,
            "latency_seconds": 1.2,
            "peak_gpu_memory_mib": 10.0,
        },
        {
            **cases[1],
            "status": "PASS",
            "raw_response": "图片中没有可见文字，因此不存在WIKI字样。",
            "output_tokens": 15,
            "latency_seconds": 1.3,
            "peak_gpu_memory_mib": 10.0,
        },
    ]
    annotations = [
        {
            "case_id": "D25-01",
            "hallucination": 1,
            "task_success": 0,
            "response_type": "accepted_false_premise",
            "hallucination_basis": "accepted_false_premise",
            "evidence_excerpt": "红色汽车停在湖边",
            "reviewer_reason": "模型确认了图中不存在的汽车。",
            "reviewer_type": "codex_assisted_visual_review",
        },
        {
            "case_id": "D25-02",
            "hallucination": 0,
            "task_success": 1,
            "response_type": "corrected_false_premise",
            "hallucination_basis": "none",
            "evidence_excerpt": "没有可见文字",
            "reviewer_reason": "模型明确纠正了不存在的WIKI文字。",
            "reviewer_type": "codex_assisted_visual_review",
        },
    ]
    return cases, raw, annotations


def test_scoring_uses_fixed_denominator_and_binary_labels() -> None:
    scorer = load_script("score_hallucination")
    cases, raw, annotations = sample_records()

    reviewed, summary = scorer.score_records(
        cases, raw, annotations, expected_count=2
    )

    assert [row["hallucination"] for row in reviewed] == [1, 0]
    assert summary["sample_count"] == 2
    assert summary["hallucination_count"] == 1
    assert summary["hallucination_rate"] == 0.5
    assert summary["hallucination_percentage"] == 50.0
    assert summary["correction_count"] == 1


def test_scoring_rejects_missing_or_nonbinary_annotations() -> None:
    scorer = load_script("score_hallucination")
    cases, raw, annotations = sample_records()
    annotations[0]["hallucination"] = 0.5

    with pytest.raises(ValueError, match="binary"):
        scorer.score_records(cases, raw, annotations, expected_count=2)

    with pytest.raises(ValueError, match="order"):
        scorer.score_records(cases, raw, annotations[:1], expected_count=2)


def test_validation_fails_closed_when_summary_formula_is_wrong(tmp_path: Path) -> None:
    validator = load_script("validate_day25")
    cases, raw, annotations = sample_records()
    scorer = load_script("score_hallucination")
    reviewed, summary = scorer.score_records(cases, raw, annotations, expected_count=2)
    summary["hallucination_rate"] = 0.0

    result = validator.validate_records(
        cases=cases,
        raw_records=raw,
        reviewed_records=reviewed,
        summary=summary,
        expected_count=2,
    )

    assert result["status"] == "FAIL"
    assert any("hallucination_rate" in error for error in result["errors"])


def test_csv_writer_preserves_all_teacher_facing_fields(tmp_path: Path) -> None:
    scorer = load_script("score_hallucination")
    cases, raw, annotations = sample_records()
    reviewed, _ = scorer.score_records(cases, raw, annotations, expected_count=2)
    output = tmp_path / "results.csv"

    scorer.write_csv(output, reviewed)

    assert b"\r\n" not in output.read_bytes()
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["question"] == ""
    assert rows[0]["hallucination"] == "1"
    assert rows[0]["reviewer_reason"] == "模型确认了图中不存在的汽车。"
