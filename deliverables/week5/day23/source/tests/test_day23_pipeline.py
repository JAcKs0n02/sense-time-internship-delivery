from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


DAY23_ROOT = Path(__file__).resolve().parents[2]
DAY22_ROOT = DAY23_ROOT.parent / "day22"


def load_script(name: str):
    path = DAY23_ROOT / "source" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_inputs_emits_the_complete_frozen_5_by_5_matrix(tmp_path: Path) -> None:
    builder = load_script("build_day23_inputs")
    result = builder.build_artifacts(
        manifest_path=DAY22_ROOT / "source" / "manifests" / "image_manifest.csv",
        output_root=tmp_path,
    )

    matrix = json.loads((tmp_path / "source" / "data" / "inference_matrix.json").read_text())
    rows = matrix["records"]
    assert result["record_count"] == 25
    assert len(rows) == 25
    assert len({row["record_id"] for row in rows}) == 25
    assert len({(row["image_id"], row["question_type"]) for row in rows}) == 25
    assert rows[0]["record_id"] == "d23-table-description"
    assert rows[-1]["record_id"] == "d23-ui-implicit"
    assert [row["execution_index"] for row in rows] == list(range(1, 26))
    assert {row["generation_config_sha256"] for row in rows} == {
        result["generation_config_sha256"]
    }

    config = json.loads((tmp_path / "configs" / "generation_config.json").read_text())
    assert config["do_sample"] is False
    assert config["seed"] == 42
    assert config["max_new_tokens"] == 1024
    assert config["processor_size"] == {
        "shortest_edge": 200704,
        "longest_edge": 301056,
    }


def test_build_inputs_fails_closed_if_manifest_order_or_count_changes(tmp_path: Path) -> None:
    builder = load_script("build_day23_inputs")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_id,relative_path,image_type,source,license,width,height,file_bytes,sha256\n"
        "w5-scene-01,w5-scene-01.jpg,natural_scene,x,CC0,1,1,1,"
        + "a" * 64
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly five frozen image IDs"):
        builder.build_artifacts(manifest_path=manifest, output_root=tmp_path / "out")


def test_runner_validates_matrix_and_effective_processor_budget() -> None:
    runner = load_script("run_day23_inference")
    assert runner.build_processor_size(200704, 301056) == {
        "shortest_edge": 200704,
        "longest_edge": 301056,
    }
    with pytest.raises(ValueError, match="pixel budget"):
        runner.build_processor_size(10, 9)

    records = [
        {
            "record_id": f"r-{index}",
            "execution_index": index,
            "image_id": f"image-{index}",
            "question_type": "description",
            "prompt": "x",
            "image_relative_path": "x.png",
            "image_sha256": "a" * 64,
            "generation_config_sha256": "b" * 64,
        }
        for index in range(1, 26)
    ]
    runner.validate_matrix(records)
    records[1]["record_id"] = records[0]["record_id"]
    with pytest.raises(ValueError, match="record_id values must be unique"):
        runner.validate_matrix(records)


def test_runner_detects_the_configured_token_ceiling() -> None:
    runner = load_script("run_day23_inference")
    assert runner.reached_token_ceiling(1024, 1024) is True
    assert runner.reached_token_ceiling(526, 1024) is False


def test_capability_summary_counts_strength_and_hallucination_labels() -> None:
    summarizer = load_script("summarize_capabilities")
    rows = [
        {
            "image_id": "w5-table-01",
            "image_type": "table_screenshot",
            "question_type": "ocr",
            "strength_label": "strong",
            "hallucination_label": "none",
        },
        {
            "image_id": "w5-table-01",
            "image_type": "table_screenshot",
            "question_type": "implicit_reasoning",
            "strength_label": "weak",
            "hallucination_label": "severe",
        },
        {
            "image_id": "w5-scene-01",
            "image_type": "natural_scene",
            "question_type": "ocr",
            "strength_label": "acceptable",
            "hallucination_label": "minor",
        },
    ]

    summary = summarizer.aggregate(rows)
    assert summary["overall"] == {
        "record_count": 3,
        "strength_counts": {"strong": 1, "acceptable": 1, "weak": 1},
        "hallucination_counts": {"none": 1, "minor": 1, "severe": 1},
    }
    assert summary["by_question_type"]["ocr"]["record_count"] == 2
    assert summary["by_image_type"]["table_screenshot"]["severe_hallucinations"] == 1


def test_review_export_joins_by_id_without_changing_raw_response(tmp_path: Path) -> None:
    exporter = load_script("apply_day23_reviews")
    raw_rows = [
        {
            "record_id": "r-1",
            "execution_index": 1,
            "raw_response": "first raw answer",
        },
        {
            "record_id": "r-2",
            "execution_index": 2,
            "raw_response": "second raw answer",
        },
    ]
    reviews = {
        "reviewer_type": "codex_assisted_visual_review",
        "reviews": [
            {
                "record_id": "r-2",
                "strength_label": "acceptable",
                "hallucination_label": "minor",
                "reviewer_reason": "minor unsupported detail",
            },
            {
                "record_id": "r-1",
                "strength_label": "strong",
                "hallucination_label": "none",
                "reviewer_reason": "matches ground truth",
            },
        ],
    }

    joined = exporter.join_reviews(raw_rows, reviews)
    assert [row["record_id"] for row in joined] == ["r-1", "r-2"]
    assert [row["raw_response"] for row in joined] == [
        "first raw answer",
        "second raw answer",
    ]
    assert joined[0]["reviewer_type"] == "codex_assisted_visual_review"
    output = tmp_path / "reviewed.csv"
    exporter.write_csv(output, joined, exporter.FULL_FIELDS)
    assert b"\r\n" not in output.read_bytes()
    reviews["reviews"].pop()
    with pytest.raises(ValueError, match="exactly once"):
        exporter.join_reviews(raw_rows, reviews)


def test_validator_accepts_only_a_complete_reviewed_first_run(tmp_path: Path) -> None:
    validator = load_script("validate_day23")
    builder = load_script("build_day23_inputs")
    build_result = builder.build_artifacts(
        manifest_path=DAY22_ROOT / "source" / "manifests" / "image_manifest.csv",
        output_root=tmp_path,
    )
    matrix = json.loads((tmp_path / "source" / "data" / "inference_matrix.json").read_text())
    raw_rows = []
    reviewed_rows = []
    for item in matrix["records"]:
        raw = {
            **item,
            "status": "PASS",
            "raw_response": "非空原始回答",
            "input_tokens": 100,
            "output_tokens": 10,
            "max_new_tokens": 1024,
            "latency_seconds": 1.0,
            "peak_gpu_memory_mib": 18000.0,
            "model_revision": builder.MODEL_REVISION,
            "generation_config_sha256": build_result["generation_config_sha256"],
        }
        raw_rows.append(raw)
        reviewed_rows.append(
            {
                **raw,
                "image_type": item["image_type"],
                "strength_label": "strong",
                "hallucination_label": "none",
                "reviewer_reason": "与冻结 ground truth 一致。",
                "reviewer_type": "codex_assisted_visual_review",
            }
        )

    result = validator.validate_records(
        matrix_records=matrix["records"],
        raw_records=raw_rows,
        reviewed_records=reviewed_rows,
        expected_config_sha256=build_result["generation_config_sha256"],
        expected_model_revision=builder.MODEL_REVISION,
    )
    assert result["status"] == "PASS"
    assert all(check["status"] == "PASS" for check in result["checks"])

    reviewed_rows[0]["output_tokens"] = 1024
    raw_rows[0]["output_tokens"] = 1024
    truncated = validator.validate_records(
        matrix_records=matrix["records"],
        raw_records=raw_rows,
        reviewed_records=reviewed_rows,
        expected_config_sha256=build_result["generation_config_sha256"],
        expected_model_revision=builder.MODEL_REVISION,
    )
    assert truncated["status"] == "FAIL"
    assert any("token limit" in error for error in truncated["errors"])
    reviewed_rows[0]["output_tokens"] = 10
    raw_rows[0]["output_tokens"] = 10

    reviewed_rows[0]["raw_response"] = "被改写的回答"
    failed = validator.validate_records(
        matrix_records=matrix["records"],
        raw_records=raw_rows,
        reviewed_records=reviewed_rows,
        expected_config_sha256=build_result["generation_config_sha256"],
        expected_model_revision=builder.MODEL_REVISION,
    )
    assert failed["status"] == "FAIL"
    assert any("raw_response changed" in error for error in failed["errors"])
