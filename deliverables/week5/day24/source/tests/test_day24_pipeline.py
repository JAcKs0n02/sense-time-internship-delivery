from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


DAY24_ROOT = Path(__file__).resolve().parents[2]
WEEK5_ROOT = DAY24_ROOT.parent
DAY22_ROOT = WEEK5_ROOT / "day22"
DAY23_ROOT = WEEK5_ROOT / "day23"


def load_script(name: str):
    path = DAY24_ROOT / "source" / "scripts" / f"{name}.py"
    if not path.is_file():
        raise AssertionError(f"implementation missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_preregisters_four_frozen_day23_cases(tmp_path: Path) -> None:
    builder = load_script("build_day24_inputs")

    result = builder.build_artifacts(
        day23_results=DAY23_ROOT / "source" / "results" / "raw_inference_results.jsonl",
        day23_generation_config=DAY23_ROOT / "configs" / "generation_config.json",
        day22_manifest=DAY22_ROOT / "source" / "manifests" / "image_manifest.csv",
        output_root=tmp_path,
    )

    payload = json.loads((tmp_path / "configs" / "attention_cases.json").read_text())
    cases = payload["cases"]
    assert result["case_count"] == 4
    assert payload["layer_number_1_based"] == 20
    assert [case["case_id"] for case in cases] == [
        "case-01-scene-mountain",
        "case-02-table-description",
        "case-03-formula-gamma",
        "case-04-logo-triangle",
    ]
    assert [case["record_id"] for case in cases] == [
        "d23-scene-description",
        "d23-table-ocr",
        "d23-formula-description",
        "d23-logo-structure",
    ]
    assert [case["target_text"] for case in cases] == [
        "山峰",
        "DESCRIPTION",
        "gamma",
        "三角形",
    ]
    assert all(case["target_text"] in case["expected_day23_response"] for case in cases)
    assert all(len(case["image_sha256"]) == 64 for case in cases)
    assert all(len(case["expected_day23_response_sha256"]) == 64 for case in cases)

    visual = json.loads((tmp_path / "configs" / "visualization_config.json").read_text())
    assert visual == {
        "schema_version": "1.0",
        "normalization": "linear_minmax_per_case",
        "colormap": "turbo",
        "overlay_alpha": 0.45,
        "interpolation": "bilinear",
        "figure_width_inches": 12,
        "figure_height_inches": 4,
        "figure_dpi": 180,
    }


def test_linear_minmax_normalization_is_fixed_and_handles_constant_maps() -> None:
    renderer = load_script("render_attention_heatmaps")

    normalized = renderer.normalize_heatmap(
        np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        normalized,
        np.array([[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        renderer.normalize_heatmap(np.ones((2, 2), dtype=np.float32)),
        np.zeros((2, 2), dtype=np.float32),
    )


def test_renderer_writes_a_fixed_size_original_heatmap_overlay_triptych(
    tmp_path: Path,
) -> None:
    renderer = load_script("render_attention_heatmaps")
    image_path = tmp_path / "source.png"
    Image.new("RGB", (8, 4), color=(100, 120, 140)).save(image_path)
    output_path = tmp_path / "triptych.png"

    result = renderer.render_triptych(
        image_path=image_path,
        heatmap=np.array([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32),
        output_path=output_path,
        title="case-01 | target=mountain | layer=20 (index=19)",
        config={
            "normalization": "linear_minmax_per_case",
            "colormap": "turbo",
            "overlay_alpha": 0.45,
            "interpolation": "bilinear",
            "figure_width_inches": 9,
            "figure_height_inches": 3,
            "figure_dpi": 100,
        },
    )

    assert result["width"] == 900
    assert result["height"] == 300
    assert result["sha256"]
    with Image.open(output_path) as rendered:
        assert rendered.size == (900, 300)
        assert rendered.mode in {"RGB", "RGBA"}


def test_render_all_adds_traceable_heatmap_metadata(tmp_path: Path) -> None:
    renderer = load_script("render_attention_heatmaps")
    day24_root = tmp_path / "day24"
    data_root = tmp_path / "data"
    results_root = day24_root / "source" / "results"
    image_path = data_root / "images" / "source.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 4), color=(100, 120, 140)).save(image_path)
    raw_path = results_root / "raw_attention" / "case-01-mean.npy"
    raw_path.parent.mkdir(parents=True)
    np.save(raw_path, np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    metadata = {
        "case_id": "case-01",
        "status": "PASS",
        "image_relative_path": "images/source.png",
        "image_sha256": renderer.sha256_file(image_path),
        "target_display_label": "mountain",
        "layer_number_1_based": 20,
        "layer_index_0_based": 19,
        "head_mean_shape": [2, 2],
        "array_paths": {"head_mean": "raw_attention/case-01-mean.npy"},
        "array_sha256": {"head_mean": renderer.sha256_file(raw_path)},
    }
    metadata_path = results_root / "attention_metadata.jsonl"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    extraction_metadata_sha = renderer.sha256_file(metadata_path)
    (results_root / "run_summary.json").write_text(
        json.dumps({"status": "PASS", "metadata_sha256": extraction_metadata_sha}),
        encoding="utf-8",
    )
    config = {
        "schema_version": "1.0",
        "normalization": "linear_minmax_per_case",
        "colormap": "turbo",
        "overlay_alpha": 0.45,
        "interpolation": "bilinear",
        "figure_width_inches": 9,
        "figure_height_inches": 3,
        "figure_dpi": 100,
    }
    config_path = day24_root / "configs" / "visualization_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = renderer.render_all(
        data_root=data_root,
        results_root=results_root,
        visualization_config_path=config_path,
    )

    assert result["status"] == "PASS"
    assert result["rendered_case_count"] == 1
    updated = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert updated["heatmap_path"] == "heatmaps/case-01-triptych.png"
    assert updated["heatmap_sha256"] == renderer.sha256_file(
        results_root / updated["heatmap_path"]
    )
    run_summary = json.loads((results_root / "run_summary.json").read_text())
    current_metadata_sha = renderer.sha256_file(metadata_path)
    assert run_summary["metadata_sha256_at_extraction"] == extraction_metadata_sha
    assert run_summary["metadata_sha256"] == current_metadata_sha
    assert run_summary["metadata_sha256_after_render"] == current_metadata_sha


def test_validator_accepts_three_traceable_cases_and_rejects_corrupt_arrays(
    tmp_path: Path,
) -> None:
    validator = load_script("validate_day24")
    day24_root = tmp_path / "day24"
    data_root = tmp_path / "data"
    results_root = day24_root / "source" / "results"
    config_root = day24_root / "configs"
    config_root.mkdir(parents=True)
    generation_path = tmp_path / "generation_config.json"
    generation_path.write_text(
        json.dumps({"repository": "Qwen/Qwen2-VL-7B-Instruct", "revision": "fixed-rev"}),
        encoding="utf-8",
    )
    generation_sha = hashlib.sha256(generation_path.read_bytes()).hexdigest()
    case_rows = []
    metadata_rows = []
    for index in range(1, 4):
        case_id = f"case-{index:02d}"
        image_relative = f"images/image-{index}.png"
        image_path = data_root / image_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 4), color=(index * 20, 100, 120)).save(image_path)
        token_heads = np.full((1, 2, 2, 2), 0.1 * index, dtype=np.float32)
        aggregated_heads = token_heads.mean(axis=0)
        head_mean = aggregated_heads.mean(axis=0)
        raw_root = results_root / "raw_attention"
        raw_root.mkdir(parents=True, exist_ok=True)
        token_path = raw_root / f"{case_id}-token-heads.npy"
        heads_path = raw_root / f"{case_id}-heads.npy"
        mean_path = raw_root / f"{case_id}-mean.npy"
        np.save(token_path, token_heads)
        np.save(heads_path, aggregated_heads)
        np.save(mean_path, head_mean)
        heatmap_path = results_root / "heatmaps" / f"{case_id}-triptych.png"
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (900, 300), color=(240, 240, 240)).save(heatmap_path)
        image_sha = validator.sha256_file(image_path)
        case_rows.append(
            {
                "case_id": case_id,
                "image_id": f"image-{index}",
                "image_relative_path": image_relative,
                "image_sha256": image_sha,
                "target_text": f"target-{index}",
                "expected_day23_response": f"recorded response target-{index}",
                "expected_day23_response_sha256": hashlib.sha256(
                    f"recorded response target-{index}".encode()
                ).hexdigest(),
            }
        )
        metadata_rows.append(
            {
                "case_id": case_id,
                "status": "PASS",
                "image_id": f"image-{index}",
                "image_relative_path": image_relative,
                "image_sha256": image_sha,
                "target_text": f"target-{index}",
                "response_source": "day23_frozen_generation",
                "source_generation_replayed": False,
                "actual_response": f"recorded response target-{index}",
                "actual_response_sha256": hashlib.sha256(
                    f"recorded response target-{index}".encode()
                ).hexdigest(),
                "response_matches_day23": True,
                "target_token_ids": [100 + index],
                "prediction_query_positions": [20 + index],
                "target_absolute_start": 21 + index,
                "target_absolute_end": 22 + index,
                "model_revision": "fixed-rev",
                "attn_implementation": "eager",
                "layer_number_1_based": 20,
                "layer_index_0_based": 19,
                "attention_module_path": "model.model.layers.19.self_attn",
                "visual_token_start": 2,
                "visual_token_end": 6,
                "visual_token_count": 4,
                "image_grid_thw": [1, 4, 4],
                "spatial_merge_size": 2,
                "merged_grid_hw": [2, 2],
                "token_head_shape": [1, 2, 2, 2],
                "aggregated_head_shape": [2, 2, 2],
                "head_mean_shape": [2, 2],
                "attention_row_sum_min": 1.0,
                "attention_row_sum_max": 1.0,
                "array_paths": {
                    "token_heads": str(token_path.relative_to(results_root)),
                    "aggregated_heads": str(heads_path.relative_to(results_root)),
                    "head_mean": str(mean_path.relative_to(results_root)),
                },
                "array_sha256": {
                    "token_heads": validator.sha256_file(token_path),
                    "aggregated_heads": validator.sha256_file(heads_path),
                    "head_mean": validator.sha256_file(mean_path),
                },
                "heatmap_path": str(heatmap_path.relative_to(results_root)),
                "heatmap_sha256": validator.sha256_file(heatmap_path),
                "heatmap_width": 900,
                "heatmap_height": 300,
            }
        )
    cases_path = config_root / "attention_cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "repository": "Qwen/Qwen2-VL-7B-Instruct",
                "revision": "fixed-rev",
                "generation_config_sha256": generation_sha,
                "layer_number_1_based": 20,
                "cases": case_rows,
            }
        ),
        encoding="utf-8",
    )
    visual_path = config_root / "visualization_config.json"
    visual_path.write_text(json.dumps({"normalization": "linear_minmax_per_case"}), encoding="utf-8")
    metadata_path = results_root / "attention_metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata_rows), encoding="utf-8"
    )
    metadata_sha = validator.sha256_file(metadata_path)
    run_summary_payload = {
        "status": "PASS",
        "case_count": 3,
        "pass_count": 3,
        "fail_count": 0,
        "metadata_sha256": metadata_sha,
        "metadata_sha256_after_render": metadata_sha,
    }
    (results_root / "run_summary.json").write_text(
        json.dumps(run_summary_payload),
        encoding="utf-8",
    )
    (results_root / "render_summary.json").write_text(
        json.dumps(
            {"status": "PASS", "rendered_case_count": 3, "metadata_sha256": metadata_sha}
        ),
        encoding="utf-8",
    )

    passed = validator.validate_artifacts(
        cases_path=cases_path,
        generation_config_path=generation_path,
        visualization_config_path=visual_path,
        data_root=data_root,
        results_root=results_root,
        minimum_cases=3,
    )
    assert passed["status"] == "PASS"
    assert all(check["status"] == "PASS" for check in passed["checks"])

    broken_summary = dict(run_summary_payload)
    broken_summary["metadata_sha256"] = "0" * 64
    (results_root / "run_summary.json").write_text(
        json.dumps(broken_summary), encoding="utf-8"
    )
    stale_summary = validator.validate_artifacts(
        cases_path=cases_path,
        generation_config_path=generation_path,
        visualization_config_path=visual_path,
        data_root=data_root,
        results_root=results_root,
        minimum_cases=3,
    )
    assert stale_summary["status"] == "FAIL"
    assert any("current metadata hash" in error for error in stale_summary["errors"])
    (results_root / "run_summary.json").write_text(
        json.dumps(run_summary_payload), encoding="utf-8"
    )

    drifted_rows = [dict(row) for row in metadata_rows]
    drifted_rows[0]["response_source"] = "eager_regeneration"
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in drifted_rows), encoding="utf-8"
    )
    drifted = validator.validate_artifacts(
        cases_path=cases_path,
        generation_config_path=generation_path,
        visualization_config_path=visual_path,
        data_root=data_root,
        results_root=results_root,
        minimum_cases=3,
    )
    assert drifted["status"] == "FAIL"
    assert any("frozen Day23" in error for error in drifted["errors"])

    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata_rows), encoding="utf-8"
    )

    np.save(results_root / metadata_rows[0]["array_paths"]["head_mean"], np.zeros((2, 2)))
    failed = validator.validate_artifacts(
        cases_path=cases_path,
        generation_config_path=generation_path,
        visualization_config_path=visual_path,
        data_root=data_root,
        results_root=results_root,
        minimum_cases=3,
    )
    assert failed["status"] == "FAIL"
    assert any("hash" in error for error in failed["errors"])
