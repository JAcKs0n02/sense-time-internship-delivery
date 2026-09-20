#!/usr/bin/env python3
"""Validate Day24 attention arrays, mappings, figures, and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_artifacts(
    *,
    cases_path: Path,
    generation_config_path: Path,
    visualization_config_path: Path,
    data_root: Path,
    results_root: Path,
    minimum_cases: int = 3,
) -> dict[str, Any]:
    cases_payload = read_json(cases_path)
    generation = read_json(generation_config_path)
    visualization = read_json(visualization_config_path)
    metadata = read_jsonl(results_root / "attention_metadata.jsonl")
    run_summary = read_json(results_root / "run_summary.json")
    render_summary = read_json(results_root / "render_summary.json")
    cases = cases_payload.get("cases")
    if not isinstance(cases, list):
        cases = []

    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def check(name: str, fn: Callable[[], str]) -> None:
        try:
            detail = fn()
            checks.append({"name": name, "status": "PASS", "detail": detail})
        except Exception as exc:
            message = f"{name}: {exc}"
            errors.append(message)
            checks.append({"name": name, "status": "FAIL", "detail": str(exc)})

    def validate_configs() -> str:
        if cases_payload.get("repository") != generation.get("repository"):
            raise ValueError("repository mismatch")
        if cases_payload.get("revision") != generation.get("revision"):
            raise ValueError("revision mismatch")
        if cases_payload.get("generation_config_sha256") != sha256_file(
            generation_config_path
        ):
            raise ValueError("generation config hash mismatch")
        if cases_payload.get("layer_number_1_based") != 20:
            raise ValueError("teacher-facing layer number is not 20")
        if visualization.get("normalization") != "linear_minmax_per_case":
            raise ValueError("visualization normalization is not the preregistered method")
        return "model/config identity and fixed normalization match"

    def validate_case_contract() -> str:
        if len(cases) < minimum_cases:
            raise ValueError(f"requires at least {minimum_cases} cases, got {len(cases)}")
        case_ids = [case.get("case_id") for case in cases]
        image_ids = [case.get("image_id") for case in cases]
        targets = [case.get("target_text") for case in cases]
        if len(set(case_ids)) != len(cases):
            raise ValueError("case IDs are not unique")
        if len(set(image_ids)) != len(cases):
            raise ValueError("formal cases do not use different images")
        if len(set(targets)) != len(cases):
            raise ValueError("formal cases do not use different target words")
        for case in cases:
            image_path = data_root / str(case["image_relative_path"])
            if not image_path.is_file() or sha256_file(image_path) != case.get("image_sha256"):
                raise ValueError(f"image hash mismatch for {case.get('case_id')}")
        return f"{len(cases)} preregistered cases use distinct images and targets"

    def validate_metadata_coverage() -> str:
        configured = {case.get("case_id") for case in cases}
        observed = {row.get("case_id") for row in metadata}
        if len(metadata) != len(cases) or observed != configured:
            raise ValueError("metadata does not cover each preregistered case exactly once")
        if any(row.get("status") != "PASS" for row in metadata):
            raise ValueError("one or more formal cases did not pass extraction")
        return f"metadata covers all {len(metadata)} cases exactly once"

    def validate_model_layer_and_hooks() -> str:
        for row in metadata:
            if row.get("model_revision") != cases_payload.get("revision"):
                raise ValueError(f"model revision mismatch for {row.get('case_id')}")
            if row.get("attn_implementation") != "eager":
                raise ValueError(f"non-eager attention for {row.get('case_id')}")
            if row.get("layer_number_1_based") != 20 or row.get("layer_index_0_based") != 19:
                raise ValueError(f"layer numbering mismatch for {row.get('case_id')}")
            if row.get("attention_module_path") != "model.model.layers.19.self_attn":
                raise ValueError(f"unexpected Hook module for {row.get('case_id')}")
        return "all cases use eager attention and Hook model.model.layers.19.self_attn"

    def validate_target_alignment() -> str:
        cases_by_id = {case.get("case_id"): case for case in cases}
        for row in metadata:
            case = cases_by_id.get(row.get("case_id"))
            if not isinstance(case, dict):
                raise ValueError(f"case contract missing for {row.get('case_id')}")
            expected_response = str(case.get("expected_day23_response", ""))
            expected_hash = str(case.get("expected_day23_response_sha256", ""))
            if (
                row.get("response_source") != "day23_frozen_generation"
                or row.get("source_generation_replayed") is not False
            ):
                raise ValueError(
                    f"attention does not use the frozen Day23 response for {row.get('case_id')}"
                )
            if (
                not expected_response
                or sha256_text(expected_response) != expected_hash
                or row.get("actual_response") != expected_response
                or row.get("actual_response_sha256") != expected_hash
                or row.get("response_matches_day23") is not True
            ):
                raise ValueError(
                    f"frozen Day23 response identity mismatch for {row.get('case_id')}"
                )
            token_ids = row.get("target_token_ids")
            queries = row.get("prediction_query_positions")
            start = int(row.get("target_absolute_start"))
            end = int(row.get("target_absolute_end"))
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError(f"target token IDs missing for {row.get('case_id')}")
            if end - start != len(token_ids):
                raise ValueError(f"target token span mismatch for {row.get('case_id')}")
            expected_queries = list(range(start - 1, end - 1))
            if queries != expected_queries:
                raise ValueError(f"prediction query offset is wrong for {row.get('case_id')}")
            if not 0.995 <= float(row.get("attention_row_sum_min")) <= 1.005:
                raise ValueError(f"attention row minimum is invalid for {row.get('case_id')}")
            if not 0.995 <= float(row.get("attention_row_sum_max")) <= 1.005:
                raise ValueError(f"attention row maximum is invalid for {row.get('case_id')}")
        return "hash-verified Day23 targets use their preceding causal prediction rows"

    def validate_visual_grids() -> str:
        for row in metadata:
            t, h, w = (int(value) for value in row["image_grid_thw"])
            merge = int(row["spatial_merge_size"])
            merged = [h // merge, w // merge]
            expected_count = t * merged[0] * merged[1]
            if t != 1 or h % merge or w % merge:
                raise ValueError(f"invalid single-image merger grid for {row.get('case_id')}")
            if row.get("merged_grid_hw") != merged:
                raise ValueError(f"merged grid mismatch for {row.get('case_id')}")
            if int(row["visual_token_end"]) - int(row["visual_token_start"]) != expected_count:
                raise ValueError(f"visual token interval mismatch for {row.get('case_id')}")
            if int(row["visual_token_count"]) != expected_count:
                raise ValueError(f"visual token count mismatch for {row.get('case_id')}")
        return "visual token intervals exactly match merger-derived 2D grids"

    def validate_arrays() -> str:
        for row in metadata:
            expected_shapes = {
                "token_heads": row["token_head_shape"],
                "aggregated_heads": row["aggregated_head_shape"],
                "head_mean": row["head_mean_shape"],
            }
            for key, shape in expected_shapes.items():
                path = results_root / str(row["array_paths"][key])
                if not path.is_file():
                    raise ValueError(f"missing {key} array for {row.get('case_id')}")
                if sha256_file(path) != row["array_sha256"][key]:
                    raise ValueError(f"{key} array hash mismatch for {row.get('case_id')}")
                value = np.load(path, allow_pickle=False)
                if list(value.shape) != list(shape):
                    raise ValueError(f"{key} array shape mismatch for {row.get('case_id')}")
                if not np.isfinite(value).all() or (value < 0).any():
                    raise ValueError(f"{key} array has invalid values for {row.get('case_id')}")
        return "all per-token, per-head, and mean arrays are finite and hash-verified"

    def validate_heatmaps() -> str:
        for row in metadata:
            path = results_root / str(row["heatmap_path"])
            if not path.is_file() or sha256_file(path) != row.get("heatmap_sha256"):
                raise ValueError(f"heatmap hash mismatch for {row.get('case_id')}")
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.size != (int(row["heatmap_width"]), int(row["heatmap_height"])):
                    raise ValueError(f"heatmap dimensions changed for {row.get('case_id')}")
        return f"{len(metadata)} traceable triptych PNG files are readable and hash-verified"

    def validate_summaries() -> str:
        current_metadata_sha = sha256_file(results_root / "attention_metadata.jsonl")
        if run_summary.get("status") != "PASS":
            raise ValueError("extraction run summary is not PASS")
        if int(run_summary.get("pass_count", -1)) != len(cases) or int(
            run_summary.get("fail_count", -1)
        ) != 0:
            raise ValueError("extraction summary counts do not match cases")
        if render_summary.get("status") != "PASS" or int(
            render_summary.get("rendered_case_count", -1)
        ) != len(cases):
            raise ValueError("render summary counts do not match cases")
        if (
            run_summary.get("metadata_sha256") != current_metadata_sha
            or run_summary.get("metadata_sha256_after_render") != current_metadata_sha
            or render_summary.get("metadata_sha256") != current_metadata_sha
        ):
            raise ValueError("current metadata hash is stale in one or more summaries")
        return "extraction and rendering summaries pass with complete counts and current hashes"

    check("configs_identity", validate_configs)
    check("case_contract", validate_case_contract)
    check("metadata_coverage", validate_metadata_coverage)
    check("model_layer_and_hooks", validate_model_layer_and_hooks)
    check("target_alignment", validate_target_alignment)
    check("visual_grid_mapping", validate_visual_grids)
    check("raw_attention_arrays", validate_arrays)
    check("heatmap_files", validate_heatmaps)
    check("run_summaries", validate_summaries)

    return {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "minimum_cases": minimum_cases,
        "configured_case_count": len(cases),
        "validated_case_count": len(metadata),
        "checks": checks,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--visualization-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--minimum-cases", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_artifacts(
        cases_path=args.cases,
        generation_config_path=args.generation_config,
        visualization_config_path=args.visualization_config,
        data_root=args.data_root,
        results_root=args.results_root,
        minimum_cases=args.minimum_cases,
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
