#!/usr/bin/env python3
"""Pre-register Day24 attention cases from the frozen Day22/Day23 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


CASE_SPECS = (
    {
        "case_id": "case-01-scene-mountain",
        "record_id": "d23-scene-description",
        "target_text": "山峰",
        "target_occurrence": 1,
        "target_display_label": "mountain_peak",
        "target_category": "grounded_object",
    },
    {
        "case_id": "case-02-table-description",
        "record_id": "d23-table-ocr",
        "target_text": "DESCRIPTION",
        "target_occurrence": 1,
        "target_display_label": "DESCRIPTION",
        "target_category": "grounded_ocr",
    },
    {
        "case_id": "case-03-formula-gamma",
        "record_id": "d23-formula-description",
        "target_text": "gamma",
        "target_occurrence": 1,
        "target_display_label": "gamma",
        "target_category": "hallucinated_symbol",
    },
    {
        "case_id": "case-04-logo-triangle",
        "record_id": "d23-logo-structure",
        "target_text": "三角形",
        "target_occurrence": 1,
        "target_display_label": "triangle",
        "target_category": "hallucinated_geometry",
    },
)

VISUALIZATION_CONFIG = {
    "schema_version": "1.0",
    "normalization": "linear_minmax_per_case",
    "colormap": "turbo",
    "overlay_alpha": 0.45,
    "interpolation": "bilinear",
    "figure_width_inches": 12,
    "figure_height_inches": 4,
    "figure_dpi": 180,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_artifacts(
    *,
    day23_results: Path,
    day23_generation_config: Path,
    day22_manifest: Path,
    output_root: Path,
) -> dict[str, Any]:
    rows = read_jsonl(day23_results)
    by_record_id = {str(row.get("record_id")): row for row in rows}
    if len(by_record_id) != len(rows):
        raise ValueError("Day23 record_id values must be unique")

    with day22_manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    by_image_id = {str(row.get("image_id")): row for row in manifest_rows}
    generation = read_json(day23_generation_config)
    generation_sha = sha256_file(day23_generation_config)

    cases: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        record = by_record_id.get(spec["record_id"])
        if record is None:
            raise ValueError(f"missing Day23 record: {spec['record_id']}")
        if record.get("status") != "PASS" or not str(record.get("raw_response", "")).strip():
            raise ValueError(f"Day23 record is not a completed PASS: {spec['record_id']}")
        response = str(record["raw_response"])
        if response.count(spec["target_text"]) < int(spec["target_occurrence"]):
            raise ValueError(
                f"configured target is absent from {spec['record_id']}: {spec['target_text']}"
            )
        image = by_image_id.get(str(record.get("image_id")))
        if image is None:
            raise ValueError(f"image missing from Day22 manifest: {record.get('image_id')}")
        if image.get("sha256") != record.get("image_sha256"):
            raise ValueError(f"Day22/Day23 image hash mismatch: {record.get('image_id')}")
        cases.append(
            {
                **spec,
                "image_id": record["image_id"],
                "image_type": record["image_type"],
                "image_relative_path": record["image_relative_path"],
                "image_sha256": record["image_sha256"],
                "prompt": record["prompt"],
                "expected_day23_response": response,
                "expected_day23_response_sha256": sha256_bytes(response.encode("utf-8")),
                "day23_image_grid_thw": record["image_grid_thw"],
                "day23_input_tokens": record["input_tokens"],
                "day23_output_tokens": record["output_tokens"],
            }
        )

    payload = {
        "schema_version": "1.0",
        "status": "preregistered",
        "repository": generation["repository"],
        "revision": generation["revision"],
        "generation_config_sha256": generation_sha,
        "layer_number_1_based": 20,
        "case_count": len(cases),
        "cases": cases,
    }
    write_json(output_root / "configs" / "attention_cases.json", payload)
    write_json(output_root / "configs" / "visualization_config.json", VISUALIZATION_CONFIG)
    return {
        "case_count": len(cases),
        "attention_cases_sha256": sha256_file(output_root / "configs" / "attention_cases.json"),
        "visualization_config_sha256": sha256_file(
            output_root / "configs" / "visualization_config.json"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day23-results", type=Path, required=True)
    parser.add_argument("--day23-generation-config", type=Path, required=True)
    parser.add_argument("--day22-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_artifacts(
        day23_results=args.day23_results,
        day23_generation_config=args.day23_generation_config,
        day22_manifest=args.day22_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
