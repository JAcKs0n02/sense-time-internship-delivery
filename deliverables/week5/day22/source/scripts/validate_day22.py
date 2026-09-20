#!/usr/bin/env python3
"""Fail-closed validation for the frozen Day 22 image bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from build_image_manifest import EXPECTED_TYPES, FIELDNAMES, safe_image_path, sha256


EXPECTED_REPOSITORY = "Qwen/Qwen2-VL-7B-Instruct"
EXPECTED_REVISION = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
EXPECTED_SCENE_SHA256 = "32c63883fbedbd2da1590245b1ccaa55bdb26ffa63bf3328456202356ed307d6"


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def read_ground_truth(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ground truth must be a JSON object")
    return payload


def add_error(errors: list[str], label: str, detail: str) -> None:
    errors.append(f"{label}: {detail}")


def validate_manifest(images_dir: Path, header: list[str], rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if header != list(FIELDNAMES):
        add_error(errors, "manifest_schema", f"actual={header}, expected={list(FIELDNAMES)}")
    if len(rows) != 5:
        add_error(errors, "exact_five_types", f"expected 5 rows, got {len(rows)}")
    image_ids = [row.get("image_id", "") for row in rows]
    if any(not value for value in image_ids) or len(set(image_ids)) != len(image_ids):
        add_error(errors, "exact_five_types", "image_id values must be non-empty and unique")
    actual_types = {row.get("image_type", "") for row in rows}
    if actual_types != EXPECTED_TYPES:
        add_error(
            errors,
            "exact_five_types",
            f"actual={sorted(actual_types)}, expected={sorted(EXPECTED_TYPES)}",
        )
    for row_number, row in enumerate(rows, start=2):
        image_id = row.get("image_id") or f"row-{row_number}"
        if not row.get("source", "").startswith(("https://", "http://")):
            add_error(errors, "license_privacy", f"{image_id} source must be an HTTP(S) URL")
        if not row.get("license", "").strip():
            add_error(errors, "license_privacy", f"{image_id} license is empty")
        try:
            path = safe_image_path(images_dir, row.get("relative_path", ""))
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                actual_width, actual_height = image.size
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    add_error(
                        errors,
                        "file_integrity",
                        f"{image_id} unsupported image format: {image.format}",
                    )
            expected_width = int(row.get("width", ""))
            expected_height = int(row.get("height", ""))
            expected_bytes = int(row.get("file_bytes", ""))
            if actual_width != expected_width:
                add_error(
                    errors,
                    "file_integrity",
                    f"{image_id} width mismatch: actual={actual_width}, expected={expected_width}",
                )
            if actual_height != expected_height:
                add_error(
                    errors,
                    "file_integrity",
                    f"{image_id} height mismatch: actual={actual_height}, expected={expected_height}",
                )
            if path.stat().st_size != expected_bytes:
                add_error(
                    errors,
                    "file_integrity",
                    f"{image_id} file_bytes mismatch: actual={path.stat().st_size}, expected={expected_bytes}",
                )
            actual_sha = sha256(path)
            if actual_sha != row.get("sha256"):
                add_error(
                    errors,
                    "file_integrity",
                    f"{image_id} sha256 mismatch: actual={actual_sha}, expected={row.get('sha256')}",
                )
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            add_error(errors, "file_integrity", f"{image_id} cannot be validated: {exc}")
    return errors


def validate_ground_truth(payload: dict[str, Any], manifest_ids: set[str]) -> tuple[list[str], int]:
    errors: list[str] = []
    if payload.get("schema_version") != "1.0":
        add_error(errors, "ground_truth", "schema_version must equal 1.0")
    entries = payload.get("images")
    if not isinstance(entries, list):
        add_error(errors, "ground_truth", "images must be a list")
        return errors, 0
    ground_truth_ids = [entry.get("image_id", "") for entry in entries if isinstance(entry, dict)]
    if len(entries) != 5 or len(set(ground_truth_ids)) != len(ground_truth_ids):
        add_error(errors, "ground_truth", "expected exactly five unique ground-truth records")
    if set(ground_truth_ids) != manifest_ids:
        add_error(
            errors,
            "ground_truth",
            f"image IDs mismatch: actual={sorted(ground_truth_ids)}, expected={sorted(manifest_ids)}",
        )
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            add_error(errors, "ground_truth", f"entry {index} must be an object")
            continue
        image_id = entry.get("image_id") or f"entry-{index}"
        if not isinstance(entry.get("summary"), str) or not entry["summary"].strip():
            add_error(errors, "ground_truth", f"{image_id} summary is empty")
        for field in ("visible_text", "key_facts", "unknown_or_not_inferable"):
            value = entry.get(field)
            if not isinstance(value, list):
                add_error(errors, "ground_truth", f"{image_id} {field} must be a list")
            elif field != "visible_text" and not value:
                add_error(errors, "ground_truth", f"{image_id} {field} must not be empty")
        for field in ("privacy_review", "license_review"):
            if entry.get(field) != "PASS":
                add_error(errors, "license_privacy", f"{image_id} {field} must equal PASS")
    return errors, len(entries)


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def validate_remote_evidence(evidence_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Validate the small evidence bundle copied back from AutoDL."""
    errors: list[str] = []
    documents: dict[str, dict[str, Any]] = {}
    filenames = (
        "environment.json",
        "model_manifest.json",
        "offline_load_smoke.json",
        "remote_status.json",
    )
    for filename in filenames:
        try:
            documents[filename] = read_json_object(evidence_dir / filename)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            add_error(errors, "remote_evidence", f"{filename} cannot be read: {exc}")
    if len(documents) != len(filenames):
        return errors, {"evidence_file_count": len(documents)}

    environment = documents["environment.json"]
    model = documents["model_manifest.json"]
    smoke = documents["offline_load_smoke.json"]
    remote_status = documents["remote_status.json"]

    if environment.get("status") != "PASS":
        add_error(errors, "remote_environment", "environment status must equal PASS")
    gpus = environment.get("gpu")
    gpu = gpus[0] if isinstance(gpus, list) and gpus and isinstance(gpus[0], dict) else {}
    if int(gpu.get("memory_total_mib", 0)) < 16000:
        add_error(errors, "remote_environment", "GPU VRAM must be at least 16000 MiB")
    torch_info = environment.get("torch") if isinstance(environment.get("torch"), dict) else {}
    if torch_info.get("cuda_available") is not True:
        add_error(errors, "remote_environment", "PyTorch CUDA must be available")
    packages = environment.get("packages") if isinstance(environment.get("packages"), dict) else {}
    for package in ("torch", "transformers", "accelerate", "qwen-vl-utils"):
        if not packages.get(package):
            add_error(errors, "remote_environment", f"missing package version: {package}")
    pip_check = environment.get("pip_check") if isinstance(environment.get("pip_check"), dict) else {}
    if pip_check.get("returncode") != 0:
        add_error(errors, "remote_environment", "pip check did not pass")

    if model.get("status") != "downloaded_verified":
        add_error(errors, "remote_model", "model status must equal downloaded_verified")
    for payload_name, payload in (("model", model), ("smoke", smoke), ("remote_status", remote_status)):
        if payload.get("repository") != EXPECTED_REPOSITORY:
            add_error(errors, "remote_model", f"{payload_name} repository mismatch")
        if payload.get("revision") != EXPECTED_REVISION:
            add_error(errors, "remote_model", f"{payload_name} revision mismatch")
    files = model.get("files")
    if not isinstance(files, list):
        files = []
        add_error(errors, "remote_model", "model files must be a list")
    if model.get("file_count") != len(files) or len(files) != 17:
        add_error(errors, "remote_model", "model file_count must equal 17 manifest rows")
    safe_rows: list[dict[str, Any]] = []
    for index, row in enumerate(files):
        if not isinstance(row, dict):
            add_error(errors, "remote_model", f"model file row {index} is not an object")
            continue
        relative_path = row.get("relative_path")
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not is_sha256(row.get("sha256"))
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] <= 0
        ):
            add_error(errors, "remote_model", f"invalid model file row {index}")
            continue
        safe_rows.append(row)
    total_bytes = sum(row["bytes"] for row in safe_rows)
    weight_rows = [row for row in safe_rows if row["relative_path"].endswith(".safetensors")]
    weight_bytes = sum(row["bytes"] for row in weight_rows)
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if model.get("total_bytes") != total_bytes or total_bytes < 15_000_000_000:
        add_error(errors, "remote_model", "model total_bytes is inconsistent or too small")
    if model.get("weight_shards") != len(weight_rows) or len(weight_rows) != 5:
        add_error(errors, "remote_model", "model must contain exactly five weight shards")
    if model.get("weight_bytes") != weight_bytes:
        add_error(errors, "remote_model", "model weight_bytes is inconsistent")
    if model.get("files_digest_sha256") != digest:
        add_error(errors, "remote_model", "model manifest digest mismatch")

    if smoke.get("status") != "PASS":
        add_error(errors, "offline_smoke", "offline smoke status must equal PASS")
    if smoke.get("local_files_only") is not True:
        add_error(errors, "offline_smoke", "local_files_only must equal true")
    offline_environment = (
        smoke.get("offline_environment")
        if isinstance(smoke.get("offline_environment"), dict)
        else {}
    )
    if offline_environment != {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}:
        add_error(errors, "offline_smoke", "offline environment flags are incomplete")
    image = smoke.get("image") if isinstance(smoke.get("image"), dict) else {}
    if image.get("sha256") != EXPECTED_SCENE_SHA256:
        add_error(errors, "offline_smoke", "smoke image sha256 mismatch")
    if not isinstance(smoke.get("raw_output"), str) or not smoke["raw_output"].strip():
        add_error(errors, "offline_smoke", "raw_output must be non-empty")
    runtime = smoke.get("runtime") if isinstance(smoke.get("runtime"), dict) else {}
    if runtime.get("dtype") not in {"bfloat16", "float16"}:
        add_error(errors, "offline_smoke", "runtime dtype must be bfloat16 or float16")
    generation = smoke.get("generation") if isinstance(smoke.get("generation"), dict) else {}
    if generation.get("do_sample") is not False or generation.get("seed") != 42:
        add_error(errors, "offline_smoke", "generation must be deterministic with seed 42")
    processor = smoke.get("processor") if isinstance(smoke.get("processor"), dict) else {}
    if processor.get("effective_size") != {
        "shortest_edge": processor.get("min_pixels"),
        "longest_edge": processor.get("max_pixels"),
    }:
        add_error(errors, "offline_smoke", "effective processor size does not match pixel budget")
    if remote_status.get("status") != "PASS":
        add_error(errors, "remote_evidence", "remote_status must equal PASS")

    summary = {
        "evidence_file_count": len(documents),
        "repository": model.get("repository"),
        "revision": model.get("revision"),
        "gpu": gpu.get("name"),
        "gpu_memory_total_mib": gpu.get("memory_total_mib"),
        "model_file_count": model.get("file_count"),
        "weight_shards": model.get("weight_shards"),
        "total_bytes": model.get("total_bytes"),
        "offline_smoke_status": smoke.get("status"),
        "offline_smoke_output": smoke.get("raw_output"),
    }
    return errors, summary


def build_result(
    rows: list[dict[str, str]],
    ground_truth_count: int,
    errors: list[str],
    remote_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels: tuple[str, ...] = (
        "manifest_schema",
        "exact_five_types",
        "file_integrity",
        "ground_truth",
        "license_privacy",
    )
    if remote_summary is not None:
        labels += ("remote_environment", "remote_model", "offline_smoke", "remote_evidence")
    checks = [
        {
            "check": label,
            "status": "FAIL" if any(error.startswith(f"{label}:") for error in errors) else "PASS",
        }
        for label in labels
    ]
    summary: dict[str, Any] = {
        "image_count": len(rows),
        "unique_image_ids": len({row.get("image_id", "") for row in rows}),
        "image_types": sorted({row.get("image_type", "") for row in rows}),
        "ground_truth_count": ground_truth_count,
    }
    if remote_summary is not None:
        summary["remote_evidence"] = remote_summary
    return {
        "status": "FAIL" if errors else "PASS",
        "summary": summary,
        "checks": checks,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--remote-evidence-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    ground_truth_count = 0
    remote_summary: dict[str, Any] | None = None
    try:
        header, rows = read_manifest(args.manifest)
        errors.extend(validate_manifest(args.images_dir, header, rows))
        ground_truth = read_ground_truth(args.ground_truth)
        ground_truth_errors, ground_truth_count = validate_ground_truth(
            ground_truth, {row.get("image_id", "") for row in rows}
        )
        errors.extend(ground_truth_errors)
        if args.remote_evidence_dir is not None:
            remote_errors, remote_summary = validate_remote_evidence(args.remote_evidence_dir)
            errors.extend(remote_errors)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        add_error(errors, "input_read", str(exc))
    result = build_result(rows, ground_truth_count, errors, remote_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        print("Day 22 image validation failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Day 22 image validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
