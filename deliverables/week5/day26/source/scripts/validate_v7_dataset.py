#!/usr/bin/env python3
"""Recompute the Day26 v7 data contract from bytes instead of trusting summaries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


CORE_RELATIVE_PATHS = (
    "data/source_image_specs_v7.json",
    "data/source_image_manifest_v7.json",
    "data/source_image_manifest_v7.csv",
    "data/source_image_facts_v7.json",
    "data/week5_vlm_train_v7_internal.json",
    "data/week5_vlm_train_v7.json",
    "data/week5_vlm_dev_v7_internal.json",
    "data/week5_vlm_dev_v7.json",
    "data/week5_vlm_final_v7_internal.json",
    "data/week5_vlm_final_v7.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"path escapes artifact root: {relative}")
    return candidate


def build_file_manifest(root: Path, relative_paths: list[str] | tuple[str, ...]) -> dict:
    files = {}
    for relative in sorted(set(relative_paths)):
        path = _resolve_inside(root, relative)
        if not path.is_file():
            raise ValueError(f"missing declared artifact: {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {"schema_version": "day26-v7-data-manifest-1", "file_count": len(files), "files": files}


def declared_files_match(root: Path, manifest: dict) -> bool:
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("file_count") != len(files):
        return False
    for relative, expected in files.items():
        try:
            path = _resolve_inside(root, relative)
        except ValueError:
            return False
        if not path.is_file():
            return False
        if path.stat().st_size != expected.get("bytes") or sha256(path) != expected.get("sha256"):
            return False
    return True


def _load_builder():
    path = Path(__file__).with_name("build_v7_dataset.py")
    spec = importlib.util.spec_from_file_location("day26_v7_builder_for_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_prompt(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _manifest_paths(source_manifest: list[dict]) -> list[str]:
    image_paths = [f"data/{row['image_relative_path']}" for row in source_manifest]
    return [*CORE_RELATIVE_PATHS, *image_paths]


def write_data_manifest(root: Path, output_path: Path) -> dict:
    source_manifest = json.loads(
        (root / "data/source_image_manifest_v7.json").read_text(encoding="utf-8")
    )
    manifest = build_file_manifest(root, _manifest_paths(source_manifest))
    manifest["image_counts"] = dict(Counter(row["split"] for row in source_manifest))
    manifest["record_counts"] = {"train": 200, "dev": 20, "final": 20}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_artifacts(root: Path) -> dict:
    from PIL import Image

    builder = _load_builder()
    data = root / "data"
    source_manifest = json.loads(
        (data / "source_image_manifest_v7.json").read_text(encoding="utf-8")
    )
    facts = json.loads((data / "source_image_facts_v7.json").read_text(encoding="utf-8"))
    declared = json.loads((data / "data_manifest_v7.json").read_text(encoding="utf-8"))

    split_counts = Counter(row.get("split") for row in source_manifest)
    image_counts = {split: split_counts.get(split, 0) for split in ("train", "dev", "final")}
    expected_image_counts = {"train": 50, "dev": 10, "final": 10}

    image_bytes_ok = True
    for row in source_manifest:
        path = _resolve_inside(data, row["image_relative_path"])
        if not path.is_file():
            image_bytes_ok = False
            continue
        if path.stat().st_size != row.get("file_bytes") or sha256(path) != row.get("sha256"):
            image_bytes_ok = False
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.size != (row.get("width"), row.get("height")):
                    image_bytes_ok = False
        except Exception:
            image_bytes_ok = False

    builder.validate_v7_fact_collection(facts)
    manifest_by_id = {row["image_id"]: row for row in source_manifest}
    facts_by_id = {row["image_id"]: row for row in facts}
    linkage_fields = (
        "split",
        "category",
        "image_relative_path",
        "source_page_url",
        "download_url",
        "source_revision",
        "source_revision_kind",
        "author",
        "license",
        "license_url",
        "sha256",
    )
    fact_linkage_ok = set(manifest_by_id) == set(facts_by_id) and all(
        all(facts_by_id[image_id].get(field) == row.get(field) for field in linkage_fields)
        for image_id, row in manifest_by_id.items()
    )

    rebuilt = builder.build_v7_records(facts)
    actual_internal = {
        split: json.loads((data / f"week5_vlm_{split}_v7_internal.json").read_text(encoding="utf-8"))
        for split in ("train", "dev", "final")
    }
    actual_export = {
        split: json.loads((data / f"week5_vlm_{split}_v7.json").read_text(encoding="utf-8"))
        for split in ("train", "dev", "final")
    }
    internal_match = actual_internal == rebuilt
    export_match = all(
        actual_export[split] == builder.to_llamafactory_records(rebuilt[split])
        for split in rebuilt
    )
    record_counts = {split: len(rows) for split, rows in actual_internal.items()}
    expected_record_counts = {"train": 200, "dev": 20, "final": 20}

    record_ids = [row["id"] for rows in actual_internal.values() for row in rows]
    train_examples = {
        json.dumps(
            [row["prompt"], row["target_answer"], row["image_sha256"]],
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in actual_internal["train"]
    }
    prompt_sets = {
        split: {_normalize_prompt(row["user_instruction"]) for row in rows}
        for split, rows in actual_internal.items()
    }
    prompt_collisions = {
        "train_dev": sorted(prompt_sets["train"] & prompt_sets["dev"]),
        "train_final": sorted(prompt_sets["train"] & prompt_sets["final"]),
        "dev_final": sorted(prompt_sets["dev"] & prompt_sets["final"]),
    }
    image_sets = {
        split: {row["sha256"] for row in facts if row["split"] == split}
        for split in ("train", "dev", "final")
    }
    image_isolation = not (
        image_sets["train"] & image_sets["dev"]
        or image_sets["train"] & image_sets["final"]
        or image_sets["dev"] & image_sets["final"]
    )

    checks = {
        "declared_file_hashes": declared_files_match(root, declared),
        "source_manifest_inventory": len(source_manifest) == 70 and image_counts == expected_image_counts,
        "source_image_bytes": image_bytes_ok,
        "fact_contract": len(facts) == 70,
        "fact_manifest_linkage": fact_linkage_ok,
        "generated_internal_records": internal_match,
        "generated_exports": export_match,
        "record_counts": record_counts == expected_record_counts,
        "unique_record_ids": len(record_ids) == len(set(record_ids)) == 240,
        "unique_train_examples": len(train_examples) == 200,
        "zero_cross_split_prompts": not any(prompt_collisions.values()),
        "zero_cross_split_images": image_isolation,
    }
    return {
        "schema_version": "day26-v7-validation-1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "image_counts": image_counts,
        "record_counts": record_counts,
        "prompt_collisions": prompt_collisions,
        "data_manifest_sha256": sha256(data / "data_manifest_v7.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.write_manifest:
        write_data_manifest(args.root, args.root / "data/data_manifest_v7.json")
    result = validate_artifacts(args.root)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
