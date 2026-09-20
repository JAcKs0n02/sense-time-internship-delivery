#!/usr/bin/env python3
"""Validation helpers for Day26 source provenance and split isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse


REQUIRED_SOURCE_FIELDS = (
    "source_page_url",
    "download_url",
    "author",
    "license",
    "license_url",
    "sha256",
)


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_source_fact(fact: dict) -> None:
    for field in REQUIRED_SOURCE_FIELDS:
        if not str(fact.get(field, "")).strip():
            raise ValueError(f"missing source field: {field}")
    for field in ("source_page_url", "download_url", "license_url"):
        if not _is_http_url(fact[field]):
            raise ValueError(f"invalid URL in {field}")
    if not re.fullmatch(r"[0-9a-f]{64}", fact["sha256"]):
        raise ValueError("invalid sha256")


def validate_split_isolation(facts: list[dict]) -> None:
    owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for fact in facts:
        validate_source_fact(fact)
        owners[fact["sha256"]].append((fact["image_id"], fact["split"]))
    duplicates = {digest: rows for digest, rows in owners.items() if len(rows) > 1}
    cross_split = {
        digest: rows
        for digest, rows in duplicates.items()
        if len({split for _, split in rows}) > 1
    }
    if cross_split:
        raise ValueError(f"cross-split sha256 overlap: {cross_split}")
    if duplicates:
        raise ValueError(f"duplicate sha256: {duplicates}")


def normalize_text(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def cross_split_text_audit(internal: dict[str, list[dict]]) -> dict:
    train_prompts = {
        normalize_text(row["user_instruction"]) for row in internal["train"]
    }
    train_answers = {normalize_text(row["target_answer"]) for row in internal["train"]}
    return {
        "evaluation_scope": "image_disjoint_shared_task_templates",
        "prompt_template_overlap_records": {
            split: sum(
                normalize_text(row["user_instruction"]) in train_prompts
                for row in internal[split]
            )
            for split in ("dev", "final")
        },
        "target_answer_exact_overlap_records": {
            split: sum(
                normalize_text(row["target_answer"]) in train_answers
                for row in internal[split]
            )
            for split in ("dev", "final")
        },
    }


def find_prompt_collisions(prompts: list[str], excluded: list[str]) -> list[dict[str, str]]:
    normalized_excluded = [(text, normalize_text(text)) for text in excluded]
    collisions = []
    for prompt in prompts:
        normalized_prompt = normalize_text(prompt)
        for original, normalized in normalized_excluded:
            if normalized_prompt == normalized:
                collisions.append({"prompt": prompt, "excluded": original})
    return collisions


def resolve_inside(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"image path escapes data root: {relative_path}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_images(data_dir: Path, facts: list[dict]) -> None:
    from PIL import Image

    for fact in facts:
        image_path = resolve_inside(data_dir, fact["image_relative_path"])
        if not image_path.is_file():
            raise ValueError(f"missing image: {image_path}")
        if _sha256(image_path) != fact["sha256"]:
            raise ValueError(f"image sha256 mismatch: {fact['image_id']}")
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            if [image.width, image.height] != [fact["width"], fact["height"]]:
                raise ValueError(f"image dimensions mismatch: {fact['image_id']}")


def _validate_counts(facts: list[dict], internal: dict[str, list[dict]], exported: dict[str, list[dict]]) -> None:
    expected_images = {"train": 50, "dev": 10, "final": 10}
    expected_records = {"train": 200, "dev": 20, "final": 20}
    image_counts = Counter(row["split"] for row in facts)
    if dict(image_counts) != expected_images:
        raise ValueError(f"image split counts mismatch: {dict(image_counts)}")
    for split, expected in expected_records.items():
        if len(internal[split]) != expected or len(exported[split]) != expected:
            raise ValueError(f"record count mismatch for {split}")
    train_categories = Counter(row["category"] for row in internal["train"])
    if set(train_categories.values()) != {40} or len(train_categories) != 5:
        raise ValueError(f"training category balance mismatch: {dict(train_categories)}")


def _validate_records(internal: dict[str, list[dict]], exported: dict[str, list[dict]]) -> None:
    all_ids = []
    for split in ("train", "dev", "final"):
        if len(internal[split]) != len(exported[split]):
            raise ValueError(f"internal/export length mismatch: {split}")
        for source, target in zip(internal[split], exported[split]):
            all_ids.append(source["id"])
            if source["split"] != split:
                raise ValueError(f"wrong split in record: {source['id']}")
            if not source["user_instruction"].strip() or not source["target_answer"].strip():
                raise ValueError(f"blank prompt or answer: {source['id']}")
            if target.get("images") != [source["image_relative_path"]]:
                raise ValueError(f"image mapping mismatch: {source['id']}")
            messages = target.get("messages", [])
            if [message.get("role") for message in messages] != ["user", "assistant"]:
                raise ValueError(f"invalid message roles: {source['id']}")
            if messages[0].get("content", "").count("<image>") != 1:
                raise ValueError(f"image token mismatch: {source['id']}")
            if messages[1].get("content") != source["target_answer"]:
                raise ValueError(f"answer mapping mismatch: {source['id']}")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate record id")


def _near_day25(prompts: list[str], excluded: list[str], threshold: float = 0.92) -> list[dict]:
    collisions = []
    for prompt in prompts:
        left = normalize_text(prompt)
        for text in excluded:
            right = normalize_text(text)
            ratio = SequenceMatcher(None, left, right).ratio()
            if ratio >= threshold:
                collisions.append({"prompt": prompt, "excluded": text, "ratio": ratio})
    return collisions


def validate_bundle(data_dir: Path, day25_path: Path) -> tuple[dict, dict, dict]:
    facts_path = data_dir / "source_image_facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    for fact in facts:
        validate_source_fact(fact)
    validate_split_isolation(facts)
    _validate_images(data_dir, facts)

    internal = {}
    exported = {}
    for split in ("train", "dev", "final"):
        internal[split] = json.loads(
            (data_dir / f"week5_vlm_{split}_internal.json").read_text(encoding="utf-8")
        )
        exported[split] = json.loads(
            (data_dir / f"week5_vlm_{split}.json").read_text(encoding="utf-8")
        )
    _validate_counts(facts, internal, exported)
    _validate_records(internal, exported)
    text_audit = cross_split_text_audit(internal)
    if any(text_audit["target_answer_exact_overlap_records"].values()):
        raise ValueError(
            "train-to-evaluation target answer overlap: "
            f'{text_audit["target_answer_exact_overlap_records"]}'
        )

    day25 = json.loads(day25_path.read_text(encoding="utf-8"))["records"]
    excluded_hashes = {row["image_sha256"] for row in day25}
    reused = sorted({row["sha256"] for row in facts} & excluded_hashes)
    if reused:
        raise ValueError(f"Day25 image pollution: {reused}")
    excluded_text = [text for row in day25 for text in (row["prompt"], row["question"], row["fake_answer"])]
    prompts = [row["user_instruction"] for split in internal.values() for row in split]
    exact_collisions = find_prompt_collisions(prompts, excluded_text)
    near_collisions = _near_day25(prompts, excluded_text)
    if exact_collisions or near_collisions:
        raise ValueError(
            f"Day25 prompt pollution: exact={exact_collisions}, near={near_collisions}"
        )

    split_manifest = {
        "schema_version": "1.0",
        "status": "FROZEN_BEFORE_TRAINING",
        "splits": {
            split: [
                {"image_id": row["image_id"], "sha256": row["sha256"]}
                for row in facts
                if row["split"] == split
            ]
            for split in ("train", "dev", "final")
        },
        "cross_split_sha256_overlap": [],
        "day25_image_sha256_overlap": [],
        "day25_prompt_exact_overlap": [],
        "day25_prompt_near_overlap": [],
        "cross_split_text_audit": text_audit,
    }
    data_files = [
        "source_image_manifest.json",
        "source_image_facts.json",
        *[
            f"week5_vlm_{split}{suffix}.json"
            for split in ("train", "dev", "final")
            for suffix in ("_internal", "")
        ],
    ]
    data_manifest = {
        "schema_version": "1.0",
        "status": "FROZEN_BEFORE_TRAINING",
        "freeze_date": "2026-08-21",
        "base_model": "Qwen/Qwen2-VL-7B-Instruct",
        "base_revision": "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
        "source_image_count": len(facts),
        "unique_image_sha256_count": len({row["sha256"] for row in facts}),
        "record_counts": {split: len(internal[split]) for split in internal},
        "training_category_counts": dict(Counter(row["category"] for row in internal["train"])),
        "files": {
            name: {"sha256": _sha256(data_dir / name), "bytes": (data_dir / name).stat().st_size}
            for name in data_files
        },
    }
    validation = {
        "schema_version": "1.0",
        "status": "PASS",
        "checks": {
            "source_fields_complete_and_revision_pinned": True,
            "images_exist_decode_and_match_sha": True,
            "70_unique_source_images": True,
            "record_counts_200_20_20": True,
            "training_categories_5x40": True,
            "llamafactory_export_schema": True,
            "cross_split_sha256_overlap_zero": True,
            "day25_image_overlap_zero": True,
            "day25_prompt_exact_and_near_overlap_zero": True,
            "cross_split_target_answer_exact_overlap_zero": True,
            "shared_task_prompt_templates_disclosed": True,
        },
    }
    return data_manifest, split_manifest, validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--day25-cases", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    data_manifest, split_manifest, validation = validate_bundle(args.data_dir, args.day25_cases)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "data_manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.data_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.results_dir / "day26_data_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
