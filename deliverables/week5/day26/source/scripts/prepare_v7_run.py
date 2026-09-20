#!/usr/bin/env python3
"""Create immutable pre-run and post-run lineage manifests for Day26 v7."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INPUT_FILES = {
    "training_dataset": "source/data/week5_vlm_train_v7.json",
    "fact_cards": "source/data/source_image_facts_v7.json",
    "source_manifest": "source/data/source_image_manifest_v7.json",
    "data_manifest": "source/data/data_manifest_v7.json",
    "dataset_info": "configs/dataset_info.json",
    "framework_lock": "configs/framework_lock.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, relative_path: str | None = None) -> dict:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    record = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if relative_path is not None:
        record["relative_path"] = relative_path
    return record


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _relative_inside(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside Day26 root: {path}") from error


def build_pre_run_manifest(day26_root: Path, config_path: Path, candidate_id: str) -> dict:
    config_relative = _relative_inside(day26_root, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    revision = config.get("model_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("training config must pin a 40-character base model revision")

    paths = {**INPUT_FILES, "training_config": config_relative}
    input_files = {
        label: file_record(day26_root / relative, relative)
        for label, relative in paths.items()
    }
    payload = {
        "schema_version": "day26-v7-run-manifest-1",
        "stage": "pre_run",
        "candidate_id": candidate_id,
        "run_name": config.get("run_name"),
        "base_model_repository": "Qwen/Qwen2-VL-7B-Instruct",
        "base_model_revision": revision,
        "training_dataset_name": config.get("dataset"),
        "development_dataset_name": config.get("eval_dataset"),
        "training_parameters": {
            "learning_rate": config.get("learning_rate"),
            "num_train_epochs": config.get("num_train_epochs"),
            "seed": config.get("seed"),
            "lora_rank": config.get("lora_rank"),
            "lora_alpha": config.get("lora_alpha"),
            "lora_target": config.get("lora_target"),
            "freeze_vision_tower": config.get("freeze_vision_tower"),
            "freeze_multi_modal_projector": config.get("freeze_multi_modal_projector"),
        },
        "input_files": input_files,
    }
    payload["pre_run_fingerprint_sha256"] = canonical_sha256(payload)
    return payload


def build_post_run_manifest(pre_run: dict, adapter_dir: Path) -> dict:
    if pre_run.get("stage") != "pre_run":
        raise ValueError("post-run lineage requires a pre_run manifest")
    required = ("adapter_config.json", "adapter_model.safetensors")
    adapter_files = {name: file_record(adapter_dir / name, name) for name in required}
    readme = adapter_dir / "README.md"
    if readme.is_file():
        adapter_files["README.md"] = file_record(readme, "README.md")
    payload = {
        **{key: value for key, value in pre_run.items() if key != "stage"},
        "stage": "post_run",
        "pre_run_manifest_sha256": canonical_sha256(pre_run),
        "adapter_files": adapter_files,
    }
    payload["post_run_fingerprint_sha256"] = canonical_sha256(payload)
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("pre")
    pre.add_argument("--day26-root", type=Path, required=True)
    pre.add_argument("--config", type=Path, required=True)
    pre.add_argument("--candidate-id", required=True)
    pre.add_argument("--output", type=Path, required=True)
    post = subparsers.add_parser("post")
    post.add_argument("--pre-run", type=Path, required=True)
    post.add_argument("--adapter-dir", type=Path, required=True)
    post.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "pre":
        payload = build_pre_run_manifest(args.day26_root, args.config, args.candidate_id)
    else:
        payload = build_post_run_manifest(
            json.loads(args.pre_run.read_text(encoding="utf-8")), args.adapter_dir
        )
    write_json(args.output, payload)
    print(json.dumps({"status": "PASS", "stage": payload["stage"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
