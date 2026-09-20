#!/usr/bin/env python3
"""Validate the complete local and AutoDL Day 18 artifact set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(source_root: Path) -> dict[str, Any]:
    final_dir = source_root / "data" / "final"
    results_dir = source_root / "results"
    manifests_dir = source_root / "manifests"
    paths = {
        "full": final_dir / "week4_preferences_full.json",
        "train": final_dir / "week4_dpo_train.json",
        "validation": final_dir / "week4_dpo_validation.json",
        "stats": results_dir / "preference_stats.json",
        "split": results_dir / "preference_split.csv",
        "duplicate_audit": results_dir / "duplicate_audit.json",
        "structural": results_dir / "day18_validation.json",
        "lengths": results_dir / "length_statistics.json",
        "schema_smoke": results_dir / "llamafactory_schema_smoke.json",
        "remote_environment": results_dir / "remote_environment.json",
        "shutdown": results_dir / "autodl_shutdown_evidence.json",
        "dataset_info": manifests_dir / "dataset_info_week4.json",
        "ultrafeedback": source_root / "data" / "interim" / "ultrafeedback_500.json",
        "self_built": source_root / "data" / "raw" / "self_built_preferences.json",
    }
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        return {"valid": False, "errors": [f"missing:{name}" for name in missing]}

    full = _read_json(paths["full"])
    train = _read_json(paths["train"])
    validation = _read_json(paths["validation"])
    stats = _read_json(paths["stats"])
    duplicate_audit = _read_json(paths["duplicate_audit"])
    structural = _read_json(paths["structural"])
    lengths = _read_json(paths["lengths"])
    smoke = _read_json(paths["schema_smoke"])
    remote = _read_json(paths["remote_environment"])
    shutdown = _read_json(paths["shutdown"])
    dataset_info = _read_json(paths["dataset_info"])
    ultrafeedback = _read_json(paths["ultrafeedback"])
    self_built = _read_json(paths["self_built"])
    with paths["split"].open(encoding="utf-8", newline="") as stream:
        split_rows = list(csv.DictReader(stream))

    errors: list[str] = []

    def require(condition: bool, code: str) -> None:
        if not condition:
            errors.append(code)

    full_ids = [record.get("id") for record in full]
    train_ids = {record.get("id") for record in train}
    validation_ids = {record.get("id") for record in validation}
    require(len(full) == 710, "full_count")
    require(len(ultrafeedback) == 500, "ultrafeedback_count")
    require(len(self_built) == 210, "self_built_count")
    require(len(train) == 639, "train_count")
    require(len(validation) == 71, "validation_count")
    require(len(full_ids) == len(set(full_ids)), "duplicate_id")
    require(not (train_ids & validation_ids), "split_overlap")
    require(train_ids | validation_ids == set(full_ids), "split_coverage")
    require(len(split_rows) == 710, "split_csv_count")

    file_hashes = stats.get("file_sha256", {})
    for name in (
        "week4_preferences_full.json",
        "week4_dpo_train.json",
        "week4_dpo_validation.json",
    ):
        require(file_hashes.get(name) == _sha256(final_dir / name), f"stats_hash:{name}")

    require(structural.get("valid") is True, "structural_validation")
    require(structural.get("exact_duplicate_group_count") == 0, "exact_duplicates")
    require(structural.get("eval_contamination_count") == 0, "eval_exact_contamination")
    require(structural.get("eval_near_contamination_count") == 0, "eval_near_contamination")
    require(not duplicate_audit.get("exact_duplicate_groups"), "duplicate_audit_exact")
    require(not duplicate_audit.get("eval_contamination"), "duplicate_audit_eval_exact")
    require(not duplicate_audit.get("eval_near_contamination"), "duplicate_audit_eval_near")
    require(
        all(item.get("decision") != "review_required" for item in duplicate_audit.get("near_duplicate_candidates", [])),
        "near_duplicate_review_pending",
    )

    require(lengths.get("status") == "pass", "length_status")
    require(lengths.get("record_count") == 710, "length_record_count")
    require(lengths.get("records_over_cutoff") == 0, "length_over_cutoff")
    require(smoke.get("status") == "pass", "llamafactory_schema_status")
    require(smoke.get("llamafactory_version") == "0.9.3", "llamafactory_version")
    require(smoke.get("loaded_smoke_samples") == 8, "llamafactory_smoke_count")
    require(smoke.get("finetuning_stage") == "dpo", "llamafactory_stage")
    require(remote.get("model_path_exists") is True, "remote_model_path")
    require(remote.get("structural_failure_count") == 0, "remote_structural_failures")
    for name in ("week4_dpo_train.json", "week4_dpo_validation.json"):
        require(
            remote.get("data_files", {}).get(name, {}).get("sha256") == _sha256(final_dir / name),
            f"remote_hash:{name}",
        )
    require(shutdown.get("shutdown_confirmed") is True, "autodl_shutdown")
    require(shutdown.get("console_status") == "已关机", "autodl_shutdown_status")
    require(
        set(dataset_info) == {"week4_dpo_train", "week4_dpo_validation"}
        and all(item.get("ranking") is True for item in dataset_info.values()),
        "dataset_info",
    )

    return {
        "valid": not errors,
        "errors": sorted(errors),
        "record_count": len(full),
        "ultrafeedback_count": len(ultrafeedback),
        "self_built_count": len(self_built),
        "train_count": len(train),
        "validation_count": len(validation),
        "records_over_cutoff": lengths.get("records_over_cutoff"),
        "llamafactory_schema_status": smoke.get("status"),
        "autodl_shutdown_confirmed": shutdown.get("shutdown_confirmed"),
        "train_sha256": _sha256(paths["train"]),
        "validation_sha256": _sha256(paths["validation"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_artifacts(args.source_root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
