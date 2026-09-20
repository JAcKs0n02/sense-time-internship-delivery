#!/usr/bin/env python3
"""Validate the complete Day26 teacher contract and pre-registered quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "data_validation": evidence.get("data_status") == "PASS",
        "training_records_200": evidence.get("training_record_count") == 200,
        "training_sampling_records_300": evidence.get("training_sampling_record_count") == 300,
        "dev_records_20": evidence.get("dev_record_count") == 20,
        "final_records_20": evidence.get("final_record_count") == 20,
        "trainable_parameters": evidence.get("trainable_status") == "PASS",
        "smoke_return_code": evidence.get("smoke_exit") == 0,
        "formal_return_code": evidence.get("formal_exit") == 0,
        "adapter_load_and_freeze": evidence.get("adapter_status") == "PASS",
        "paired_responses_20": evidence.get("paired_response_count") == 20,
        "blind_scores_20": evidence.get("blind_score_count") == 20,
        "quality_mean_gain": evidence.get("mean_gain_check") is True,
        "quality_lora_wins": evidence.get("lora_wins_check") is True,
        "quality_hallucination_not_worse": evidence.get("hallucination_check") is True,
        "training_input_files_sha256": evidence.get("training_input_files_match_manifest") is True,
        "training_dataset_lineage": evidence.get("training_dataset_lineage_match") is True,
        "training_config_lineage": evidence.get("training_config_lineage_match") is True,
        "adapter_lineage": evidence.get("adapter_lineage_match") is True,
        "evaluation_lineage": evidence.get("evaluation_lineage_match") is True,
        "scoring_config_lineage": evidence.get("scoring_config_lineage_match") is True,
    }
    return {
        "schema_version": "1.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "evidence": evidence,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_match_manifest(root: Path, manifest: dict, *, expected_count: int) -> bool:
    files = manifest.get("files", {})
    if not isinstance(files, dict) or len(files) != expected_count:
        return False
    return all(
        (root / name).is_file()
        and (root / name).stat().st_size == metadata.get("bytes")
        and sha256(root / name) == metadata.get("sha256")
        for name, metadata in files.items()
    )


def evaluation_lineage_checks(
    evaluation_summary: dict[str, Any], expected: dict[str, Any]
) -> dict[str, bool]:
    fields = (
        "generation_config_sha256",
        "cases_sha256",
        "adapter_config_sha256",
        "adapter_model_sha256",
        "lora_scale",
        "base_model_repository",
        "base_model_revision",
        "base_model_files_digest_sha256",
        "base_model_manifest_sha256",
    )
    return {
        field: evaluation_summary.get(field) == expected.get(field) for field in fields
    }


def collect_evidence(root: Path) -> dict[str, Any]:
    data = root / "source" / "data"
    results = root / "source" / "results"
    data_validation = read_json(results / "day26_data_validation.json")
    split_manifest = read_json(data / "split_manifest.json")
    text_audit = split_manifest["cross_split_text_audit"]
    trainable = read_json(results / "trainable_parameters_summary.json")
    smoke = read_json(results / "smoke_training_summary.json")
    training = read_json(results / "training_summary_v6.json")
    adapter = read_json(results / "adapter_manifest_v6_compact.json")
    evaluation_run = read_json(results / "final_eval_summary_v6_remediation.json")
    comparison = read_json(results / "comparison_summary_v6_remediation.json")
    paired = read_json(results / "base_lora_paired_final_v6_remediation.json")
    blind_scores_path = results / "blind_scores_final_v6_remediation.json"
    blind_scores = read_json(blind_scores_path)
    evaluation = read_json(root / "configs" / "evaluation_config_v6_final.json")
    training_input_manifest_path = data / "training_input_manifest_v6.json"
    training_input_manifest = read_json(training_input_manifest_path)
    input_files_match = files_match_manifest(data, training_input_manifest, expected_count=3)
    unique_sha = training_input_manifest["files"]["week5_vlm_train_grounded.json"]["sha256"]
    weighted_sha = training_input_manifest["files"][
        "week5_vlm_train_grounded_weighted.json"
    ]["sha256"]
    training_config_path = root / "configs" / "qwen2vl_lora_candidate_v6_grounded.yaml"
    evaluation_config_path = root / "configs" / "evaluation_config_v6_final.json"
    adapter_sha = adapter.get("adapter_model", {}).get("sha256")
    base_manifest_path = (
        root.parent / "day22" / "source" / "results" / "remote_evidence" / "model_manifest.json"
    )
    base_manifest = read_json(base_manifest_path)
    final_cases_path = data / "week5_vlm_final_internal.json"
    expected_evaluation_lineage = {
        "generation_config_sha256": sha256(evaluation_config_path),
        "cases_sha256": sha256(final_cases_path),
        "adapter_config_sha256": adapter.get("adapter_config", {}).get("sha256"),
        "adapter_model_sha256": adapter_sha,
        "lora_scale": training.get("selected_lora_scale"),
        "base_model_repository": base_manifest.get("repository"),
        "base_model_revision": base_manifest.get("revision"),
        "base_model_files_digest_sha256": base_manifest.get("files_digest_sha256"),
        "base_model_manifest_sha256": sha256(base_manifest_path),
    }
    evaluation_checks = evaluation_lineage_checks(
        evaluation_run, expected_evaluation_lineage
    )
    evidence = {
        "data_status": data_validation.get("status"),
        "training_record_count": len(read_json(data / "week5_vlm_train_internal.json")),
        "training_sampling_record_count": len(
            read_json(data / "week5_vlm_train_grounded_weighted.json")
        ),
        "training_input_files_match_manifest": input_files_match,
        "training_dataset_lineage_match": (
            training.get("training_dataset_unique_sha256") == unique_sha
            and training.get("training_dataset_weighted_sha256") == weighted_sha
            and training.get("training_input_source_manifest_sha256")
            == sha256(training_input_manifest_path)
        ),
        "training_config_lineage_match": (
            training.get("training_config_sha256") == sha256(training_config_path)
        ),
        "adapter_lineage_match": training.get("adapter_model_sha256") == adapter_sha,
        "evaluation_lineage_checks": evaluation_checks,
        "evaluation_lineage_match": all(evaluation_checks.values()),
        "scoring_config_lineage_match": (
            comparison.get("scoring_config_sha256") == sha256(evaluation_config_path)
        ),
        "dev_record_count": len(read_json(data / "week5_vlm_dev_internal.json")),
        "final_record_count": len(read_json(data / "week5_vlm_final_internal.json")),
        "trainable_status": trainable.get("status"),
        "smoke_exit": smoke.get("exit_code"),
        "formal_exit": training.get("exit_code"),
        "adapter_status": adapter.get("status"),
        "paired_response_count": len(paired),
        "blind_score_count": len(blind_scores),
        "evaluation_scope": text_audit.get("evaluation_scope"),
        "prompt_template_overlap_records": text_audit.get(
            "prompt_template_overlap_records"
        ),
        "target_answer_exact_overlap_records": text_audit.get(
            "target_answer_exact_overlap_records"
        ),
        "selected_candidate": f'{training.get("run_id")}/{training.get("selected_checkpoint")}',
        "selected_lora_scale": training.get("selected_lora_scale"),
        "blind_seed": evaluation.get("blind_seed"),
        "blind_scores_sha256_before_reveal": hashlib.sha256(
            blind_scores_path.read_bytes()
        ).hexdigest(),
        "mean_base": comparison.get("mean_base"),
        "mean_lora": comparison.get("mean_lora"),
        "mean_gain": comparison.get("mean_gain"),
        "lora_wins": comparison.get("lora_wins"),
        "base_hallucination_rate": comparison.get("base_hallucination_rate"),
        "lora_hallucination_rate": comparison.get("lora_hallucination_rate"),
        "mean_gain_check": comparison.get("checks", {}).get("mean_gain"),
        "lora_wins_check": comparison.get("checks", {}).get("lora_wins"),
        "hallucination_check": comparison.get("checks", {}).get("hallucination_rate"),
    }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_evidence(collect_evidence(args.root))
    result["completed_at"] = utc_now()
    result["conclusion"] = (
        "数据、训练、冻结、adapter 与盲评流程均完成；纠偏候选通过胜题数和幻觉率门槛，"
        "但未达到预注册的均分提升 0.50，因此 Day26 效果总门槛如实判定为 FAIL。"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
