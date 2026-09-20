#!/usr/bin/env python3
"""Merge, validate, de-duplicate, and split the Week 4 preference dataset."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def normalize_prompt(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "Z", "C"))
    )


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _prompt(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ""
    last = conversations[-1]
    return last.get("value", "") if isinstance(last, dict) else ""


def validate_preference_record(record: Any) -> list[str]:
    errors: set[str] = set()
    if not isinstance(record, dict):
        return ["record_not_object"]
    if not _non_empty_string(record.get("id")):
        errors.add("id_missing")

    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        errors.add("conversations_invalid")
    else:
        for message in conversations:
            if (
                not isinstance(message, dict)
                or message.get("from") not in {"human", "gpt"}
                or not _non_empty_string(message.get("value"))
            ):
                errors.add("conversations_invalid")
                break
        if isinstance(conversations[-1], dict) and conversations[-1].get("from") != "human":
            errors.add("conversation_must_end_with_human")

    chosen = record.get("chosen")
    rejected = record.get("rejected")
    for label, message in (("chosen", chosen), ("rejected", rejected)):
        if (
            not isinstance(message, dict)
            or message.get("from") != "gpt"
            or not _non_empty_string(message.get("value"))
        ):
            errors.add(f"{label}_invalid")
    if (
        isinstance(chosen, dict)
        and isinstance(rejected, dict)
        and isinstance(chosen.get("value"), str)
        and isinstance(rejected.get("value"), str)
        and chosen["value"].strip() == rejected["value"].strip()
    ):
        errors.add("responses_not_distinct")

    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("source_kind") == "ultrafeedback":
        if metadata.get("preference_type") != "mixed_external":
            errors.add("ultrafeedback_preference_type_invalid")
        if not _non_empty_string(metadata.get("source_id")):
            errors.add("ultrafeedback_source_id_missing")
    elif isinstance(record.get("source"), dict) and record["source"].get("type") == "self_built":
        if record.get("preference_type") not in {
            "factuality", "safety", "completeness", "helpfulness", "format"
        }:
            errors.add("self_built_preference_type_invalid")
        if record.get("scene_kind") not in {"business", "general"}:
            errors.add("self_built_scene_kind_invalid")
        if not _non_empty_string(record.get("construction_reason")):
            errors.add("construction_reason_missing")
        review = record.get("quality_review")
        if (
            not isinstance(review, dict)
            or review.get("status") != "approved"
            or review.get("single_dimension") is not True
            or review.get("safe_to_publish") is not True
            or not _non_empty_string(review.get("reviewer_kind"))
        ):
            errors.add("quality_review_invalid")
    else:
        errors.add("source_kind_invalid")
    return sorted(errors)


def find_exact_duplicate_groups(records: Iterable[dict[str, Any]]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        normalized = normalize_prompt(_prompt(record))
        if normalized:
            groups[normalized].append(record.get("id", ""))
    return sorted(sorted(ids) for ids in groups.values() if len(ids) > 1)


def find_near_duplicate_candidates(
    records: list[dict[str, Any]],
    *,
    threshold: float = 0.9,
) -> list[dict[str, Any]]:
    if len(records) < 2:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer

    prompts = [normalize_prompt(_prompt(record)) for record in records]
    matrix = TfidfVectorizer(analyzer="char", ngram_range=(2, 5)).fit_transform(prompts)
    similarities = (matrix @ matrix.T).tocoo()
    candidates: list[dict[str, Any]] = []
    prefilter_threshold = max(0.5, threshold - 0.35)
    for left, right, tfidf_score in zip(similarities.row, similarities.col, similarities.data):
        if left >= right or tfidf_score < prefilter_threshold or prompts[left] == prompts[right]:
            continue
        score = difflib.SequenceMatcher(None, prompts[left], prompts[right], autojunk=False).ratio()
        if score < threshold:
            continue
        left_id, right_id = sorted((records[left]["id"], records[right]["id"]))
        candidates.append(
            {
                "left_id": left_id,
                "right_id": right_id,
                "similarity": round(float(score), 6),
                "decision": "review_required",
            }
        )
    return sorted(candidates, key=lambda item: (item["left_id"], item["right_id"]))


def resolve_near_duplicate_candidates(
    records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {record["id"]: record for record in records}
    resolved: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        left = by_id[item["left_id"]]
        right = by_id[item["right_id"]]
        if _source_kind(left) == _source_kind(right) == "self_built":
            if left.get("preference_type") == right.get("preference_type"):
                item["decision"] = "retained_distinct_topic"
                item["review_note"] = (
                    "两条记录共享受控写作模板，但主题事实不同，偏好标签相同且 Prompt 语义目标不同。"
                )
            else:
                item["decision"] = "retained_distinct_preference_dimension"
                item["review_note"] = (
                    "两条记录可能复用同一主题，但分别检验不同主偏好维度，不构成同一训练任务。"
                )
        resolved.append(item)
    return resolved


def find_eval_contamination(
    records: Iterable[dict[str, Any]],
    holdout: dict[str, Any],
) -> list[dict[str, str]]:
    eval_by_prompt = {
        normalize_prompt(item["text"]): item["id"]
        for item in holdout.get("prompts", [])
        if isinstance(item, dict) and _non_empty_string(item.get("text"))
    }
    matches: list[dict[str, str]] = []
    for record in records:
        eval_id = eval_by_prompt.get(normalize_prompt(_prompt(record)))
        if eval_id is not None:
            matches.append(
                {
                    "record_id": record.get("id", ""),
                    "eval_id": eval_id,
                    "match_type": "exact",
                }
            )
    return sorted(matches, key=lambda item: (item["record_id"], item["eval_id"]))


def find_near_eval_contamination(
    records: Iterable[dict[str, Any]],
    holdout: dict[str, Any],
    *,
    threshold: float = 0.9,
) -> list[dict[str, Any]]:
    eval_prompts = [
        (item["id"], normalize_prompt(item["text"]))
        for item in holdout.get("prompts", [])
        if isinstance(item, dict)
        and _non_empty_string(item.get("id"))
        and _non_empty_string(item.get("text"))
    ]
    matches: list[dict[str, Any]] = []
    for record in records:
        prompt = normalize_prompt(_prompt(record))
        for eval_id, eval_prompt in eval_prompts:
            if not prompt or prompt == eval_prompt:
                continue
            similarity = difflib.SequenceMatcher(
                None,
                prompt,
                eval_prompt,
                autojunk=False,
            ).ratio()
            if similarity >= threshold:
                matches.append(
                    {
                        "record_id": record.get("id", ""),
                        "eval_id": eval_id,
                        "match_type": "high_similarity",
                        "similarity": round(float(similarity), 6),
                    }
                )
    return sorted(matches, key=lambda item: (item["record_id"], item["eval_id"]))


def _source_kind(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("source_kind") == "ultrafeedback":
        return "ultrafeedback"
    return "self_built"


def _stratum(record: dict[str, Any]) -> str:
    if _source_kind(record) == "ultrafeedback":
        return f"ultrafeedback:{record['metadata'].get('upstream_source', 'unknown')}"
    return f"self_built:{record.get('preference_type')}:{record.get('scene_kind')}"


def _split_priority(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:split:{record_id}".encode("utf-8")).hexdigest()


def stratified_split(
    records: list[dict[str, Any]],
    *,
    seed: int,
    validation_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio_out_of_range")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_stratum(record)].append(record)

    target = round(len(records) * validation_ratio)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for name, group in groups.items():
        exact = len(group) * validation_ratio
        quotas[name] = math.floor(exact)
        remainders.append((exact - quotas[name], name))
    remaining = target - sum(quotas.values())
    for _, name in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        quotas[name] += 1

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for name in sorted(groups):
        ordered = sorted(groups[name], key=lambda record: _split_priority(seed, record["id"]))
        quota = quotas[name]
        validation.extend(ordered[:quota])
        train.extend(ordered[quota:])
    return (
        sorted(train, key=lambda record: record["id"]),
        sorted(validation, key=lambda record: record["id"]),
    )


def build_dataset(
    ultrafeedback: list[dict[str, Any]],
    self_built: list[dict[str, Any]],
    holdout: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    full = sorted([*ultrafeedback, *self_built], key=lambda record: record.get("id", ""))
    errors: set[str] = set()
    record_errors = {
        record.get("id", f"index-{index}"): validate_preference_record(record)
        for index, record in enumerate(full)
    }
    record_errors = {record_id: value for record_id, value in record_errors.items() if value}
    if record_errors:
        errors.add("record_validation")

    ids = [record.get("id") for record in full]
    if len(ids) != len(set(ids)):
        errors.add("duplicate_id")
    duplicate_groups = find_exact_duplicate_groups(full)
    if duplicate_groups:
        errors.add("exact_prompt_duplicate")
    contamination = find_eval_contamination(full, holdout)
    if contamination:
        errors.add("eval_contamination")
    near_eval_contamination = find_near_eval_contamination(full, holdout)
    if near_eval_contamination:
        errors.add("eval_near_contamination")

    source_counts = Counter(_source_kind(record) for record in full)
    if source_counts != Counter({"ultrafeedback": 500, "self_built": 210}):
        errors.add("source_count_mismatch")
    category_counts = Counter(record.get("preference_type") for record in self_built)
    if category_counts != Counter(
        {category: 42 for category in ("factuality", "safety", "completeness", "helpfulness", "format")}
    ):
        errors.add("self_built_category_count_mismatch")

    train, validation_records = stratified_split(full, seed=seed, validation_ratio=0.1)
    near_duplicate_candidates = resolve_near_duplicate_candidates(
        full,
        find_near_duplicate_candidates(full),
    )
    train_ids = {record["id"] for record in train}
    validation_ids = {record["id"] for record in validation_records}
    if train_ids & validation_ids or train_ids | validation_ids != set(ids):
        errors.add("split_integrity")

    validation = {
        "valid": not errors,
        "errors": sorted(errors),
        "total_count": len(full),
        "ultrafeedback_count": source_counts["ultrafeedback"],
        "self_built_count": source_counts["self_built"],
        "train_count": len(train),
        "validation_count": len(validation_records),
        "exact_duplicate_group_count": len(duplicate_groups),
        "eval_contamination_count": len(contamination),
        "eval_near_contamination_count": len(near_eval_contamination),
    }
    return {
        "full": full,
        "train": train,
        "validation_records": validation_records,
        "validation": validation,
        "record_errors": record_errors,
        "duplicate_groups": duplicate_groups,
        "near_duplicate_candidates": near_duplicate_candidates,
        "eval_contamination": contamination,
        "eval_near_contamination": near_eval_contamination,
        "stats": {
            "source_counts": dict(sorted(source_counts.items())),
            "self_built_category_counts": dict(sorted(category_counts.items())),
            "self_built_scene_counts": dict(
                sorted(Counter(record.get("scene_kind") for record in self_built).items())
            ),
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_split_csv(
    path: Path,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": record["id"],
            "split": split,
            "source_kind": _source_kind(record),
            "stratum": _stratum(record),
        }
        for split, records in (("train", train), ("validation", validation))
        for record in records
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "split", "source_kind", "stratum"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["id"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ultrafeedback", type=Path, required=True)
    parser.add_argument("--self-built", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ultrafeedback = json.loads(args.ultrafeedback.read_text(encoding="utf-8"))
    self_built = json.loads(args.self_built.read_text(encoding="utf-8"))
    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    result = build_dataset(ultrafeedback, self_built, holdout, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_path = args.output_dir / "week4_preferences_full.json"
    train_path = args.output_dir / "week4_dpo_train.json"
    validation_path = args.output_dir / "week4_dpo_validation.json"
    _write_json(full_path, result["full"])
    _write_json(train_path, result["train"])
    _write_json(validation_path, result["validation_records"])
    _write_split_csv(
        args.output_dir / "preference_split.csv",
        result["train"],
        result["validation_records"],
    )

    stats = {
        **result["stats"],
        "total_count": len(result["full"]),
        "train_count": len(result["train"]),
        "validation_count": len(result["validation_records"]),
        "seed": args.seed,
        "validation_ratio": 0.1,
        "file_sha256": {
            "week4_preferences_full.json": _sha256(full_path),
            "week4_dpo_train.json": _sha256(train_path),
            "week4_dpo_validation.json": _sha256(validation_path),
        },
    }
    _write_json(args.output_dir / "preference_stats.json", stats)
    _write_json(
        args.output_dir / "duplicate_audit.json",
        {
            "exact_duplicate_groups": result["duplicate_groups"],
            "near_duplicate_candidates": result["near_duplicate_candidates"],
            "eval_contamination": result["eval_contamination"],
            "eval_near_contamination": result["eval_near_contamination"],
            "record_errors": result["record_errors"],
        },
    )
    _write_json(args.output_dir / "day18_validation.json", result["validation"])
    prompt_type_counts = Counter(item.get("prompt_type") for item in holdout.get("prompts", []))
    _write_json(
        args.output_dir / "evaluation_holdout_manifest.json",
        {
            "schema_version": "1.0",
            "frozen_at": holdout.get("frozen_at"),
            "prompt_count": len(holdout.get("prompts", [])),
            "prompt_type_counts": dict(sorted(prompt_type_counts.items())),
            "source_file": args.holdout.name,
            "source_sha256": _sha256(args.holdout),
            "training_allowed": False,
        },
    )
    dataset_info = {
        name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }
        for name, file_name in (
            ("week4_dpo_train", "week4_dpo_train.json"),
            ("week4_dpo_validation", "week4_dpo_validation.json"),
        )
    }
    _write_json(args.output_dir / "dataset_info_week4.json", dataset_info)
    print(json.dumps(result["validation"], ensure_ascii=False, sort_keys=True))
    return 0 if result["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
