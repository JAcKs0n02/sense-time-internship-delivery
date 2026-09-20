#!/usr/bin/env python3
"""Build the audited, balanced Day26 v7 multimodal datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


CATEGORIES = ("natural_scene", "ocr", "chart_table", "ui", "formula")
SPLITS = ("train", "dev", "final")
TRAIN_MODES = ("direct", "evidence", "verification", "grounding")
EVAL_MODES = ("direct", "verification")
EXPECTED_IMAGE_COUNTS = {"train": 50, "dev": 10, "final": 10}
EXPECTED_CATEGORY_COUNTS = {"train": 10, "dev": 2, "final": 2}

REQUIRED_FIELDS = (
    "image_id",
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
    "subject",
    "observable_facts",
    "gold_transcription",
    "uncertainty_notes",
    "verification_claim",
    "claim_supported",
    "prompts",
    "answers",
    "annotation_status",
)


def _is_nonempty(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple)):
        return bool(value)
    return value is not None


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _require_text(mapping: dict, field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid or missing {field}")
    return value.strip()


def validate_v7_fact(fact: dict) -> None:
    """Validate one manually reviewed image fact card."""

    for field in REQUIRED_FIELDS:
        if field not in fact or not _is_nonempty(fact[field]):
            if field == "claim_supported" and isinstance(fact.get(field), bool):
                continue
            raise ValueError(f"missing {field}")

    if fact["split"] not in SPLITS:
        raise ValueError(f"unsupported split: {fact['split']}")
    if fact["category"] not in CATEGORIES:
        raise ValueError(f"unsupported category: {fact['category']}")
    if fact["annotation_status"] != "double_checked":
        raise ValueError("annotation_status must be double_checked")
    if not isinstance(fact["claim_supported"], bool):
        raise ValueError("claim_supported must be boolean")

    for field in ("source_page_url", "download_url", "license_url"):
        if not _is_http_url(fact[field]):
            raise ValueError(f"invalid {field}")
    revision_kind = fact["source_revision_kind"]
    if revision_kind == "git_commit":
        if not re.fullmatch(r"[0-9a-f]{40}", fact["source_revision"]):
            raise ValueError("invalid source_revision for git_commit")
    elif revision_kind == "retrieval_date":
        if not re.fullmatch(r"20[0-9]{2}-[01][0-9]-[0-3][0-9]", fact["source_revision"]):
            raise ValueError("invalid source_revision for retrieval_date")
    else:
        raise ValueError("unsupported source_revision_kind")
    if not re.fullmatch(r"[0-9a-f]{64}", fact["sha256"]):
        raise ValueError("invalid sha256")

    observable = fact["observable_facts"]
    if (
        not isinstance(observable, list)
        or len(observable) < 3
        or any(not isinstance(item, str) or not item.strip() for item in observable)
        or len({item.strip() for item in observable}) != len(observable)
    ):
        raise ValueError("observable_facts must contain at least three unique statements")
    notes = fact["uncertainty_notes"]
    if not isinstance(notes, list) or any(
        not isinstance(item, str) or not item.strip() for item in notes
    ):
        raise ValueError("uncertainty_notes must be a list of non-empty strings")

    transcription = fact["gold_transcription"]
    if not isinstance(transcription, dict):
        raise ValueError("gold_transcription must be an object")
    for field in ("applicable", "value", "review_status", "ambiguous_tokens"):
        if field not in transcription:
            raise ValueError(f"gold_transcription missing {field}")
    transcription_required = fact["category"] in {"ocr", "formula"}
    if transcription.get("applicable") is not transcription_required:
        raise ValueError("gold_transcription applicable/category mismatch")
    if transcription_required:
        _require_text(transcription, "value")
        if transcription.get("review_status") != "double_checked":
            raise ValueError("gold_transcription review_status must be double_checked")
        ambiguous = transcription.get("ambiguous_tokens")
        if not isinstance(ambiguous, list) or ambiguous:
            raise ValueError("gold_transcription ambiguous_tokens must be empty")
    elif transcription.get("review_status") != "not_applicable":
        raise ValueError("gold_transcription review_status must be not_applicable")

    expected_modes = TRAIN_MODES if fact["split"] == "train" else EVAL_MODES
    prompts = fact["prompts"]
    answers = fact["answers"]
    if not isinstance(prompts, dict) or set(prompts) != set(expected_modes):
        raise ValueError(f"prompts must contain exactly {expected_modes}")
    if not isinstance(answers, dict) or set(answers) != set(expected_modes):
        raise ValueError(f"answers must contain exactly {expected_modes}")
    for mode in expected_modes:
        _require_text(prompts, mode)
        if mode != "verification":
            _require_text(answers, mode)

    verification = answers["verification"]
    if not isinstance(verification, dict):
        raise ValueError("verification answer must be an object")
    expected_verdict = "supported" if fact["claim_supported"] else "unsupported"
    if verification.get("verdict") != expected_verdict:
        raise ValueError("verification verdict contradicts claim_supported")
    _require_text(verification, "evidence")


def validate_v7_fact_collection(facts: list[dict]) -> None:
    """Validate complete split sizes, category balance, and claim balance."""

    if not isinstance(facts, list):
        raise ValueError("facts must be a list")
    for fact in facts:
        validate_v7_fact(fact)

    image_ids = [fact["image_id"] for fact in facts]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("duplicate image_id")
    digests = [fact["sha256"] for fact in facts]
    if len(digests) != len(set(digests)):
        raise ValueError("duplicate image sha256")

    split_counts = Counter(fact["split"] for fact in facts)
    if dict(split_counts) != EXPECTED_IMAGE_COUNTS:
        raise ValueError(f"image split counts mismatch: {dict(split_counts)}")
    for split in SPLITS:
        category_counts = Counter(
            fact["category"] for fact in facts if fact["split"] == split
        )
        expected = Counter(
            {category: EXPECTED_CATEGORY_COUNTS[split] for category in CATEGORIES}
        )
        if category_counts != expected:
            raise ValueError(
                f"category balance mismatch for {split}: {dict(category_counts)}"
            )
        for category in CATEGORIES:
            claims = Counter(
                fact["claim_supported"]
                for fact in facts
                if fact["split"] == split and fact["category"] == category
            )
            half = EXPECTED_CATEGORY_COUNTS[split] // 2
            if claims != Counter({True: half, False: half}):
                raise ValueError(
                    "verification balance mismatch for "
                    f"{split}/{category}: {dict(claims)}"
                )


def _verification_target(fact: dict) -> str:
    verdict = "符合" if fact["claim_supported"] else "不符合"
    return f"结论：{verdict}。可见依据：{fact['answers']['verification']['evidence']}"


def _prompt(fact: dict, mode: str) -> str:
    instruction = fact["prompts"][mode].strip()
    if mode == "verification":
        instruction = f"{instruction}\n待核验陈述：{fact['verification_claim']}"
    return f"<image>\n{instruction}"


def build_v7_records(facts: list[dict]) -> dict[str, list[dict]]:
    """Convert a complete v7 fact-card collection into auditable records."""

    validate_v7_fact_collection(facts)
    result = {split: [] for split in SPLITS}
    for fact in sorted(facts, key=lambda row: (row["split"], row["category"], row["image_id"])):
        modes = TRAIN_MODES if fact["split"] == "train" else EVAL_MODES
        for mode in modes:
            target = (
                _verification_target(fact)
                if mode == "verification"
                else fact["answers"][mode].strip()
            )
            result[fact["split"]].append(
                {
                    "id": f"{fact['image_id']}--{mode}",
                    "split": fact["split"],
                    "category": fact["category"],
                    "mode": mode,
                    "source_image_id": fact["image_id"],
                    "image_relative_path": fact["image_relative_path"],
                    "image_sha256": fact["sha256"],
                    "task_type": fact["category"],
                    "prompt": _prompt(fact, mode),
                    "user_instruction": _prompt(fact, mode).removeprefix("<image>\n"),
                    "reference_answer": target,
                    "target_answer": target,
                    "verification_claim": (
                        fact["verification_claim"] if mode == "verification" else None
                    ),
                    "claim_supported": (
                        fact["claim_supported"] if mode == "verification" else None
                    ),
                    "source": fact["source_page_url"],
                    "license": fact["license"],
                    "construction_method": "human-double-checked-v7",
                }
            )
    return result


def to_llamafactory_records(records: list[dict]) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["target_answer"]},
            ],
            "images": [row["image_relative_path"]],
        }
        for row in records
    ]


def write_v7_outputs(facts_path: Path, output_dir: Path) -> None:
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    records = build_v7_records(facts)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        internal_path = output_dir / f"week5_vlm_{split}_v7_internal.json"
        export_path = output_dir / f"week5_vlm_{split}_v7.json"
        internal_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        export_path.write_text(
            json.dumps(to_llamafactory_records(rows), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_v7_outputs(args.facts, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
