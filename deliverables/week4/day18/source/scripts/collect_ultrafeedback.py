#!/usr/bin/env python3
"""Validate, normalize, and deterministically sample UltraFeedback pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping


ROLE_MAP = {"user": "human", "assistant": "gpt"}


def stable_priority(seed: int, source: str, source_id: str) -> str:
    payload = f"{seed}:{source}:{source_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_messages(value: Any, label: str) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{label}_messages_invalid")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"{label}_message_not_object:{index}")
        role = message.get("role")
        content = message.get("content")
        if role not in ROLE_MAP or not _non_empty_string(content):
            raise ValueError(f"{label}_message_invalid:{index}")
        messages.append({"role": role, "content": content.strip()})
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{label}_final_role")
    if messages[-2]["role"] != "user":
        raise ValueError(f"{label}_context_must_end_with_user")
    return messages


def normalize_ultrafeedback_row(
    row: dict[str, Any],
    *,
    upstream_index: int,
    revision: str,
) -> dict[str, Any]:
    prompt_id = row.get("prompt_id")
    if not _non_empty_string(prompt_id):
        raise ValueError("prompt_id_missing")
    if not _non_empty_string(revision):
        raise ValueError("revision_missing")

    chosen_messages = _validate_messages(row.get("chosen"), "chosen")
    rejected_messages = _validate_messages(row.get("rejected"), "rejected")
    if chosen_messages[:-1] != rejected_messages[:-1]:
        raise ValueError("context_mismatch")

    chosen_text = chosen_messages[-1]["content"]
    rejected_text = rejected_messages[-1]["content"]
    if chosen_text.strip() == rejected_text.strip():
        raise ValueError("responses_not_distinct")

    score_chosen = row.get("score_chosen")
    score_rejected = row.get("score_rejected")
    if not isinstance(score_chosen, Real) or not isinstance(score_rejected, Real):
        raise ValueError("score_invalid")
    if float(score_chosen) <= float(score_rejected):
        raise ValueError("score_chosen_not_greater")

    upstream_source = row.get("source")
    if not _non_empty_string(upstream_source):
        raise ValueError("source_missing")

    conversations = [
        {"from": ROLE_MAP[message["role"]], "value": message["content"]}
        for message in chosen_messages[:-1]
    ]
    chosen_score = float(score_chosen)
    rejected_score = float(score_rejected)
    return {
        "id": f"uf-{prompt_id.strip()}",
        "conversations": conversations,
        "chosen": {"from": "gpt", "value": chosen_text},
        "rejected": {"from": "gpt", "value": rejected_text},
        "metadata": {
            "source_kind": "ultrafeedback",
            "preference_type": "mixed_external",
            "preference_basis": "upstream_score",
            "source_id": prompt_id.strip(),
            "upstream_source": upstream_source.strip(),
            "upstream_index": upstream_index,
            "revision": revision.strip(),
            "score_chosen": chosen_score,
            "score_rejected": rejected_score,
            "score_margin": chosen_score - rejected_score,
            "construction_reason": (
                "UltraFeedback 上游评分将 chosen 排在 rejected 之前"
                f"（{chosen_score} > {rejected_score}）。"
            ),
        },
    }


def select_records(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    seed: int,
    revision: str,
    content_exclusions: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible: list[tuple[str, dict[str, Any]]] = []
    excluded_by_reason: Counter[str] = Counter()
    exclusions = content_exclusions or {}
    input_count = 0
    for upstream_index, row in enumerate(rows):
        input_count += 1
        source_id = row.get("prompt_id")
        if isinstance(source_id, str) and source_id in exclusions:
            excluded_by_reason[exclusions[source_id]] += 1
            continue
        try:
            normalized = normalize_ultrafeedback_row(
                row,
                upstream_index=upstream_index,
                revision=revision,
            )
        except ValueError as exc:
            excluded_by_reason[str(exc).split(":", 1)[0]] += 1
            continue
        priority = stable_priority(seed, "ultrafeedback", normalized["metadata"]["source_id"])
        eligible.append((priority, normalized))

    if len(eligible) < count:
        raise ValueError(f"eligible_pool_below_requested:{len(eligible)}<{count}")

    eligible.sort(key=lambda item: item[0])
    selected = [record for _, record in eligible[:count]]
    audit = {
        "input_count": input_count,
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "excluded_count": sum(excluded_by_reason.values()),
        "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
        "seed": seed,
        "revision": revision,
        "selection_rule": 'sort SHA256("42:ultrafeedback:<prompt_id>") ascending',
    }
    return selected, audit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_content_exclusions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    exclusions: dict[str, str] = {}
    for item in payload.get("exclusions", []):
        source_id = item.get("source_id")
        reason = item.get("reason")
        if not _non_empty_string(source_id) or not _non_empty_string(reason):
            raise ValueError("content_exclusion_invalid")
        exclusions[source_id.strip()] = reason.strip()
    return exclusions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--retrieved-at", required=True)
    parser.add_argument("--content-exclusions", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import pandas as pd

    frame = pd.read_parquet(args.input_parquet)
    content_exclusions = _load_content_exclusions(args.content_exclusions)
    records, audit = select_records(
        frame.to_dict(orient="records"),
        count=args.count,
        seed=args.seed,
        revision=args.revision,
        content_exclusions=content_exclusions,
    )
    _write_json(args.output, records)
    manifest = {
        "schema_version": "1.0",
        "dataset": {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "split": args.split,
            "license": args.license,
            "retrieved_at": args.retrieved_at,
        },
        "selection": audit,
        "files": {
            "input_parquet": {
                "name": args.input_parquet.name,
                "size_bytes": args.input_parquet.stat().st_size,
                "sha256": _sha256(args.input_parquet),
            },
            "output_json": {
                "name": args.output.name,
                "size_bytes": args.output.stat().st_size,
                "sha256": _sha256(args.output),
            },
        },
    }
    _write_json(args.manifest, manifest)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
