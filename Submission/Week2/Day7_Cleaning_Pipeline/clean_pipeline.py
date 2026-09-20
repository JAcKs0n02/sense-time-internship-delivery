#!/usr/bin/env python3
"""Week 2 Day 7 data-cleaning pipeline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from dataclasses import dataclass, field, replace
import hashlib
import html as html_lib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from typing import Any
import unicodedata


KNOWN_INLINE_TAGS = {
    "a",
    "abbr",
    "b",
    "bdi",
    "bdo",
    "cite",
    "code",
    "del",
    "em",
    "font",
    "i",
    "ins",
    "kbd",
    "mark",
    "q",
    "s",
    "samp",
    "small",
    "span",
    "strike",
    "strong",
    "sub",
    "sup",
    "time",
    "u",
    "var",
}
KNOWN_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
VOID_TAGS = {"br", "hr", "img", "input", "meta", "link", "source", "wbr"}
DROP_CONTENT_TAGS = {"script", "style"}
UNWANTED_FORMAT_CHARACTERS = {
    "\u061c",  # Arabic letter mark
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2060",  # word joiner
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
    "\ufeff",  # BOM / zero-width no-break space
}


class RecordRejected(ValueError):
    """A structured input record cannot become a valid training sample."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class CanonicalRecord:
    sample_id: str
    messages: list[dict[str, str]]
    provenance: dict[str, Any] = field(default_factory=dict)
    alpaca_instruction: str | None = None
    alpaca_input: str = ""
    was_truncated: bool = False


class _ConservativeHTMLStripper(HTMLParser):
    """Remove known HTML while preserving unknown angle-bracket syntax."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.drop_depth = 0
        self.changed = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        normalized = tag.lower()
        if normalized in DROP_CONTENT_TAGS:
            self.changed = True
            self.drop_depth += 1
        elif self.drop_depth:
            return
        elif normalized in KNOWN_BLOCK_TAGS or normalized == "br":
            self.changed = True
            self.parts.append("\n")
        elif normalized in KNOWN_INLINE_TAGS or normalized in VOID_TAGS:
            self.changed = True
        else:
            self.parts.append(self.get_starttag_text())

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in DROP_CONTENT_TAGS:
            if self.drop_depth:
                self.drop_depth -= 1
        elif self.drop_depth:
            return
        elif normalized in KNOWN_BLOCK_TAGS:
            self.changed = True
            self.parts.append("\n")
        elif normalized in KNOWN_INLINE_TAGS or normalized in VOID_TAGS:
            self.changed = True
        else:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if not self.drop_depth:
            self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if not self.drop_depth:
            raw = f"&{name};"
            decoded = html_lib.unescape(raw)
            self.parts.append(decoded)
            self.changed |= decoded != raw

    def handle_charref(self, name: str) -> None:
        if not self.drop_depth:
            raw = f"&#{name};"
            decoded = html_lib.unescape(raw)
            self.parts.append(decoded)
            self.changed |= decoded != raw

    def handle_comment(self, data: str) -> None:
        self.changed = True

    def handle_decl(self, decl: str) -> None:
        self.changed = True


def strip_html(text: str) -> str:
    """Remove known HTML markup without deleting code-like angle brackets."""
    parser = _ConservativeHTMLStripper()
    parser.feed(text)
    parser.close()
    if not parser.changed:
        return text
    return "".join(parser.parts).strip()


def clean_controls(text: str) -> str:
    """Remove unwanted controls while preserving useful line and Unicode data."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    kept: list[str] = []
    for character in normalized:
        if character in UNWANTED_FORMAT_CHARACTERS:
            continue
        if unicodedata.category(character) == "Cc" and character not in {"\n", "\t"}:
            continue
        kept.append(character)
    return "".join(kept)


def normalize_whitespace(text: str) -> str:
    """Normalize line whitespace without flattening indentation or inner spaces."""
    lines = [line.expandtabs(4).rstrip() for line in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = is_blank
    return "\n".join(normalized)


def parse_alpaca_record(record: object, sample_id: str) -> CanonicalRecord:
    """Validate an Alpaca record and project it into canonical messages."""
    if not isinstance(record, dict):
        raise RecordRejected("invalid_record_type")

    for field_name in ("instruction", "input", "output"):
        value = record.get(field_name)
        if not isinstance(value, str):
            raise RecordRejected(f"invalid_{field_name}_type")

    instruction = record["instruction"]
    input_text = record["input"]
    output = record["output"]
    if not instruction.strip():
        raise RecordRejected("empty_instruction")
    if not output.strip():
        raise RecordRejected("empty_output")

    user_content = instruction
    if input_text.strip():
        user_content = f"{instruction}\n\n{input_text}"
    return CanonicalRecord(
        sample_id=sample_id,
        messages=[
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ],
        alpaca_instruction=instruction,
        alpaca_input=input_text,
    )


def parse_sharegpt_record(record: object, sample_id: str) -> CanonicalRecord:
    """Validate ShareGPT role order and return canonical role names."""
    if not isinstance(record, dict):
        raise RecordRejected("invalid_record_type")
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise RecordRejected("invalid_conversations_type")
    if not conversations:
        raise RecordRejected("empty_conversations")

    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }
    messages: list[dict[str, str]] = []
    expected_role = "user"
    for index, message in enumerate(conversations):
        if not isinstance(message, dict):
            raise RecordRejected("invalid_message_type")
        raw_role = message.get("from")
        if not isinstance(raw_role, str) or raw_role not in role_map:
            raise RecordRejected("invalid_role")
        content = message.get("value")
        if not isinstance(content, str):
            raise RecordRejected("invalid_message_content_type")
        if not content.strip():
            raise RecordRejected("empty_message")

        role = role_map[raw_role]
        if role == "system":
            if index != 0:
                raise RecordRejected("invalid_role_order")
            messages.append({"role": role, "content": content})
            continue
        if role != expected_role:
            raise RecordRejected("invalid_role_order")
        messages.append({"role": role, "content": content})
        expected_role = "assistant" if role == "user" else "user"

    non_system = [message for message in messages if message["role"] != "system"]
    if not non_system or non_system[-1]["role"] != "assistant":
        raise RecordRejected("missing_assistant")
    provenance = record.get("_provenance")
    return CanonicalRecord(
        sample_id=sample_id,
        messages=messages,
        provenance=provenance if isinstance(provenance, dict) else {},
    )


def clean_record_text(
    record: CanonicalRecord,
) -> tuple[CanonicalRecord, dict[str, bool]]:
    """Apply text rules to every message and report which stages changed it."""
    changes = {"html": False, "controls": False, "whitespace": False}
    messages: list[dict[str, str]] = []
    for message in record.messages:
        raw = message["content"]
        without_html = strip_html(raw)
        without_controls = clean_controls(without_html)
        normalized = normalize_whitespace(without_controls)
        changes["html"] |= without_html != raw
        changes["controls"] |= without_controls != without_html
        changes["whitespace"] |= normalized != without_controls
        if not normalized:
            raise RecordRejected("empty_after_cleaning")
        messages.append({"role": message["role"], "content": normalized})
    return replace(record, messages=messages), changes


def chat_token_length(messages: list[dict[str, str]], tokenizer: Any) -> int:
    """Return the exact training-template length visible to Qwen."""
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return len(token_ids)


def truncate_messages(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_tokens: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Structurally truncate a conversation to the model-visible token limit."""
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须为正整数")
    truncated = deepcopy(messages)
    original_length = chat_token_length(truncated, tokenizer)
    audit: dict[str, Any] = {
        "truncated": False,
        "original_length": original_length,
        "final_length": original_length,
        "removed_pairs": 0,
        "truncated_roles": [],
        "removed_content_tokens": 0,
    }
    if original_length <= max_tokens:
        return truncated, audit

    pair_start = 1 if truncated and truncated[0]["role"] == "system" else 0
    while (
        chat_token_length(truncated, tokenizer) > max_tokens
        and len(truncated) - pair_start >= 4
    ):
        del truncated[pair_start : pair_start + 2]
        audit["removed_pairs"] += 1

    while chat_token_length(truncated, tokenizer) > max_tokens:
        candidates: list[tuple[int, int, list[int]]] = []
        for index, message in enumerate(truncated):
            content_ids = tokenizer.encode(
                message["content"],
                add_special_tokens=False,
            )
            if len(content_ids) > 1:
                candidates.append((len(content_ids), index, content_ids))
        if not candidates:
            raise RecordRejected("cannot_truncate_to_limit")

        _, message_index, content_ids = max(candidates, key=lambda item: item[0])
        original_content_length = len(content_ids)
        low, high = 1, original_content_length - 1
        best_content: str | None = None
        best_kept_tokens = 0
        while low <= high:
            middle = (low + high) // 2
            candidate_content = tokenizer.decode(
                content_ids[:middle],
                skip_special_tokens=True,
            ).rstrip()
            if not candidate_content:
                low = middle + 1
                continue
            candidate_messages = deepcopy(truncated)
            candidate_messages[message_index]["content"] = candidate_content
            if chat_token_length(candidate_messages, tokenizer) <= max_tokens:
                best_content = candidate_content
                best_kept_tokens = middle
                low = middle + 1
            else:
                high = middle - 1

        if best_content is None:
            best_content = tokenizer.decode(
                content_ids[:1],
                skip_special_tokens=True,
            ).strip()
            if not best_content:
                raise RecordRejected("cannot_preserve_nonempty_message")
            best_kept_tokens = 1

        role = truncated[message_index]["role"]
        truncated[message_index]["content"] = best_content
        if role not in audit["truncated_roles"]:
            audit["truncated_roles"].append(role)
        audit["removed_content_tokens"] += (
            original_content_length - best_kept_tokens
        )

    audit["truncated"] = True
    audit["final_length"] = chat_token_length(truncated, tokenizer)
    return truncated, audit


def canonical_text(record: CanonicalRecord) -> str:
    """Project a record into the provenance-free text used for deduplication."""
    return "\n".join(
        f"{message['role']}:{message['content']}" for message in record.messages
    )


def exact_deduplicate(
    records: list[CanonicalRecord],
) -> tuple[list[CanonicalRecord], list[dict[str, Any]]]:
    """Keep the first canonical SHA-256 occurrence and map later duplicates."""
    kept: list[CanonicalRecord] = []
    seen: dict[str, CanonicalRecord] = {}
    duplicates: list[dict[str, Any]] = []
    for record in records:
        digest = hashlib.sha256(canonical_text(record).encode("utf-8")).hexdigest()
        keeper = seen.get(digest)
        if keeper is None:
            seen[digest] = record
            kept.append(record)
            continue
        duplicates.append(
            {
                "duplicate_id": record.sample_id,
                "kept_id": keeper.sample_id,
                "reason": "exact",
                "distance": 0,
            }
        )
    return kept, duplicates


def normalize_for_similarity(text: str) -> str:
    """Create the deterministic text surface used by character n-grams."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _character_ngram_set(text: str, ngram_size: int = 3) -> set[str]:
    normalized = normalize_for_similarity(text)
    if not normalized:
        return set()
    if len(normalized) < ngram_size:
        return {normalized}
    return {
        normalized[index : index + ngram_size]
        for index in range(len(normalized) - ngram_size + 1)
    }


def character_ngram_jaccard(
    left: str,
    right: str,
    ngram_size: int = 3,
) -> float:
    """Measure overlap of normalized character n-gram sets."""
    left_features = _character_ngram_set(left, ngram_size)
    right_features = _character_ngram_set(right, ngram_size)
    if not left_features and not right_features:
        return 1.0
    if not left_features or not right_features:
        return 0.0
    return len(left_features & right_features) / len(
        left_features | right_features
    )


def _role_text(record: CanonicalRecord, role: str) -> str:
    return "\n".join(
        message["content"]
        for message in record.messages
        if message["role"] == role
    )


def role_jaccard_scores(
    left: CanonicalRecord,
    right: CanonicalRecord,
) -> dict[str, float]:
    """Return overall and per-training-role similarity evidence."""
    overall = character_ngram_jaccard(canonical_text(left), canonical_text(right))
    user = character_ngram_jaccard(
        _role_text(left, "user"),
        _role_text(right, "user"),
    )
    assistant = character_ngram_jaccard(
        _role_text(left, "assistant"),
        _role_text(right, "assistant"),
    )
    return {
        "jaccard": overall,
        "user_jaccard": user,
        "assistant_jaccard": assistant,
        "min_role_jaccard": min(user, assistant),
    }


def simhash64(text: str, ngram_size: int = 3) -> int:
    """Compute a self-contained 64-bit SimHash from character n-grams."""
    normalized = normalize_for_similarity(text)
    if not normalized:
        return 0
    if len(normalized) < ngram_size:
        features = Counter({normalized: 1})
    else:
        features = Counter(
            normalized[index : index + ngram_size]
            for index in range(len(normalized) - ngram_size + 1)
        )

    vector = [0] * 64
    for feature, weight in features.items():
        feature_hash = int.from_bytes(
            hashlib.sha256(feature.encode("utf-8")).digest()[:8],
            "big",
        )
        for bit in range(64):
            vector[bit] += weight if feature_hash & (1 << bit) else -weight

    fingerprint = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(left: int, right: int) -> int:
    """Count differing bits between two integer fingerprints."""
    return (left ^ right).bit_count()


def banded_candidate_pairs(
    fingerprints: list[int],
    bands: int = 4,
) -> set[tuple[int, int]]:
    """Return index pairs sharing at least one equal SimHash band."""
    if bands <= 0 or 64 % bands:
        raise ValueError("bands 必须是 64 的正因数")
    bits_per_band = 64 // bands
    mask = (1 << bits_per_band) - 1
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for index, fingerprint in enumerate(fingerprints):
        for band in range(bands):
            band_value = (fingerprint >> (band * bits_per_band)) & mask
            key = (band, band_value)
            for other_index in buckets[key]:
                candidates.add((other_index, index))
            buckets[key].append(index)
    return candidates


def collect_simhash_candidates(
    records: list[CanonicalRecord],
    max_distance: int = 6,
    min_chars: int = 20,
    preview_chars: int = 240,
) -> list[dict[str, Any]]:
    """Export bounded, reviewable real-data candidates for threshold calibration."""
    eligible = [
        (index, canonical_text(record))
        for index, record in enumerate(records)
        if len(normalize_for_similarity(canonical_text(record))) >= min_chars
    ]
    fingerprints = [simhash64(text) for _, text in eligible]
    rows: list[dict[str, Any]] = []
    for left_local, right_local in sorted(banded_candidate_pairs(fingerprints)):
        distance = hamming_distance(
            fingerprints[left_local],
            fingerprints[right_local],
        )
        if distance > max_distance:
            continue
        left_index, left_text = eligible[left_local]
        right_index, right_text = eligible[right_local]
        left = records[left_index]
        right = records[right_index]
        role_scores = role_jaccard_scores(left, right)
        rows.append(
            {
                "left_id": left.sample_id,
                "right_id": right.sample_id,
                "left_source": left.provenance.get("source", ""),
                "right_source": right.provenance.get("source", ""),
                "distance": distance,
                **{
                    key: round(value, 6)
                    for key, value in role_scores.items()
                },
                "left_preview": left_text[:preview_chars],
                "right_preview": right_text[:preview_chars],
            }
        )
    rows.sort(
        key=lambda row: (
            row["distance"],
            row["left_id"],
            row["right_id"],
        )
    )
    return rows


def _representative_key(
    record: CanonicalRecord,
    position: int,
) -> tuple[Any, ...]:
    source = str(record.provenance.get("source", ""))
    raw_source_index = record.provenance.get("source_index", position)
    source_index = (
        raw_source_index if isinstance(raw_source_index, int) else position
    )
    return (
        record.was_truncated,
        -len(record.messages),
        -len(canonical_text(record)),
        source,
        source_index,
        record.sample_id,
        position,
    )


def fuzzy_deduplicate(
    records: list[CanonicalRecord],
    threshold: int,
    min_chars: int = 20,
    min_jaccard: float = 0.85,
    min_role_jaccard: float = 0.90,
) -> tuple[
    list[CanonicalRecord],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Cluster near duplicates using banded SimHash and deterministic keepers."""
    if threshold < 0 or threshold > 64:
        raise ValueError("threshold 必须在 0 到 64 之间")
    fingerprints = [simhash64(canonical_text(record)) for record in records]
    eligible_indices = [
        index
        for index, record in enumerate(records)
        if len(normalize_for_similarity(canonical_text(record))) >= min_chars
    ]
    eligible_fingerprints = [fingerprints[index] for index in eligible_indices]

    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left_local, right_local in sorted(
        banded_candidate_pairs(eligible_fingerprints)
    ):
        left = eligible_indices[left_local]
        right = eligible_indices[right_local]
        distance = hamming_distance(fingerprints[left], fingerprints[right])
        role_scores = role_jaccard_scores(records[left], records[right])
        if (
            distance <= threshold
            and role_scores["jaccard"] >= min_jaccard
            and role_scores["min_role_jaccard"] >= min_role_jaccard
        ):
            union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[find(index)].append(index)

    duplicate_indices: set[int] = set()
    duplicates: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    for member_indices in groups.values():
        if len(member_indices) < 2:
            continue
        keeper_index = min(
            member_indices,
            key=lambda index: _representative_key(records[index], index),
        )
        member_indices = sorted(member_indices)
        clusters.append(
            {
                "kept_id": records[keeper_index].sample_id,
                "member_ids": [
                    records[index].sample_id for index in member_indices
                ],
                "size": len(member_indices),
            }
        )
        for index in member_indices:
            if index == keeper_index:
                continue
            duplicate_indices.add(index)
            role_scores = role_jaccard_scores(
                records[index],
                records[keeper_index],
            )
            duplicates.append(
                {
                    "duplicate_id": records[index].sample_id,
                    "kept_id": records[keeper_index].sample_id,
                    "reason": "fuzzy",
                    "distance": hamming_distance(
                        fingerprints[index],
                        fingerprints[keeper_index],
                    ),
                    **{
                        key: round(value, 6)
                        for key, value in role_scores.items()
                    },
                }
            )

    duplicates.sort(
        key=lambda item: next(
            index
            for index, record in enumerate(records)
            if record.sample_id == item["duplicate_id"]
        )
    )
    clusters.sort(key=lambda item: item["kept_id"])
    kept = [
        record
        for index, record in enumerate(records)
        if index not in duplicate_indices
    ]
    return kept, duplicates, clusters


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone Day 7 command-line interface."""
    parser = argparse.ArgumentParser(
        description="Clean Day 6 data for Qwen2.5 SFT.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-alpaca", type=Path, required=True)
    parser.add_argument("--output-sharegpt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--simhash-threshold", type=int, default=3)
    parser.add_argument("--min-jaccard", type=float, default=0.85)
    parser.add_argument("--min-role-jaccard", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    return parser


def _read_required_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} 不是合法 JSON：{error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} 顶层必须是对象")
            records.append(record)
    return records


def _read_input_lines(
    path: Path,
) -> list[tuple[dict[str, Any] | None, str | None]]:
    records: list[tuple[dict[str, Any] | None, str | None]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                records.append((None, "invalid_json"))
                continue
            if not isinstance(record, dict):
                records.append((None, "invalid_record_type"))
                continue
            records.append((record, None))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=False,
                )
                + "\n"
            )


def _percentile(sorted_values: list[int], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _length_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "min": 0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": round(_percentile(ordered, 0.25), 3),
        "median": round(_percentile(ordered, 0.50), 3),
        "mean": round(sum(ordered) / len(ordered), 3),
        "p75": round(_percentile(ordered, 0.75), 3),
        "p90": round(_percentile(ordered, 0.90), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "p99": round(_percentile(ordered, 0.99), 3),
        "max": ordered[-1],
    }


def _render_sharegpt(record: CanonicalRecord) -> dict[str, Any]:
    role_map = {"user": "human", "assistant": "gpt", "system": "system"}
    return {
        "conversations": [
            {
                "from": role_map[message["role"]],
                "value": message["content"],
            }
            for message in record.messages
        ]
    }


def _render_alpaca(record: CanonicalRecord) -> dict[str, str]:
    assistant_index = len(record.messages) - 1
    user_index = assistant_index - 1
    if (
        user_index < 0
        or record.messages[user_index]["role"] != "user"
        or record.messages[assistant_index]["role"] != "assistant"
    ):
        raise RecordRejected("missing_final_training_pair")
    context = "\n\n".join(
        f"{message['role']}: {message['content']}"
        for message in record.messages[:user_index]
    )
    return {
        "instruction": record.messages[user_index]["content"],
        "input": context,
        "output": record.messages[assistant_index]["content"],
    }


def _write_duplicate_pairs(
    path: Path,
    duplicate_pairs: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "duplicate_id",
                "kept_id",
                "reason",
                "distance",
                "jaccard",
                "user_jaccard",
                "assistant_jaccard",
                "min_role_jaccard",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(duplicate_pairs)


def _write_calibration_csv(
    path: Path,
    candidates: list[dict[str, Any]],
    selected_threshold: int,
    selected_min_jaccard: float,
    selected_min_role_jaccard: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "left_id",
        "right_id",
        "left_source",
        "right_source",
        "distance",
        "jaccard",
        "user_jaccard",
        "assistant_jaccard",
        "min_role_jaccard",
        "selected_threshold",
        "selected_min_jaccard",
        "selected_min_role_jaccard",
        "decision",
        "label_source",
        "left_preview",
        "right_preview",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    **candidate,
                    "selected_threshold": selected_threshold,
                    "selected_min_jaccard": selected_min_jaccard,
                    "selected_min_role_jaccard": selected_min_role_jaccard,
                    "decision": (
                        "duplicate"
                        if candidate["distance"] <= selected_threshold
                        and float(candidate["jaccard"]) >= selected_min_jaccard
                        and float(candidate["min_role_jaccard"])
                        >= selected_min_role_jaccard
                        else "review_nonduplicate"
                    ),
                    "label_source": "threshold_candidate_review",
                }
            )


def _write_stats_csv(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["category", "metric", "value"])
        for metric, value in stats["counts"].items():
            writer.writerow(["counts", metric, value])
        for phase, summary in stats["lengths"].items():
            for metric, value in summary.items():
                writer.writerow([phase, metric, value])


def _plot_evidence(
    evidence_dir: Path,
    raw_lengths: list[int],
    clean_lengths: list[int],
    final_lengths: list[int],
    counts: dict[str, int],
    max_tokens: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evidence_dir.mkdir(parents=True, exist_ok=True)
    all_lengths = raw_lengths + clean_lengths + final_lengths
    ordered = sorted(all_lengths)
    p99 = _percentile(ordered, 0.99) if ordered else max_tokens
    x_limit = max(max_tokens * 1.15, p99 * 1.05, 1)

    figure, axis = plt.subplots(figsize=(11, 6.5))
    for values, label, color in (
        (raw_lengths, "Before cleaning", "#8c8c8c"),
        (clean_lengths, "After cleaning, before truncation", "#d97706"),
        (final_lengths, "Final retained data", "#1677ff"),
    ):
        if values:
            axis.hist(
                [min(value, x_limit) for value in values],
                bins=60,
                histtype="step",
                linewidth=1.8,
                label=f"{label} (n={len(values)})",
                color=color,
            )
    axis.axvline(
        max_tokens,
        color="#cf1322",
        linestyle="--",
        linewidth=1.6,
        label=f"cutoff = {max_tokens}",
    )
    axis.set_title("Qwen Chat-Template Token Length Distribution")
    axis.set_xlabel("Tokens per sample (values above P99 clipped to right edge)")
    axis.set_ylabel("Number of samples")
    axis.set_xlim(0, x_limit)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        evidence_dir / "length_distribution.png",
        dpi=160,
        metadata={"Software": "matplotlib"},
    )
    plt.close(figure)

    stage_names = [
        "Raw",
        "After schema/empty",
        "After post-clean empty",
        "After exact dedup",
        "Final",
    ]
    stage_values = [
        counts["raw_total"],
        counts["after_initial_filter"],
        counts["after_postclean_filter"],
        counts["after_exact_dedup"],
        counts["final_total"],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].bar(stage_names, stage_values, color="#1677ff")
    axes[0].set_title("Records Remaining by Stage")
    axes[0].set_ylabel("Samples")
    axes[0].tick_params(axis="x", rotation=28)
    for index, value in enumerate(stage_values):
        axes[0].text(index, value, str(value), ha="center", va="bottom")

    removal_names = [
        "Structure",
        "Initial empty",
        "Post-clean empty",
        "Exact duplicate",
        "Fuzzy duplicate",
    ]
    removal_values = [
        counts["structure_errors_removed"],
        counts["initial_empty_removed"],
        counts["postclean_empty_removed"],
        counts["exact_duplicates_removed"],
        counts["fuzzy_duplicates_removed"],
    ]
    axes[1].bar(removal_names, removal_values, color="#d97706")
    axes[1].set_title("Records Removed by Reason")
    axes[1].set_ylabel("Samples")
    axes[1].tick_params(axis="x", rotation=28)
    for index, value in enumerate(removal_values):
        axes[1].text(index, value, str(value), ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(
        evidence_dir / "cleaning_counts.png",
        dpi=160,
        metadata={"Software": "matplotlib"},
    )
    plt.close(figure)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed-order Day 7 pipeline and write all core artifacts."""
    from transformers import AutoTokenizer

    input_lines = _read_input_lines(args.input)
    provenance_records = _read_required_jsonl(args.provenance)
    if len(input_lines) != len(provenance_records):
        raise ValueError(
            "input 与 provenance 行数不一致："
            f"{len(input_lines)} != {len(provenance_records)}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
    )

    counts = {
        "raw_total": len(input_lines),
        "structure_errors_removed": 0,
        "initial_empty_removed": 0,
        "html_modified": 0,
        "controls_modified": 0,
        "whitespace_modified": 0,
        "postclean_empty_removed": 0,
        "overlength_total": 0,
        "truncated_total": 0,
        "exact_duplicates_removed": 0,
        "fuzzy_duplicates_removed": 0,
        "after_initial_filter": 0,
        "after_postclean_filter": 0,
        "after_exact_dedup": 0,
        "final_total": 0,
    }
    raw_lengths: list[int] = []
    clean_lengths: list[int] = []
    final_lengths_by_id: dict[str, int] = {}
    records: list[CanonicalRecord] = []
    audits: list[dict[str, Any]] = []
    audit_by_id: dict[str, dict[str, Any]] = {}
    initial_empty_reasons = {
        "empty_conversations",
        "empty_message",
        "missing_assistant",
    }

    for line_index, ((raw_record, parse_error), provenance) in enumerate(
        zip(input_lines, provenance_records),
        start=1,
    ):
        sample_id = str(provenance.get("sample_id", f"line-{line_index}"))
        audit: dict[str, Any] = {
            "line_number": line_index,
            "sample_id": sample_id,
            "source": provenance.get("source", ""),
            "source_index": provenance.get("source_index", ""),
            "status": "pending",
            "reason": "",
            "html_modified": False,
            "controls_modified": False,
            "whitespace_modified": False,
            "raw_token_length": None,
            "clean_token_length": None,
            "final_token_length": None,
            "truncated": False,
            "removed_pairs": 0,
            "truncated_roles": [],
            "removed_content_tokens": 0,
            "duplicate_of": "",
            "duplicate_distance": None,
        }
        audits.append(audit)
        audit_by_id[sample_id] = audit
        if parse_error or raw_record is None:
            counts["structure_errors_removed"] += 1
            audit["status"] = "removed"
            audit["reason"] = parse_error
            continue

        try:
            parsed = parse_sharegpt_record(raw_record, sample_id=sample_id)
        except RecordRejected as error:
            if error.reason in initial_empty_reasons:
                counts["initial_empty_removed"] += 1
            else:
                counts["structure_errors_removed"] += 1
            audit["status"] = "removed"
            audit["reason"] = error.reason
            continue
        parsed = replace(parsed, provenance=provenance)
        raw_length = chat_token_length(parsed.messages, tokenizer)
        raw_lengths.append(raw_length)
        audit["raw_token_length"] = raw_length
        counts["after_initial_filter"] += 1

        try:
            cleaned, changes = clean_record_text(parsed)
        except RecordRejected as error:
            counts["postclean_empty_removed"] += 1
            audit["status"] = "removed"
            audit["reason"] = error.reason
            continue
        for change_name, changed in changes.items():
            audit[f"{change_name}_modified"] = changed
            if changed:
                counts[f"{change_name}_modified"] += 1

        clean_length = chat_token_length(cleaned.messages, tokenizer)
        clean_lengths.append(clean_length)
        audit["clean_token_length"] = clean_length
        if clean_length > args.max_tokens:
            counts["overlength_total"] += 1

        truncated_messages, truncation = truncate_messages(
            cleaned.messages,
            tokenizer,
            max_tokens=args.max_tokens,
        )
        cleaned = replace(
            cleaned,
            messages=truncated_messages,
            was_truncated=truncation["truncated"],
        )
        if truncation["truncated"]:
            counts["truncated_total"] += 1
        audit.update(
            {
                "final_token_length": truncation["final_length"],
                "truncated": truncation["truncated"],
                "removed_pairs": truncation["removed_pairs"],
                "truncated_roles": truncation["truncated_roles"],
                "removed_content_tokens": truncation["removed_content_tokens"],
            }
        )
        final_lengths_by_id[sample_id] = truncation["final_length"]
        counts["after_postclean_filter"] += 1
        records.append(cleaned)

    exact_kept, exact_duplicates = exact_deduplicate(records)
    counts["exact_duplicates_removed"] = len(exact_duplicates)
    counts["after_exact_dedup"] = len(exact_kept)
    calibration_candidates = collect_simhash_candidates(
        exact_kept,
        max_distance=6,
        min_chars=20,
        preview_chars=240,
    )
    _write_calibration_csv(
        args.results_dir / "simhash_calibration.csv",
        calibration_candidates,
        selected_threshold=args.simhash_threshold,
        selected_min_jaccard=args.min_jaccard,
        selected_min_role_jaccard=args.min_role_jaccard,
    )
    fuzzy_kept, fuzzy_duplicates, clusters = fuzzy_deduplicate(
        exact_kept,
        threshold=args.simhash_threshold,
        min_chars=20,
        min_jaccard=args.min_jaccard,
        min_role_jaccard=args.min_role_jaccard,
    )
    counts["fuzzy_duplicates_removed"] = len(fuzzy_duplicates)
    counts["final_total"] = len(fuzzy_kept)

    all_duplicates = exact_duplicates + fuzzy_duplicates
    for duplicate in all_duplicates:
        audit = audit_by_id[duplicate["duplicate_id"]]
        audit["status"] = "removed"
        audit["reason"] = f"{duplicate['reason']}_duplicate"
        audit["duplicate_of"] = duplicate["kept_id"]
        audit["duplicate_distance"] = duplicate["distance"]
    for record in fuzzy_kept:
        audit_by_id[record.sample_id]["status"] = "kept"
        audit_by_id[record.sample_id]["reason"] = ""

    output_alpaca = [_render_alpaca(record) for record in fuzzy_kept]
    output_sharegpt = [_render_sharegpt(record) for record in fuzzy_kept]
    _write_jsonl(args.output_alpaca, output_alpaca)
    _write_jsonl(args.output_sharegpt, output_sharegpt)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.results_dir / "cleaning_audit.jsonl", audits)
    cleaned_provenance = []
    for record in fuzzy_kept:
        item = dict(record.provenance)
        item["sample_id"] = record.sample_id
        item["was_truncated"] = record.was_truncated
        item["final_token_length"] = final_lengths_by_id[record.sample_id]
        cleaned_provenance.append(item)
    _write_jsonl(
        args.results_dir / "cleaned_provenance.jsonl",
        cleaned_provenance,
    )
    _write_duplicate_pairs(
        args.results_dir / "duplicate_pairs.csv",
        all_duplicates,
    )
    _write_jsonl(args.results_dir / "dedup_clusters.jsonl", clusters)

    final_lengths = [
        final_lengths_by_id[record.sample_id] for record in fuzzy_kept
    ]
    stats = {
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "simhash": {
            "bits": 64,
            "ngram_size": 3,
            "bands": 4,
            "threshold": args.simhash_threshold,
            "min_chars": 20,
            "min_jaccard": args.min_jaccard,
            "min_role_jaccard": args.min_role_jaccard,
        },
        "counts": counts,
        "lengths": {
            "before_cleaning": _length_summary(raw_lengths),
            "after_cleaning_before_truncation": _length_summary(clean_lengths),
            "final_retained": _length_summary(final_lengths),
        },
    }
    (args.results_dir / "cleaning_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_stats_csv(args.results_dir / "cleaning_stats.csv", stats)

    count_balance = (
        counts["raw_total"]
        == counts["structure_errors_removed"]
        + counts["initial_empty_removed"]
        + counts["postclean_empty_removed"]
        + counts["exact_duplicates_removed"]
        + counts["fuzzy_duplicates_removed"]
        + counts["final_total"]
    )
    validation_errors: list[str] = []
    if not count_balance:
        validation_errors.append("count_balance_failed")
    if len(output_alpaca) != len(output_sharegpt):
        validation_errors.append("dual_format_count_mismatch")
    if final_lengths and max(final_lengths) > args.max_tokens:
        validation_errors.append("token_limit_exceeded")
    validation = {
        "valid": not validation_errors,
        "errors": validation_errors,
        "input_count": counts["raw_total"],
        "final_count": counts["final_total"],
        "aligned_output_counts": len(output_alpaca) == len(output_sharegpt),
        "max_final_tokens": max(final_lengths, default=0),
        "max_tokens": args.max_tokens,
        "count_balance": count_balance,
    }
    (args.results_dir / "day7_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_evidence(
        args.evidence_dir,
        raw_lengths,
        clean_lengths,
        final_lengths,
        counts,
        args.max_tokens,
    )
    if validation_errors:
        raise ValueError("最终验证失败：" + ", ".join(validation_errors))
    return stats


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the pipeline and return a shell exit code."""
    args = build_parser().parse_args(argv)
    try:
        stats = run_pipeline(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "input": stats["counts"]["raw_total"],
                "output": stats["counts"]["final_total"],
                "truncated": stats["counts"]["truncated_total"],
                "exact_duplicates": stats["counts"][
                    "exact_duplicates_removed"
                ],
                "fuzzy_duplicates": stats["counts"][
                    "fuzzy_duplicates_removed"
                ],
                "output_alpaca": str(args.output_alpaca),
                "output_sharegpt": str(args.output_sharegpt),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
