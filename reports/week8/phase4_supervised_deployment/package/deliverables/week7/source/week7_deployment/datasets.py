"""Deterministically freeze mutually exclusive quantization datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .hashing import sha256_file, stable_record_sha256


class DatasetContractError(ValueError):
    """Raised when a source cannot satisfy the frozen Week 7 protocol."""


@dataclass(frozen=True)
class FrozenDatasets:
    calibration: tuple[dict[str, object], ...]
    perplexity: tuple[dict[str, object], ...]
    seed: int
    calibration_sha256: str | None = None
    perplexity_sha256: str | None = None

    @property
    def calibration_ids(self) -> tuple[str, ...]:
        return tuple(str(row["source_id"]) for row in self.calibration)

    @property
    def perplexity_ids(self) -> tuple[str, ...]:
        return tuple(str(row["source_id"]) for row in self.perplexity)


def _normalize_sharegpt_row(
    raw: Mapping[str, object], source_line: int
) -> dict[str, object]:
    conversations = raw.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise DatasetContractError(
            f"source line {source_line} must contain alternating human/gpt messages"
        )

    messages: list[dict[str, str]] = []
    expected = "human"
    role_map = {"human": "user", "gpt": "assistant"}
    for message in conversations:
        if not isinstance(message, Mapping):
            raise DatasetContractError(
                f"source line {source_line} must contain alternating human/gpt messages"
            )
        role = message.get("from")
        value = message.get("value")
        if role != expected or not isinstance(value, str) or not value.strip():
            raise DatasetContractError(
                f"source line {source_line} must contain alternating human/gpt messages"
            )
        messages.append({"role": role_map[expected], "content": value})
        expected = "gpt" if expected == "human" else "human"

    if expected != "human":
        raise DatasetContractError(
            f"source line {source_line} must contain alternating human/gpt messages"
        )

    canonical = {"messages": messages}
    source_id = stable_record_sha256(canonical)
    return {
        "source_id": source_id,
        "source_line": source_line,
        "messages": messages,
    }


def _selection_key(seed: int, source_id: str) -> str:
    return hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()


def freeze_rows(
    rows: Sequence[Mapping[str, object]],
    calibration_count: int,
    perplexity_count: int,
    seed: int,
) -> FrozenDatasets:
    """Normalize, deduplicate and deterministically split ShareGPT rows."""
    if calibration_count <= 0 or perplexity_count <= 0:
        raise DatasetContractError("dataset counts must be positive")

    unique: dict[str, dict[str, object]] = {}
    for source_line, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise DatasetContractError(f"source line {source_line} must be an object")
        normalized = _normalize_sharegpt_row(raw, source_line)
        unique.setdefault(str(normalized["source_id"]), normalized)

    required = calibration_count + perplexity_count
    if len(unique) < required:
        raise DatasetContractError(
            f"need at least {required} unique rows; received {len(unique)}"
        )

    ordered = sorted(
        unique.values(),
        key=lambda row: _selection_key(seed, str(row["source_id"])),
    )
    calibration = tuple(ordered[:calibration_count])
    perplexity = tuple(ordered[calibration_count:required])
    return FrozenDatasets(
        calibration=calibration,
        perplexity=perplexity,
        seed=seed,
    )


def freeze_generation_prompts(
    safety_payload: Mapping[str, object],
    business_payload: Mapping[str, object],
    general_payload: Mapping[str, object],
) -> tuple[dict[str, str], ...]:
    """Freeze 10 safety, 5 business and 5 general generation prompts."""
    safety = safety_payload.get("prompts")
    business = business_payload.get("prompts")
    general = general_payload.get("questions")
    if not isinstance(safety, list) or len(safety) < 10:
        raise DatasetContractError("generation eval requires 10 safety prompts")
    if not isinstance(business, list) or len(business) < 5:
        raise DatasetContractError("generation eval requires 5 business prompts")
    if not isinstance(general, list) or len(general) < 5:
        raise DatasetContractError("generation eval requires 5 general prompts")

    rows: list[dict[str, str]] = []
    for category, selected in (
        ("safety", safety[:10]),
        ("business", business[:5]),
    ):
        for raw in selected:
            if not isinstance(raw, Mapping):
                raise DatasetContractError(f"{category} prompt must be an object")
            source_id = raw.get("id")
            prompt = raw.get("text")
            if not isinstance(source_id, str) or not isinstance(prompt, str):
                raise DatasetContractError(
                    f"{category} prompt requires string id and text"
                )
            rows.append(
                {
                    "id": f"w7-gen-{category}-{len(rows) + 1:02d}",
                    "category": category,
                    "source_id": source_id,
                    "prompt": prompt,
                }
            )

    for raw in general[:5]:
        if not isinstance(raw, Mapping):
            raise DatasetContractError("general prompt must be an object")
        source_id = raw.get("id")
        messages = raw.get("messages")
        if (
            not isinstance(source_id, str)
            or not isinstance(messages, list)
            or not messages
            or not isinstance(messages[0], Mapping)
            or messages[0].get("role") != "user"
            or not isinstance(messages[0].get("content"), str)
        ):
            raise DatasetContractError(
                "general prompt requires an id and first user message"
            )
        rows.append(
            {
                "id": f"w7-gen-general-{len(rows) - 14:02d}",
                "category": "general",
                "source_id": source_id,
                "prompt": str(messages[0]["content"]),
            }
        )

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise DatasetContractError("generation prompt ids must be unique")
    return tuple(rows)


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetContractError(
                    f"invalid JSON at source line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, Mapping):
                raise DatasetContractError(
                    f"source line {line_number} must be an object"
                )
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8")


def freeze_quantization_datasets(
    *,
    source: Path,
    calibration_output: Path,
    perplexity_output: Path,
    manifest_output: Path,
    calibration_count: int = 128,
    perplexity_count: int = 256,
    seed: int = 42,
) -> FrozenDatasets:
    """Freeze files and bind them to a machine-readable manifest."""
    if not source.is_file():
        raise FileNotFoundError(source)
    frozen = freeze_rows(
        _read_jsonl(source),
        calibration_count=calibration_count,
        perplexity_count=perplexity_count,
        seed=seed,
    )
    _write_jsonl(calibration_output, frozen.calibration)
    _write_jsonl(perplexity_output, frozen.perplexity)

    calibration_sha256 = sha256_file(calibration_output)
    perplexity_sha256 = sha256_file(perplexity_output)
    overlap = set(frozen.calibration_ids) & set(frozen.perplexity_ids)
    manifest = {
        "schema_version": "1.0",
        "protocol": "week7_quantization_dataset_v1",
        "seed": seed,
        "source": {
            "path": source.as_posix(),
            "sha256": sha256_file(source),
            "unique_row_count": len(
                {row["source_id"] for row in frozen.calibration + frozen.perplexity}
            ),
        },
        "calibration": {
            "path": calibration_output.as_posix(),
            "row_count": len(frozen.calibration),
            "sha256": calibration_sha256,
            "max_tokens": 1024,
        },
        "perplexity": {
            "path": perplexity_output.as_posix(),
            "row_count": len(frozen.perplexity),
            "sha256": perplexity_sha256,
            "window_tokens": 512,
        },
        "overlap_count": len(overlap),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FrozenDatasets(
        calibration=frozen.calibration,
        perplexity=frozen.perplexity,
        seed=seed,
        calibration_sha256=calibration_sha256,
        perplexity_sha256=perplexity_sha256,
    )
