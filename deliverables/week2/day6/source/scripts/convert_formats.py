#!/usr/bin/env python3
"""Convert the approved Day 6 raw subsets into training-visible formats."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_FILES = (
    ("alpaca_gpt4_zh", "alpaca_gpt4_zh_2k.jsonl", "alpaca"),
    ("coig_pc", "coig_pc_2k.jsonl", "alpaca"),
    ("sharegpt_zh", "sharegpt_zh_1k.jsonl", "sharegpt"),
)


def alpaca_to_alpaca(record: dict[str, Any]) -> dict[str, str]:
    """Return only the three standard Alpaca training fields."""
    return {
        "instruction": record["instruction"],
        "input": record.get("input", ""),
        "output": record["output"],
    }


def alpaca_to_sharegpt(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Convert one Alpaca record to a single-turn ShareGPT conversation."""
    instruction = record["instruction"]
    input_text = record.get("input", "")
    human_text = f"{instruction}\n\n{input_text}" if input_text else instruction
    return {
        "conversations": [
            {"from": "human", "value": human_text},
            {"from": "gpt", "value": record["output"]},
        ]
    }


def normalize_sharegpt(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Normalize common ShareGPT role aliases without altering message text."""
    role_map = {
        "human": "human",
        "user": "human",
        "gpt": "gpt",
        "assistant": "gpt",
    }
    raw_messages = record.get("conversations")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("ShareGPT conversations must contain complete message pairs")
    if len(raw_messages) % 2:
        tail = raw_messages[-1]
        tail_role = (
            role_map.get(tail.get("from")) if isinstance(tail, dict) else None
        )
        if len(raw_messages) == 1 or tail_role != "human":
            raise ValueError(
                "ShareGPT conversations must contain complete message pairs"
            )
        raw_messages = raw_messages[:-1]

    conversations = []
    for index, message in enumerate(raw_messages):
        role = role_map.get(message.get("from"))
        expected_role = "human" if index % 2 == 0 else "gpt"
        if role != expected_role:
            raise ValueError(
                f"message {index} has role {message.get('from')!r}; "
                f"expected {expected_role!r}"
            )
        value = message.get("value")
        if not isinstance(value, str):
            raise ValueError(f"message {index} value must be a string")
        conversations.append({"from": role, "value": value})
    return {"conversations": conversations}


def sharegpt_to_alpaca(record: dict[str, Any]) -> dict[str, str]:
    """Use the final complete turn as target and serialize earlier turns."""
    messages = normalize_sharegpt(record)["conversations"]
    history = "\n".join(
        f"[{message['from']}]\n{message['value']}" for message in messages[:-2]
    )
    return {
        "instruction": messages[-2]["value"],
        "input": history,
        "output": messages[-1]["value"],
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def _write_json_line(handle, value: Any) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def convert_raw_subsets(
    raw_dir: Path, formatted_dir: Path, provenance_file: Path
) -> dict[str, int]:
    """Convert raw subsets in fixed source order and write aligned outputs."""
    raw_dir = Path(raw_dir)
    formatted_dir = Path(formatted_dir)
    provenance_file = Path(provenance_file)
    formatted_dir.mkdir(parents=True, exist_ok=True)
    provenance_file.parent.mkdir(parents=True, exist_ok=True)
    alpaca_path = formatted_dir / "week2_5k_alpaca.jsonl"
    sharegpt_path = formatted_dir / "week2_5k_sharegpt.jsonl"
    counts: dict[str, int] = {}

    with (
        alpaca_path.open("w", encoding="utf-8") as alpaca_handle,
        sharegpt_path.open("w", encoding="utf-8") as sharegpt_handle,
        provenance_file.open("w", encoding="utf-8") as provenance_handle,
    ):
        for source_name, filename, source_format in SOURCE_FILES:
            count = 0
            for record in _read_jsonl(raw_dir / filename):
                raw_provenance = record.get("_provenance")
                if not isinstance(raw_provenance, dict):
                    raise ValueError(f"{filename}: missing _provenance")
                if raw_provenance.get("source") != source_name:
                    raise ValueError(f"{filename}: provenance source mismatch")
                source_index = raw_provenance.get("source_index")
                revision = raw_provenance.get("revision")
                if not isinstance(source_index, int) or not isinstance(revision, str):
                    raise ValueError(f"{filename}: invalid provenance fields")

                visible_record = {
                    key: value for key, value in record.items() if key != "_provenance"
                }
                if source_format == "alpaca":
                    alpaca_record = alpaca_to_alpaca(record)
                    sharegpt_record = alpaca_to_sharegpt(record)
                    format_adjustments: list[str] = []
                else:
                    raw_messages = record.get("conversations")
                    format_adjustments = (
                        ["dropped_trailing_unanswered_human"]
                        if isinstance(raw_messages, list) and len(raw_messages) % 2
                        else []
                    )
                    alpaca_record = sharegpt_to_alpaca(record)
                    sharegpt_record = normalize_sharegpt(record)

                sample_key = f"{source_name}:{source_index}:{revision}"
                provenance = {
                    "sample_id": hashlib.sha256(sample_key.encode("utf-8")).hexdigest(),
                    "source": source_name,
                    "source_index": source_index,
                    "revision": revision,
                    "raw_record_sha256": _canonical_sha256(visible_record),
                    "format_adjustments": format_adjustments,
                }
                _write_json_line(alpaca_handle, alpaca_record)
                _write_json_line(sharegpt_handle, sharegpt_record)
                _write_json_line(provenance_handle, provenance)
                count += 1
            counts[source_name] = count

    counts["total"] = sum(counts.values())
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--formatted-dir", type=Path, required=True)
    parser.add_argument("--provenance-file", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = convert_raw_subsets(
        raw_dir=args.raw_dir,
        formatted_dir=args.formatted_dir,
        provenance_file=args.provenance_file,
    )
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
