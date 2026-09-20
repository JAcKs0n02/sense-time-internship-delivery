#!/usr/bin/env python3
"""Render one cleaned Alpaca sample through LLaMA-Factory's Qwen template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IGNORE_INDEX = -100


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    required = ("instruction", "input", "output")
    if set(record) != set(required):
        raise ValueError(
            f"expected exactly {required}, got {tuple(record.keys())}"
        )
    if not all(isinstance(record[key], str) for key in required):
        raise TypeError("instruction, input and output must all be strings")
    query_parts = [
        record["instruction"].strip(),
        record["input"].strip(),
    ]
    user_content = "\n".join(part for part in query_parts if part)
    assistant_content = record["output"].strip()
    if not user_content or not assistant_content:
        raise ValueError("user and assistant content must be non-empty")
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def build_sft_labels(
    prompt_ids: list[int],
    response_ids: list[int],
) -> list[int]:
    return [IGNORE_INDEX] * len(prompt_ids) + response_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview actual Qwen template tokens and SFT label masking."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from transformers import AutoTokenizer

    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments

    with args.dataset.open("r", encoding="utf-8") as file:
        first_line = next(line for line in file if line.strip())
    record = json.loads(first_line)
    messages = build_messages(record)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    data_args = DataArguments(template=args.template)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    prompt_ids, response_ids = template.encode_oneturn(
        tokenizer,
        messages,
        system="",
        tools="",
    )
    labels = build_sft_labels(prompt_ids, response_ids)
    total_tokens = len(prompt_ids) + len(response_ids)
    if total_tokens > args.cutoff_len:
        raise ValueError(
            f"sample has {total_tokens} tokens, exceeding {args.cutoff_len}"
        )
    if any(label != IGNORE_INDEX for label in labels[: len(prompt_ids)]):
        raise AssertionError("prompt labels are not fully masked")
    if labels[len(prompt_ids) :] != response_ids:
        raise AssertionError("assistant labels do not match response token ids")

    report = {
        "dataset": str(args.dataset),
        "sample_index": 0,
        "source_keys": list(record.keys()),
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "message_roles": [message["role"] for message in messages],
        "user_content": messages[0]["content"],
        "assistant_content": messages[1]["content"],
        "prompt_token_count": len(prompt_ids),
        "response_token_count": len(response_ids),
        "total_token_count": total_tokens,
        "ignored_prompt_label_count": sum(
            label == IGNORE_INDEX for label in labels
        ),
        "supervised_assistant_label_count": sum(
            label != IGNORE_INDEX for label in labels
        ),
        "prompt_text": tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
        ),
        "response_text": tokenizer.decode(
            response_ids,
            skip_special_tokens=False,
        ),
        "provenance_field_present": "provenance" in record,
        "checks": {
            "roles_are_user_then_assistant": (
                [message["role"] for message in messages]
                == ["user", "assistant"]
            ),
            "within_cutoff": total_tokens <= args.cutoff_len,
            "prompt_labels_all_masked": True,
            "assistant_labels_supervised": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
