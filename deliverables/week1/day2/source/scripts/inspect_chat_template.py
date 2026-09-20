from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    messages = [
        {"role": "system", "content": "你是一名严谨的助教。"},
        {"role": "user", "content": "用一句话解释 GQA。"},
    ]
    report = {"messages": messages}
    for flag in (False, True):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=flag,
        )
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=flag,
        )
        report[str(flag).lower()] = {
            "text": text,
            "ids": ids,
            "tokens": tokenizer.convert_ids_to_tokens(ids),
        }
    report["special_tokens"] = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
