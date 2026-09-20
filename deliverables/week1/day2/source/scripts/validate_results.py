from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def extract_python_block(text: str) -> str:
    match = re.search(
        r"```python\s*\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise ValueError("python code block not found")
    return match.group(1)


def validate_result_rows(rows: list[dict], expected_ids: set[str]) -> None:
    ids = [row.get("id") for row in rows]
    if (
        len(rows) != len(expected_ids)
        or set(ids) != expected_ids
        or len(ids) != len(set(ids))
    ):
        raise ValueError(f"result ids do not match: {ids}")
    for row in rows:
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            raise ValueError(f"empty response: {row.get('id')}")
        for key in (
            "messages",
            "chat_template_text",
            "generation",
            "input_tokens",
            "output_tokens",
            "elapsed_seconds",
        ):
            if key not in row:
                raise ValueError(f"missing {key}: {row.get('id')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--code-output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_result_rows(
        rows,
        {"code_generation", "logic_reasoning", "role_play"},
    )
    code_row = next(row for row in rows if row["id"] == "code_generation")
    args.code_output.parent.mkdir(parents=True, exist_ok=True)
    args.code_output.write_text(
        extract_python_block(code_row["response"]),
        encoding="utf-8",
    )
    print("PASS 3 result rows")
    print(args.code_output)


if __name__ == "__main__":
    main()
