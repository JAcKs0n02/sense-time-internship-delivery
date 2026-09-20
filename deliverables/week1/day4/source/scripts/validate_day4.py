#!/usr/bin/env python3
"""Validate executed Day 4 notebook and raw result artifacts without ML deps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_CASE_IDS = {
    "pure_chinese",
    "pure_english",
    "mixed_language_version",
    "emoji",
    "rare_cjk",
    "unicode_composition",
    "fullwidth_halfwidth",
    "whitespace_controls",
    "python_code",
    "json_url_escape",
    "latex_math",
    "literal_special_tokens",
    "repeated_hanzi",
    "zero_width",
    "long_chinese",
}
EXPECTED_SPECIAL_TOKENS = {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}


def _check(condition: bool, message: str, checks: list[str], errors: list[str]) -> None:
    (checks if condition else errors).append(message)


def validate(day4_root: Path, notebook_path: Path) -> dict:
    day4_root = day4_root.resolve()
    notebook_path = notebook_path.resolve()
    results_dir = day4_root / "results" if (day4_root / "results").is_dir() else day4_root / "source" / "results"
    checks: list[str] = []
    errors: list[str] = []

    _check(notebook_path.is_file(), "executed notebook exists", checks, errors)
    if notebook_path.is_file():
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
        _check(bool(code_cells), "notebook contains code cells", checks, errors)
        _check(
            all(cell.get("execution_count") is not None for cell in code_cells),
            "all code cells have execution counts",
            checks,
            errors,
        )
        error_outputs = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        _check(not error_outputs, "notebook contains no error output", checks, errors)

    extreme_path = results_dir / "tokenizer_extreme_cases.csv"
    _check(extreme_path.is_file(), "extreme case CSV exists", checks, errors)
    if extreme_path.is_file():
        with extreme_path.open(encoding="utf-8", newline="") as handle:
            extreme_rows = list(csv.DictReader(handle))
        ids = {row.get("case_id") for row in extreme_rows}
        _check(len(extreme_rows) == 15, "extreme case CSV has exactly 15 rows", checks, errors)
        _check(ids == EXPECTED_CASE_IDS, "extreme case IDs match fixed specification", checks, errors)

    special_path = results_dir / "special_token_results.json"
    _check(special_path.is_file(), "special token JSON exists", checks, errors)
    if special_path.is_file():
        special = json.loads(special_path.read_text(encoding="utf-8"))
        _check(
            set(special.get("tokens", {})) == EXPECTED_SPECIAL_TOKENS,
            "three required Qwen special tokens are recorded",
            checks,
            errors,
        )
        chat = special.get("chat_template", {})
        _check(
            {"add_generation_prompt_true", "add_generation_prompt_false"}.issubset(chat),
            "both add_generation_prompt settings are recorded",
            checks,
            errors,
        )

    truncation_path = results_dir / "truncation_results.json"
    _check(truncation_path.is_file(), "truncation JSON exists", checks, errors)
    if truncation_path.is_file():
        truncation = json.loads(truncation_path.read_text(encoding="utf-8"))
        _check(truncation.get("max_length") == 64, "truncation max_length is 64", checks, errors)
        runs = truncation.get("runs")
        if runs is None:
            _check(truncation.get("after_tokens") == 64, "synthetic truncation count is 64", checks, errors)
        else:
            _check(set(runs) == {"left", "right"}, "left and right truncation are recorded", checks, errors)
            _check(
                all(run.get("after_tokens") == 64 for run in runs.values()),
                "both truncation runs contain 64 tokens",
                checks,
                errors,
            )

    comparison_path = results_dir / "tokenizer_comparison.csv"
    _check(comparison_path.is_file(), "tokenizer comparison CSV exists", checks, errors)
    if comparison_path.is_file():
        with comparison_path.open(encoding="utf-8", newline="") as handle:
            comparison_rows = list(csv.DictReader(handle))
        names = {row.get("tokenizer") for row in comparison_rows}
        _check(
            names == {"ByteLevel BPE", "SentencePiece Unigram"},
            "comparison includes both declared tokenizer algorithms",
            checks,
            errors,
        )
        _check(
            all(row.get("target_vocab_size") == "800" for row in comparison_rows),
            "all comparison rows use target vocab size 800",
            checks,
            errors,
        )
        _check(
            all(row.get("actual_vocab_size") == "800" for row in comparison_rows),
            "both trained tokenizers have actual vocab size 800",
            checks,
            errors,
        )

    environment_path = results_dir / "day4_environment.txt"
    _check(environment_path.is_file(), "environment version log exists", checks, errors)
    if environment_path.is_file():
        environment_text = environment_path.read_text(encoding="utf-8")
        _check(
            all(name in environment_text for name in ("Python", "transformers", "tokenizers", "sentencepiece", "pandas")),
            "environment log contains all required package versions",
            checks,
            errors,
        )

    return {
        "ok": not errors,
        "day4_root": str(day4_root),
        "notebook": str(notebook_path),
        "results_dir": str(results_dir),
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day4-root", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.day4_root, args.notebook)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
