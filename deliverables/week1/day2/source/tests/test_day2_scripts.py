import json
from pathlib import Path

import pytest
import torch

from scripts.inference import load_prompt_items, slice_new_tokens
from scripts.validate_results import extract_python_block, validate_result_rows


def test_load_prompt_items_accepts_three_unique_items(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([
        {"id": "a", "system": "s", "user": "u", "max_new_tokens": 10},
        {"id": "b", "system": "s", "user": "u", "max_new_tokens": 20},
        {"id": "c", "system": "s", "user": "u", "max_new_tokens": 30},
    ]), encoding="utf-8")
    assert [item["id"] for item in load_prompt_items(path)] == ["a", "b", "c"]


def test_load_prompt_items_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([
        {"id": "same", "system": "s", "user": "u", "max_new_tokens": 10},
        {"id": "same", "system": "s", "user": "u", "max_new_tokens": 10},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_prompt_items(path)


def test_slice_new_tokens_removes_prompt_prefix() -> None:
    generated = torch.tensor([[11, 12, 13, 21, 22]])
    assert slice_new_tokens(generated, 3).tolist() == [[21, 22]]


def test_extract_python_block_returns_first_python_fence() -> None:
    text = "说明\n```python\nprint('ok')\n```\n```python\nprint('later')\n```"
    assert extract_python_block(text) == "print('ok')\n"


def test_validate_result_rows_requires_exact_ids() -> None:
    def row(item_id: str, response: str) -> dict:
        return {
            "id": item_id,
            "response": response,
            "messages": [],
            "chat_template_text": "template",
            "generation": {"do_sample": False},
            "input_tokens": 1,
            "output_tokens": 1,
            "elapsed_seconds": 0.1,
        }

    rows = [
        row("code_generation", "x"),
        row("logic_reasoning", "y"),
        row("role_play", "z"),
    ]
    validate_result_rows(rows, {"code_generation", "logic_reasoning", "role_play"})
    with pytest.raises(ValueError, match="ids"):
        validate_result_rows(rows[:2], {"code_generation", "logic_reasoning", "role_play"})
