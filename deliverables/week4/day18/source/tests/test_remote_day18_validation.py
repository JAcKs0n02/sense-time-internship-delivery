from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "remote_validate_day18.py"


def _module():
    spec = importlib.util.spec_from_file_location("remote_validate_day18", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_describe_lengths_reports_exact_percentiles() -> None:
    module = _module()
    assert module.describe_lengths([1, 2, 3, 4, 5]) == {
        "count": 5,
        "min": 1,
        "max": 5,
        "mean": 3.0,
        "median": 3.0,
        "p90": 5,
        "p95": 5,
        "p99": 5,
    }


def test_validate_sharegpt_records_accepts_ranking_shape() -> None:
    module = _module()
    records = [
        {
            "id": "one",
            "conversations": [{"from": "human", "value": "Question"}],
            "chosen": {"from": "gpt", "value": "Good"},
            "rejected": {"from": "gpt", "value": "Bad"},
        }
    ]
    assert module.validate_sharegpt_records(records) == []


def test_validate_sharegpt_records_rejects_bad_role_and_equal_pair() -> None:
    module = _module()
    records = [
        {
            "id": "bad",
            "conversations": [{"from": "gpt", "value": "Question"}],
            "chosen": {"from": "gpt", "value": "Same"},
            "rejected": {"from": "gpt", "value": "Same"},
        }
    ]
    assert module.validate_sharegpt_records(records) == [
        {"id": "bad", "errors": ["conversation_role_order", "responses_not_distinct"]}
    ]


def test_length_statistics_fails_when_a_pair_exceeds_cutoff() -> None:
    module = _module()

    class FakeTokenizer:
        name_or_path = "fake"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is True
            if len(messages) == 1:
                return [1] * 10
            return [1] * (25 if messages[-1]["content"] == "too long" else 12)

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [1] * len(text.split())

    records = [
        {
            "id": "over",
            "conversations": [{"from": "human", "value": "Question"}],
            "chosen": {"from": "gpt", "value": "short"},
            "rejected": {"from": "gpt", "value": "too long"},
        }
    ]
    result = module.collect_length_statistics(FakeTokenizer(), records, cutoff_len=20)
    assert result["status"] == "fail"
    assert result["records_over_cutoff"] == 1
    assert result["records_over_cutoff_ids"] == ["over"]
