from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_teacher_tables.py"
DAY20 = Path(__file__).parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location("build_teacher_tables_day20", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_teacher_tables_cover_fixed_denominators_without_private_mapping():
    module = load_module()
    results = DAY20 / "source" / "results"
    records = module.read_responses(
        [results / "sft_only_responses.jsonl", results / "sft_dpo_responses.jsonl"]
    )
    safety = module.build_safety_rows(records, module.read_csv(results / "safety_review.csv"))
    business = module.build_business_rows(records, module.read_csv(results / "business_comparison.csv"))

    assert len(safety) == 10
    assert len(business) == 5
    assert {row["prompt_id"] for row in safety} == {
        f"w4-eval-safety-{index:02d}" for index in range(1, 11)
    }
    assert {row["prompt_id"] for row in business} == {
        f"w4-eval-business-{index:02d}" for index in range(1, 6)
    }
    assert all("model_dir" not in row and "candidate_label" not in row for row in safety + business)
    assert all(row["outcome"] in {"sft_only", "sft_dpo", "tie"} for row in business)
