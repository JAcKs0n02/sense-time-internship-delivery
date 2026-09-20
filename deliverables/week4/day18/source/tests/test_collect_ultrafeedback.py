from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import pandas as pd


DAY18_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = DAY18_ROOT / "source" / "scripts" / "collect_ultrafeedback.py"


def load_collector():
    assert SCRIPT_PATH.is_file(), f"collector is missing: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("collect_ultrafeedback", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_row(prompt_id: str = "prompt-001") -> dict:
    return {
        "prompt": "请解释为什么需要固定数据版本。",
        "prompt_id": prompt_id,
        "chosen": [
            {"role": "user", "content": "请解释为什么需要固定数据版本。"},
            {
                "role": "assistant",
                "content": "固定 revision 能让下载内容、抽样结果和实验输入保持可复现。",
            },
        ],
        "rejected": [
            {"role": "user", "content": "请解释为什么需要固定数据版本。"},
            {"role": "assistant", "content": "直接使用 main 就行，不需要记录版本。"},
        ],
        "messages": [
            {"role": "user", "content": "请解释为什么需要固定数据版本。"},
            {
                "role": "assistant",
                "content": "固定 revision 能让下载内容、抽样结果和实验输入保持可复现。",
            },
        ],
        "score_chosen": 8.0,
        "score_rejected": 5.5,
        "source": "project_fixture",
    }


def test_stable_priority_has_a_hand_checked_digest():
    collector = load_collector()

    result = collector.stable_priority(42, "ultrafeedback", "prompt-001")

    assert result == "6ffdcc671469f0902beaffd304bdc29997b03dcd9cb7db48f440754a626fd411"


def test_normalize_ultrafeedback_row_produces_sharegpt_ranking():
    collector = load_collector()

    result = collector.normalize_ultrafeedback_row(
        make_row(),
        upstream_index=17,
        revision="abc123",
    )

    assert result["id"] == "uf-prompt-001"
    assert result["conversations"] == [
        {"from": "human", "value": "请解释为什么需要固定数据版本。"}
    ]
    assert result["chosen"] == {
        "from": "gpt",
        "value": "固定 revision 能让下载内容、抽样结果和实验输入保持可复现。",
    }
    assert result["rejected"] == {
        "from": "gpt",
        "value": "直接使用 main 就行，不需要记录版本。",
    }
    assert result["metadata"] == {
        "source_kind": "ultrafeedback",
        "preference_type": "mixed_external",
        "preference_basis": "upstream_score",
        "source_id": "prompt-001",
        "upstream_source": "project_fixture",
        "upstream_index": 17,
        "revision": "abc123",
        "score_chosen": 8.0,
        "score_rejected": 5.5,
        "score_margin": 2.5,
        "construction_reason": "UltraFeedback 上游评分将 chosen 排在 rejected 之前（8.0 > 5.5）。",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.pop("prompt_id"), "prompt_id"),
        (lambda row: row.update(score_chosen=5.5), "score_chosen_not_greater"),
        (
            lambda row: row["rejected"][-1].update(
                content=f"  {row['chosen'][-1]['content']}\n"
            ),
            "responses_not_distinct",
        ),
        (
            lambda row: row["rejected"][0].update(content="另一个问题"),
            "context_mismatch",
        ),
        (
            lambda row: row["chosen"][-1].update(role="user"),
            "chosen_final_role",
        ),
    ],
)
def test_normalize_ultrafeedback_row_rejects_invalid_pairs(mutate, message):
    collector = load_collector()
    row = make_row()
    mutate(row)

    with pytest.raises(ValueError, match=message):
        collector.normalize_ultrafeedback_row(
            row,
            upstream_index=0,
            revision="abc123",
        )


def test_select_records_is_order_independent_and_audits_rejections():
    collector = load_collector()
    valid_rows = [make_row(f"prompt-{index:03d}") for index in range(1, 5)]
    invalid = deepcopy(make_row("prompt-bad"))
    invalid["score_rejected"] = invalid["score_chosen"]
    rows = [valid_rows[2], invalid, valid_rows[0], valid_rows[3], valid_rows[1]]

    forward_records, forward_audit = collector.select_records(
        rows,
        count=3,
        seed=42,
        revision="abc123",
    )
    reverse_records, reverse_audit = collector.select_records(
        list(reversed(rows)),
        count=3,
        seed=42,
        revision="abc123",
    )

    assert [item["id"] for item in forward_records] == [
        item["id"] for item in reverse_records
    ]
    assert len(forward_records) == 3
    assert forward_audit["excluded_count"] == 1
    assert forward_audit["excluded_by_reason"] == {"score_chosen_not_greater": 1}
    assert reverse_audit["excluded_by_reason"] == {"score_chosen_not_greater": 1}


def test_select_records_fails_when_eligible_pool_is_too_small():
    collector = load_collector()

    with pytest.raises(ValueError, match="eligible_pool_below_requested:1<2"):
        collector.select_records(
            [make_row()],
            count=2,
            seed=42,
            revision="abc123",
        )


def test_select_records_applies_reviewed_content_exclusions_before_sampling():
    collector = load_collector()
    rows = [make_row("prompt-001"), make_row("prompt-002"), make_row("prompt-003")]

    records, audit = collector.select_records(
        rows,
        count=2,
        seed=42,
        revision="abc123",
        content_exclusions={"prompt-002": "unsafe_to_publish"},
    )

    assert "uf-prompt-002" not in {record["id"] for record in records}
    assert audit["excluded_by_reason"] == {"unsafe_to_publish": 1}


def test_cli_writes_selected_records_and_auditable_manifest(tmp_path: Path):
    input_path = tmp_path / "train_prefs.parquet"
    output_path = tmp_path / "ultrafeedback_3.json"
    manifest_path = tmp_path / "manifest.json"
    exclusions_path = tmp_path / "content_exclusions.json"
    exclusions_path.write_text(
        json.dumps(
            {
                "exclusions": [
                    {"source_id": "prompt-004", "reason": "unsafe_to_publish"}
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([make_row(f"prompt-{index:03d}") for index in range(1, 5)]).to_parquet(
        input_path,
        index=False,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-parquet",
            str(input_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--revision",
            "abc123",
            "--count",
            "3",
            "--seed",
            "42",
            "--repo-id",
            "allenai/ultrafeedback_binarized_cleaned",
            "--split",
            "train_prefs",
            "--license",
            "mit",
            "--retrieved-at",
            "2026-08-11",
            "--content-exclusions",
            str(exclusions_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    records = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(records) == 3
    assert manifest["dataset"] == {
        "repo_id": "allenai/ultrafeedback_binarized_cleaned",
        "revision": "abc123",
        "split": "train_prefs",
        "license": "mit",
        "retrieved_at": "2026-08-11",
    }
    assert manifest["selection"]["selected_count"] == 3
    assert manifest["selection"]["excluded_by_reason"] == {"unsafe_to_publish": 1}
    assert manifest["files"]["input_parquet"]["size_bytes"] == input_path.stat().st_size
    assert manifest["files"]["output_json"]["sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
