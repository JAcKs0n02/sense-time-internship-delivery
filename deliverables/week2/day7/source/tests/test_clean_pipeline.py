from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PIPELINE_PATH = Path(__file__).resolve().parents[2] / "clean_pipeline.py"
DEFAULT_MODEL_DIR = Path(
    "/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct"
)


def load_pipeline():
    spec = importlib.util.spec_from_file_location("day7_clean_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载清洗脚本：{PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from transformers import AutoTokenizer

    model_dir = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    if not model_dir.exists():
        pytest.skip(f"Qwen tokenizer 不存在：{model_dir}")
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def conversation_with_exact_template_length(tokenizer, target: int):
    messages = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "答案"},
    ]

    def length() -> int:
        return len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )

    while length() < target:
        missing = target - length()
        messages[0]["content"] += "甲" * missing
    while length() > target:
        messages[0]["content"] = messages[0]["content"][:-1]
    assert length() == target
    return messages


def load_cleaned_day6_records(pipeline, tokenizer, wanted_ids: set[str]):
    day6_dir = Path(__file__).resolve().parents[3] / "day6" / "source" / "data"
    data_path = day6_dir / "formatted" / "week2_5k_sharegpt.jsonl"
    provenance_path = day6_dir / "interim" / "day6_provenance.jsonl"
    found = {}
    with data_path.open(encoding="utf-8") as data_handle, provenance_path.open(
        encoding="utf-8"
    ) as provenance_handle:
        for data_line, provenance_line in zip(data_handle, provenance_handle):
            provenance = json.loads(provenance_line)
            sample_id = provenance["sample_id"]
            if sample_id not in wanted_ids:
                continue
            record = pipeline.parse_sharegpt_record(
                json.loads(data_line),
                sample_id=sample_id,
            )
            record.provenance = provenance
            cleaned, _ = pipeline.clean_record_text(record)
            messages, truncation = pipeline.truncate_messages(
                cleaned.messages,
                tokenizer,
                max_tokens=2048,
            )
            cleaned.messages = messages
            cleaned.was_truncated = truncation["truncated"]
            found[sample_id] = cleaned
    return [found[sample_id] for sample_id in sorted(wanted_ids)]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<p>甲&amp;乙<br>第二行</p>", "甲&乙\n第二行"),
        ("<script>bad()</script><b>正文</b>", "正文"),
        ("a < b and vector<string>", "a < b and vector<string>"),
        ("第一段\n\n第二段  \n", "第一段\n\n第二段  \n"),
    ],
)
def test_strip_html_removes_real_markup_without_destroying_angle_bracket_text(
    raw: str, expected: str
) -> None:
    """Catches tag stripping that glues blocks or deletes code-like text."""
    pipeline = load_pipeline()
    assert pipeline.strip_html(raw) == expected


def test_clean_controls_removes_unwanted_formats_without_breaking_unicode() -> None:
    """Catches over-broad Unicode deletion and failure to normalize CRLF."""
    pipeline = load_pipeline()
    raw = "A\r\nB\x00\x1b\u200b\u200c\ufeff👩\u200d💻e\u0301\tcode"
    assert pipeline.clean_controls(raw) == "A\nB👩\u200d💻e\u0301\tcode"


def test_normalize_whitespace_keeps_code_indentation_and_internal_spaces() -> None:
    """Catches whitespace flattening that destroys code or meaningful spacing."""
    pipeline = load_pipeline()
    raw = "\n\tdef x():  \n\t\treturn 1   \n\n\n\ntext  inside\n"
    expected = "    def x():\n        return 1\n\ntext  inside"
    assert pipeline.normalize_whitespace(raw) == expected


def test_parse_alpaca_requires_string_instruction_and_output_but_allows_empty_input() -> None:
    """Catches silent string coercion and rejection of legal empty input."""
    pipeline = load_pipeline()

    parsed = pipeline.parse_alpaca_record(
        {"instruction": "解释概念", "input": "", "output": "这是答案"},
        sample_id="alpaca:1",
    )
    assert [message["role"] for message in parsed.messages] == ["user", "assistant"]
    assert [message["content"] for message in parsed.messages] == [
        "解释概念",
        "这是答案",
    ]

    invalid_cases = [
        ({"instruction": None, "input": "", "output": "答案"}, "invalid_instruction_type"),
        ({"instruction": "  ", "input": "", "output": "答案"}, "empty_instruction"),
        ({"instruction": "任务", "input": "", "output": []}, "invalid_output_type"),
        ({"instruction": "任务", "input": "", "output": "\n"}, "empty_output"),
        ({"instruction": "任务", "input": 3, "output": "答案"}, "invalid_input_type"),
    ]
    for record, expected_reason in invalid_cases:
        with pytest.raises(pipeline.RecordRejected) as error:
            pipeline.parse_alpaca_record(record, sample_id="bad")
        assert error.value.reason == expected_reason


def test_parse_sharegpt_requires_nonempty_alternating_complete_pairs() -> None:
    """Catches empty messages, invalid roles, wrong order and missing answers."""
    pipeline = load_pipeline()
    valid = {
        "conversations": [
            {"from": "human", "value": "问题一"},
            {"from": "gpt", "value": "答案一"},
            {"from": "human", "value": "问题二"},
            {"from": "gpt", "value": "答案二"},
        ]
    }
    parsed = pipeline.parse_sharegpt_record(valid, sample_id="sharegpt:1")
    assert [message["role"] for message in parsed.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    invalid_cases = [
        ({"conversations": None}, "invalid_conversations_type"),
        ({"conversations": []}, "empty_conversations"),
        (
            {"conversations": [{"from": "human", "value": "  "}]},
            "empty_message",
        ),
        (
            {
                "conversations": [
                    {"from": "gpt", "value": "先回答"},
                    {"from": "human", "value": "再提问"},
                ]
            },
            "invalid_role_order",
        ),
        (
            {
                "conversations": [
                    {"from": "human", "value": "问题"},
                    {"from": "tool", "value": "结果"},
                ]
            },
            "invalid_role",
        ),
        (
            {"conversations": [{"from": "human", "value": "没有回答"}]},
            "missing_assistant",
        ),
    ]
    for record, expected_reason in invalid_cases:
        with pytest.raises(pipeline.RecordRejected) as error:
            pipeline.parse_sharegpt_record(record, sample_id="bad")
        assert error.value.reason == expected_reason


def test_clean_record_text_applies_rules_and_rejects_postclean_empty_messages() -> None:
    """Catches records that become empty only after HTML/control cleaning."""
    pipeline = load_pipeline()
    record = pipeline.CanonicalRecord(
        sample_id="sample:1",
        messages=[
            {"role": "user", "content": "<p>问题</p>\u200b"},
            {"role": "assistant", "content": "<b>答案</b>"},
        ],
    )
    cleaned, changes = pipeline.clean_record_text(record)
    assert [message["content"] for message in cleaned.messages] == ["问题", "答案"]
    assert changes == {"html": True, "controls": True, "whitespace": True}

    empty_after_cleaning = pipeline.CanonicalRecord(
        sample_id="sample:2",
        messages=[
            {"role": "user", "content": "<script>only markup</script>"},
            {"role": "assistant", "content": "答案"},
        ],
    )
    with pytest.raises(pipeline.RecordRejected) as error:
        pipeline.clean_record_text(empty_after_cleaning)
    assert error.value.reason == "empty_after_cleaning"


def test_token_boundary_keeps_2048_and_truncates_2049(
    qwen_tokenizer,
) -> None:
    """Catches off-by-one truncation and raw-character length checks."""
    pipeline = load_pipeline()
    exactly_2048 = conversation_with_exact_template_length(qwen_tokenizer, 2048)
    over_2048 = conversation_with_exact_template_length(qwen_tokenizer, 2049)

    kept, kept_audit = pipeline.truncate_messages(
        exactly_2048, qwen_tokenizer, max_tokens=2048
    )
    assert kept == exactly_2048
    assert kept_audit["truncated"] is False
    assert pipeline.chat_token_length(kept, qwen_tokenizer) == 2048

    truncated, audit = pipeline.truncate_messages(
        over_2048, qwen_tokenizer, max_tokens=2048
    )
    assert audit["truncated"] is True
    assert audit["original_length"] == 2049
    assert audit["final_length"] <= 2048
    assert [message["role"] for message in truncated] == ["user", "assistant"]
    assert all(message["content"] for message in truncated)


def test_multiturn_truncation_drops_oldest_complete_pair_first(
    qwen_tokenizer,
) -> None:
    """Catches message-level slicing that leaves broken role sequences."""
    pipeline = load_pipeline()
    messages = [
        {"role": "user", "content": "甲" * 1500},
        {"role": "assistant", "content": "乙" * 900},
        {"role": "user", "content": "必须保留的新问题"},
        {"role": "assistant", "content": "必须保留的新答案"},
    ]
    assert (
        len(
            qwen_tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
        )
        > 2048
    )

    truncated, audit = pipeline.truncate_messages(
        messages, qwen_tokenizer, max_tokens=2048
    )
    assert truncated == messages[-2:]
    assert audit["removed_pairs"] == 1
    assert audit["truncated_roles"] == []
    assert pipeline.chat_token_length(truncated, qwen_tokenizer) <= 2048


def test_exact_deduplication_ignores_provenance_and_maps_duplicate_to_keeper() -> None:
    """Catches provenance leaking into duplicate identity."""
    pipeline = load_pipeline()
    first = pipeline.CanonicalRecord(
        sample_id="source-a:1",
        messages=[
            {"role": "user", "content": "同一个问题"},
            {"role": "assistant", "content": "同一个答案"},
        ],
        provenance={"source": "source-a", "source_index": 1},
    )
    second = pipeline.CanonicalRecord(
        sample_id="source-b:9",
        messages=first.messages,
        provenance={"source": "source-b", "source_index": 9},
    )

    kept, duplicates = pipeline.exact_deduplicate([first, second])
    assert [record.sample_id for record in kept] == ["source-a:1"]
    assert duplicates == [
        {
            "duplicate_id": "source-b:9",
            "kept_id": "source-a:1",
            "reason": "exact",
            "distance": 0,
        }
    ]


def test_simhash_uses_stable_sha_features_and_bands_find_distance_three() -> None:
    """Catches process-random hash(), wrong bit voting and broken banding."""
    pipeline = load_pipeline()
    assert pipeline.simhash64("abc") == 0xBA7816BF8F01CFEA
    assert pipeline.hamming_distance(0b1010, 0b0011) == 2

    fingerprints = [
        0,
        (1 << 0) | (1 << 16) | (1 << 32),
        (1 << 0) | (1 << 16) | (1 << 32) | (1 << 48),
    ]
    candidates = pipeline.banded_candidate_pairs(fingerprints, bands=4)
    assert (0, 1) in candidates
    assert (0, 2) not in candidates


def test_fuzzy_deduplication_removes_near_copy_but_protects_short_and_far_text() -> None:
    """Catches false positives from topic similarity and very short replies."""
    pipeline = load_pipeline()
    base = "机器学习能够从数据中学习规律，帮助完成分类、预测和生成任务。" * 8
    near = base.replace("任务。", "任务！", 1)
    far = "今天阳光很好，我们去公园散步并准备晚餐。" * 8

    def record(sample_id: str, answer: str):
        return pipeline.CanonicalRecord(
            sample_id=sample_id,
            messages=[
                {"role": "user", "content": "请解释"},
                {"role": "assistant", "content": answer},
            ],
        )

    records = [
        record("base", base),
        record("near", near),
        record("far", far),
        record("short-a", "好的"),
        record("short-b", "好哒"),
    ]
    kept, duplicates, clusters = pipeline.fuzzy_deduplicate(
        records,
        threshold=3,
        min_chars=20,
        min_jaccard=0.85,
    )
    assert [item.sample_id for item in kept] == [
        "base",
        "far",
        "short-a",
        "short-b",
    ]
    assert len(duplicates) == 1
    assert duplicates[0]["duplicate_id"] == "near"
    assert duplicates[0]["kept_id"] == "base"
    assert duplicates[0]["reason"] == "fuzzy"
    assert duplicates[0]["distance"] <= 3
    assert clusters == [
        {
            "kept_id": "base",
            "member_ids": ["base", "near"],
            "size": 2,
        }
    ]


def test_jaccard_guard_separates_near_copy_from_template_collision() -> None:
    """Catches fuzzy confirmation based on Hamming distance alone."""
    pipeline = load_pipeline()
    base = "机器学习能够从数据中学习规律，帮助完成分类、预测和生成任务。" * 8
    near = base.replace("任务。", "任务！", 1)
    unrelated = "今天阳光很好，我们去公园散步并准备晚餐。" * 8
    assert pipeline.character_ngram_jaccard(base, near) > 0.85
    assert pipeline.character_ngram_jaccard(base, unrelated) < 0.20


def test_real_distance_zero_collision_is_retained_by_jaccard_guard(
    qwen_tokenizer,
) -> None:
    """Regression for two unrelated ShareGPT samples with identical SimHash."""
    pipeline = load_pipeline()
    wanted_ids = {
        "86080173a6021c5709e0fdf0c423e45fde12ba0c65e3194e6cc2d7d36391610a",
        "d54d1661514320b3bf29c7c086dee03bce0b5623add3f248ae10cb24cc92e7c6",
    }
    records = load_cleaned_day6_records(pipeline, qwen_tokenizer, wanted_ids)
    assert len(records) == 2
    left_text = pipeline.canonical_text(records[0])
    right_text = pipeline.canonical_text(records[1])
    assert (
        pipeline.hamming_distance(
            pipeline.simhash64(left_text),
            pipeline.simhash64(right_text),
        )
        == 0
    )
    assert pipeline.character_ngram_jaccard(left_text, right_text) < 0.20

    kept, duplicates, clusters = pipeline.fuzzy_deduplicate(
        records,
        threshold=3,
        min_chars=20,
        min_jaccard=0.85,
    )
    assert len(kept) == 2
    assert duplicates == []
    assert clusters == []


def test_real_shared_case_with_different_task_and_answer_is_not_removed(
    qwen_tokenizer,
) -> None:
    """Regression for COIG records sharing a case but asking different questions."""
    pipeline = load_pipeline()
    wanted_ids = {
        "8c7f41d5c35f1318d7413f3b08f1638efa1397c0383aa0e36fb19da03c75a4bf",
        "38cffd4528c1e1309b66b77b948cf429cd453e5764ef2ff81d0d4b2eba52d116",
    }
    records = load_cleaned_day6_records(pipeline, qwen_tokenizer, wanted_ids)
    assert len(records) == 2
    left_text = pipeline.canonical_text(records[0])
    right_text = pipeline.canonical_text(records[1])
    assert (
        pipeline.hamming_distance(
            pipeline.simhash64(left_text),
            pipeline.simhash64(right_text),
        )
        <= 3
    )
    assert pipeline.character_ngram_jaccard(left_text, right_text) > 0.85
    left_answer = "\n".join(
        item["content"] for item in records[0].messages if item["role"] == "assistant"
    )
    right_answer = "\n".join(
        item["content"] for item in records[1].messages if item["role"] == "assistant"
    )
    assert pipeline.character_ngram_jaccard(left_answer, right_answer) < 0.90

    kept, duplicates, clusters = pipeline.fuzzy_deduplicate(
        records,
        threshold=3,
        min_chars=20,
        min_jaccard=0.85,
        min_role_jaccard=0.90,
    )
    assert len(kept) == 2
    assert duplicates == []
    assert clusters == []


def test_collect_simhash_candidates_preserves_distance_and_bounded_evidence() -> None:
    """Catches calibration output that cannot be reviewed or reproduced."""
    pipeline = load_pipeline()
    base = "数据清洗需要删除噪声、统一格式并保留有效语义。" * 8
    near = base.replace("语义。", "语义！", 1)
    records = [
        pipeline.CanonicalRecord(
            sample_id="left",
            messages=[
                {"role": "user", "content": "说明"},
                {"role": "assistant", "content": base},
            ],
            provenance={"source": "source-left"},
        ),
        pipeline.CanonicalRecord(
            sample_id="right",
            messages=[
                {"role": "user", "content": "说明"},
                {"role": "assistant", "content": near},
            ],
            provenance={"source": "source-right"},
        ),
    ]
    candidates = pipeline.collect_simhash_candidates(
        records,
        max_distance=6,
        min_chars=20,
        preview_chars=80,
    )
    assert len(candidates) == 1
    assert candidates[0]["left_id"] == "left"
    assert candidates[0]["right_id"] == "right"
    assert candidates[0]["left_source"] == "source-left"
    assert candidates[0]["right_source"] == "source-right"
    assert candidates[0]["distance"] <= 3
    assert len(candidates[0]["left_preview"]) <= 80
    assert len(candidates[0]["right_preview"]) <= 80


def test_cli_help_lists_all_required_pipeline_arguments() -> None:
    """Catches a script that only works through imports or edited constants."""
    result = subprocess.run(
        [sys.executable, str(PIPELINE_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for option in (
        "--input",
        "--provenance",
        "--output-alpaca",
        "--output-sharegpt",
        "--tokenizer",
        "--max-tokens",
        "--simhash-threshold",
        "--min-jaccard",
        "--min-role-jaccard",
        "--seed",
        "--results-dir",
        "--evidence-dir",
    ):
        assert option in result.stdout


def test_cli_cleans_deduplicates_and_writes_aligned_outputs_and_evidence(
    tmp_path: Path,
    qwen_tokenizer,
) -> None:
    """Catches orchestration that omits a required stage or misaligns formats."""
    base = "机器学习能够从数据中学习规律，帮助完成分类、预测和生成任务。" * 8
    near = base.replace("任务。", "任务！", 1)
    far = "今天阳光很好，我们去公园散步并准备晚餐。" * 8
    input_records = [
        {
            "conversations": [
                {"from": "human", "value": "<p>请解释</p>\u200b"},
                {"from": "gpt", "value": f"<div>{base}</div>"},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "请解释"},
                {"from": "gpt", "value": base},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "请解释"},
                {"from": "gpt", "value": near},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "请解释"},
                {"from": "gpt", "value": far},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "   "},
                {"from": "gpt", "value": "无效"},
            ]
        },
    ]
    provenance_records = [
        {
            "sample_id": f"sample-{index}",
            "source": f"source-{index % 3}",
            "source_index": index,
            "revision": "test-revision",
        }
        for index in range(len(input_records))
    ]
    input_path = tmp_path / "input.jsonl"
    provenance_path = tmp_path / "provenance.jsonl"
    input_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in input_records
        ),
        encoding="utf-8",
    )
    provenance_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in provenance_records
        ),
        encoding="utf-8",
    )

    output_alpaca = tmp_path / "data" / "clean_alpaca.jsonl"
    output_sharegpt = tmp_path / "data" / "clean_sharegpt.jsonl"
    results_dir = tmp_path / "results"
    evidence_dir = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_PATH),
            "--input",
            str(input_path),
            "--provenance",
            str(provenance_path),
            "--output-alpaca",
            str(output_alpaca),
            "--output-sharegpt",
            str(output_sharegpt),
            "--tokenizer",
            os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR)),
            "--max-tokens",
            "2048",
            "--simhash-threshold",
            "3",
            "--seed",
            "42",
            "--results-dir",
            str(results_dir),
            "--evidence-dir",
            str(evidence_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, result.stderr

    alpaca_rows = [
        json.loads(line)
        for line in output_alpaca.read_text(encoding="utf-8").splitlines()
    ]
    sharegpt_rows = [
        json.loads(line)
        for line in output_sharegpt.read_text(encoding="utf-8").splitlines()
    ]
    assert len(alpaca_rows) == len(sharegpt_rows) == 2
    assert all(set(row) == {"instruction", "input", "output"} for row in alpaca_rows)
    assert all(set(row) == {"conversations"} for row in sharegpt_rows)

    for row in sharegpt_rows:
        messages = [
            {
                "role": "user" if item["from"] == "human" else "assistant",
                "content": item["value"],
            }
            for item in row["conversations"]
        ]
        assert (
            len(
                qwen_tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
            <= 2048
        )

    stats = json.loads(
        (results_dir / "cleaning_stats.json").read_text(encoding="utf-8")
    )
    assert stats["counts"]["raw_total"] == 5
    assert stats["counts"]["initial_empty_removed"] == 1
    assert stats["counts"]["exact_duplicates_removed"] == 1
    assert stats["counts"]["fuzzy_duplicates_removed"] == 1
    assert stats["counts"]["final_total"] == 2

    for path in (
        results_dir / "cleaning_stats.csv",
        results_dir / "cleaning_audit.jsonl",
        results_dir / "cleaned_provenance.jsonl",
        results_dir / "simhash_calibration.csv",
        results_dir / "duplicate_pairs.csv",
        results_dir / "dedup_clusters.jsonl",
        results_dir / "day7_validation.json",
        evidence_dir / "length_distribution.png",
        evidence_dir / "cleaning_counts.png",
    ):
        assert path.exists() and path.stat().st_size > 0

    for csv_path in (
        results_dir / "cleaning_stats.csv",
        results_dir / "simhash_calibration.csv",
        results_dir / "duplicate_pairs.csv",
    ):
        assert b"\r\n" not in csv_path.read_bytes()
