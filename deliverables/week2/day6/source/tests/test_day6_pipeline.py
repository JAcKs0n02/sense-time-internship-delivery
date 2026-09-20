from __future__ import annotations

import importlib.util
import csv
import json
from pathlib import Path
import sys

import pytest


DAY6_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = DAY6_ROOT / "source" / "scripts"


def load_script(module_name: str):
    path = SCRIPTS_DIR / f"{module_name}.py"
    if not path.exists():
        pytest.fail(f"missing production script: {path.name}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_alpaca_to_alpaca_keeps_only_model_visible_fields():
    converter = load_script("convert_formats")
    source = {
        "instruction": "请概括",
        "input": "春天来了。",
        "output": "春日来临。",
        "_provenance": {"source_index": 7},
    }

    assert converter.alpaca_to_alpaca(source) == {
        "instruction": "请概括",
        "input": "春天来了。",
        "output": "春日来临。",
    }


@pytest.mark.parametrize(
    ("source", "expected_human"),
    [
        (
            {"instruction": "翻译", "input": "你好", "output": "Hello"},
            "翻译\n\n你好",
        ),
        (
            {"instruction": "写一句问候", "input": "", "output": "早上好！"},
            "写一句问候",
        ),
    ],
)
def test_alpaca_to_sharegpt_uses_fixed_input_separator(source, expected_human):
    converter = load_script("convert_formats")

    assert converter.alpaca_to_sharegpt(source) == {
        "conversations": [
            {"from": "human", "value": expected_human},
            {"from": "gpt", "value": source["output"]},
        ]
    }


def test_normalize_sharegpt_maps_roles_and_preserves_all_complete_turns():
    converter = load_script("convert_formats")
    source = {
        "conversations": [
            {"from": "user", "value": "第一问"},
            {"from": "assistant", "value": "第一答"},
            {"from": "human", "value": "第二问"},
            {"from": "gpt", "value": "第二答"},
        ],
        "id": "upstream-id",
    }

    assert converter.normalize_sharegpt(source) == {
        "conversations": [
            {"from": "human", "value": "第一问"},
            {"from": "gpt", "value": "第一答"},
            {"from": "human", "value": "第二问"},
            {"from": "gpt", "value": "第二答"},
        ]
    }


@pytest.mark.parametrize(
    "messages",
    [
        [{"from": "gpt", "value": "没有问题"}],
        [
            {"from": "human", "value": "问题一"},
            {"from": "human", "value": "问题二"},
        ],
        [{"from": "human", "value": "没有回答"}],
        [
            {"from": "human", "value": "问题"},
            {"from": "tool", "value": "工具结果"},
        ],
    ],
)
def test_normalize_sharegpt_rejects_incomplete_or_misaligned_turns(messages):
    converter = load_script("convert_formats")

    with pytest.raises(ValueError):
        converter.normalize_sharegpt({"conversations": messages})


def test_normalize_sharegpt_drops_only_unanswered_tail_after_complete_pairs():
    converter = load_script("convert_formats")
    source = {
        "conversations": [
            {"from": "human", "value": "第一问"},
            {"from": "gpt", "value": "第一答"},
            {"from": "human", "value": "尚未回答的问题"},
        ]
    }

    assert converter.normalize_sharegpt(source) == {
        "conversations": [
            {"from": "human", "value": "第一问"},
            {"from": "gpt", "value": "第一答"},
        ]
    }


def test_sharegpt_to_alpaca_uses_last_turn_and_serializes_prior_history():
    converter = load_script("convert_formats")
    source = {
        "conversations": [
            {"from": "human", "value": "第一问"},
            {"from": "gpt", "value": "第一答"},
            {"from": "human", "value": "第二问"},
            {"from": "gpt", "value": "第二答"},
        ]
    }

    assert converter.sharegpt_to_alpaca(source) == {
        "instruction": "第二问",
        "input": "[human]\n第一问\n[gpt]\n第一答",
        "output": "第二答",
    }


def test_convert_raw_subsets_writes_aligned_formats_and_provenance(tmp_path):
    converter = load_script("convert_formats")
    raw_dir = tmp_path / "raw"
    formatted_dir = tmp_path / "formatted"
    provenance_file = tmp_path / "interim" / "day6_provenance.jsonl"
    raw_dir.mkdir()
    fixtures = {
        "alpaca_gpt4_zh_2k.jsonl": {
            "instruction": "A",
            "input": "",
            "output": "a",
            "_provenance": {
                "source": "alpaca_gpt4_zh",
                "source_index": 10,
                "revision": "rev-a",
            },
        },
        "coig_pc_2k.jsonl": {
            "instruction": "B",
            "input": "b-input",
            "output": "b",
            "_provenance": {
                "source": "coig_pc",
                "source_index": 20,
                "revision": "rev-b",
            },
        },
        "sharegpt_zh_1k.jsonl": {
            "conversations": [
                {"from": "human", "value": "C"},
                {"from": "gpt", "value": "c"},
                {"from": "human", "value": "unanswered"},
            ],
            "_provenance": {
                "source": "sharegpt_zh",
                "source_index": 30,
                "revision": "rev-c",
            },
        },
    }
    for filename, record in fixtures.items():
        (raw_dir / filename).write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    counts = converter.convert_raw_subsets(
        raw_dir=raw_dir,
        formatted_dir=formatted_dir,
        provenance_file=provenance_file,
    )

    assert counts == {
        "alpaca_gpt4_zh": 1,
        "coig_pc": 1,
        "sharegpt_zh": 1,
        "total": 3,
    }
    alpaca_lines = [
        json.loads(line)
        for line in (formatted_dir / "week2_5k_alpaca.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    sharegpt_lines = [
        json.loads(line)
        for line in (formatted_dir / "week2_5k_sharegpt.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    provenance_lines = [
        json.loads(line)
        for line in provenance_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(alpaca_lines) == len(sharegpt_lines) == len(provenance_lines) == 3
    assert alpaca_lines[2] == {"instruction": "C", "input": "", "output": "c"}
    assert sharegpt_lines[1]["conversations"][0]["value"] == "B\n\nb-input"
    assert [item["source"] for item in provenance_lines] == [
        "alpaca_gpt4_zh",
        "coig_pc",
        "sharegpt_zh",
    ]
    assert len({item["sample_id"] for item in provenance_lines}) == 3
    assert provenance_lines[2]["format_adjustments"] == [
        "dropped_trailing_unanswered_human"
    ]


def test_stable_sample_selects_literal_hash_winners_and_restores_source_order():
    collector = load_script("collect_day6")
    records = [{"text": f"record-{index}"} for index in range(10)]

    selected = collector.stable_sample(
        records,
        sample_count=3,
        seed=42,
        source_name="demo",
        revision="fixed-revision",
    )

    assert [item["_provenance"]["source_index"] for item in selected] == [0, 6, 9]
    assert [item["text"] for item in selected] == [
        "record-0",
        "record-6",
        "record-9",
    ]
    assert all(
        item["_provenance"]["revision"] == "fixed-revision" for item in selected
    )


def test_stable_sample_fails_when_upstream_has_too_few_records():
    collector = load_script("collect_day6")

    with pytest.raises(ValueError, match="requested 3 records but upstream has 2"):
        collector.stable_sample(
            [{"text": "a"}, {"text": "b"}],
            sample_count=3,
            seed=42,
            source_name="demo",
            revision="fixed-revision",
        )


def test_collect_source_writes_sampled_rows_and_actual_metadata(tmp_path):
    collector = load_script("collect_day6")
    spec = collector.SourceSpec(
        source_name="demo",
        dataset_name="Demo Dataset",
        repo_id="owner/demo",
        revision="fixed-revision",
        split="train",
        config_name=None,
        sample_count=2,
        output_filename="demo.jsonl",
        license_text="Apache-2.0",
        gated=False,
    )

    def local_loader(repo_id, **kwargs):
        assert repo_id == "owner/demo"
        assert kwargs["revision"] == "fixed-revision"
        return [{"text": f"record-{index}"} for index in range(4)]

    metadata = collector.collect_source(
        spec,
        raw_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        seed=42,
        dataset_loader=local_loader,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "raw" / "demo.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["_provenance"]["source_index"] for record in records] == [0, 1]
    assert metadata["upstream_count"] == 4
    assert metadata["sample_count"] == 2
    assert metadata["raw_file"] == "source/data/raw/demo.jsonl"
    assert len(metadata["sha256"]) == 64
    assert metadata["file_size_bytes"] > 0


def test_collect_all_writes_machine_readable_metadata(tmp_path):
    collector = load_script("collect_day6")
    spec = collector.SourceSpec(
        source_name="demo",
        dataset_name="Demo Dataset",
        repo_id="owner/demo",
        revision="fixed-revision",
        split="train",
        config_name=None,
        sample_count=2,
        output_filename="demo.jsonl",
        license_text="Apache-2.0",
        gated=False,
    )

    def local_loader(repo_id, **kwargs):
        return [{"text": f"record-{index}"} for index in range(4)]

    manifest_json = tmp_path / "manifests" / "collection_metadata.json"
    result = collector.collect_all(
        specs=(spec,),
        raw_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        manifest_json=manifest_json,
        seed=42,
        dataset_loader=local_loader,
    )

    saved = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert result == saved
    assert result["seed"] == 42
    assert result["total_sample_count"] == 2
    assert result["sources"][0]["revision"] == "fixed-revision"


def test_load_local_records_reads_json_array_without_datasets_package(tmp_path):
    collector = load_script("collect_day6")
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps([{"value": "甲"}, {"value": "乙"}], ensure_ascii=False),
        encoding="utf-8",
    )

    records = collector.load_local_records(source_path, "json")

    assert len(records) == 2
    assert list(records) == [{"value": "甲"}, {"value": "乙"}]


def test_load_local_records_streams_parquet_rows_without_datasets_package(tmp_path):
    collector = load_script("collect_day6")
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_path = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"instruction": "问题一", "output": "答案一"},
                {"instruction": "问题二", "output": "答案二"},
            ]
        ),
        source_path,
    )

    records = collector.load_local_records(source_path, "parquet")

    assert len(records) == 2
    assert list(records) == [
        {"instruction": "问题一", "output": "答案一"},
        {"instruction": "问题二", "output": "答案二"},
    ]


def test_collect_source_downloads_fixed_repository_file_without_datasets(tmp_path):
    collector = load_script("collect_day6")
    upstream_path = tmp_path / "upstream.json"
    upstream_path.write_text(
        json.dumps(
            [{"instruction": f"问题{index}", "input": "", "output": f"答案{index}"}
             for index in range(4)],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    spec = collector.SourceSpec(
        source_name="demo",
        dataset_name="Demo Dataset",
        repo_id="owner/demo",
        revision="fixed-revision",
        split="train",
        config_name=None,
        sample_count=2,
        output_filename="demo.jsonl",
        license_text="Apache-2.0",
        gated=True,
        source_filename="data/demo.json",
        file_format="json",
        upstream_file_size_bytes=123,
        upstream_file_sha256="f" * 64,
    )

    def local_downloader(**kwargs):
        assert kwargs == {
            "repo_id": "owner/demo",
            "filename": "data/demo.json",
            "repo_type": "dataset",
            "revision": "fixed-revision",
            "cache_dir": str(tmp_path / "cache"),
            "token": True,
            "resume_download": True,
        }
        return str(upstream_path)

    metadata = collector.collect_source(
        spec,
        raw_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        seed=42,
        file_downloader=local_downloader,
    )

    assert metadata["upstream_count"] == 4
    assert metadata["upstream_file"] == "data/demo.json"
    assert metadata["file_format"] == "json"
    assert metadata["upstream_file_size_bytes"] == 123
    assert metadata["upstream_file_sha256"] == "f" * 64


def test_write_raw_manifests_uses_actual_file_size_and_sha256(tmp_path):
    collector = load_script("collect_day6")
    validator = load_script("validate_day6")
    spec = collector.SourceSpec(
        source_name="demo",
        dataset_name="Demo Dataset",
        repo_id="owner/demo",
        revision="fixed-revision",
        split="train",
        config_name=None,
        sample_count=2,
        output_filename="demo.jsonl",
        license_text="Apache-2.0",
        gated=False,
        source_filename="data/demo.json",
        file_format="json",
        upstream_file_size_bytes=456,
        upstream_file_sha256="a" * 64,
    )

    def local_loader(repo_id, **kwargs):
        return [{"text": f"record-{index}"} for index in range(4)]

    metadata_file = (
        tmp_path / "source" / "manifests" / "collection_metadata.json"
    )
    collected = collector.collect_all(
        specs=(spec,),
        raw_dir=tmp_path / "source" / "data" / "raw",
        cache_dir=tmp_path / "cache",
        manifest_json=metadata_file,
        seed=42,
        dataset_loader=local_loader,
    )

    manifest_paths = validator.write_raw_manifests(tmp_path)

    with manifest_paths["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert b"\r\n" not in manifest_paths["csv"].read_bytes()
    assert len(rows) == 1
    assert rows[0]["dataset_name"] == "Demo Dataset"
    assert rows[0]["revision"] == "fixed-revision"
    assert int(rows[0]["file_size_bytes"]) == collected["sources"][0][
        "file_size_bytes"
    ]
    assert rows[0]["sha256"] == collected["sources"][0]["sha256"]
    assert "upstream_file=data/demo.json" in rows[0]["notes"]
    assert "upstream_file_size_bytes=456" in rows[0]["notes"]
    assert f"upstream_file_sha256={'a' * 64}" in rows[0]["notes"]
    markdown = manifest_paths["markdown"].read_text(encoding="utf-8")
    assert "Demo Dataset" in markdown
    assert "Apache-2.0" in markdown
    assert "https://huggingface.co/datasets/owner/demo" in markdown
    assert "data/demo.json" in markdown
    assert "456" in markdown
    assert "a" * 64 in markdown


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _build_valid_small_delivery(root: Path) -> None:
    raw_dir = root / "source" / "data" / "raw"
    formatted_dir = root / "source" / "data" / "formatted"
    interim_dir = root / "source" / "data" / "interim"
    raw_records = {
        "alpaca_gpt4_zh_2k.jsonl": {
            "instruction": "A",
            "input": "",
            "output": "a",
            "_provenance": {
                "source": "alpaca_gpt4_zh",
                "source_index": 1,
                "revision": "rev-a",
            },
        },
        "coig_pc_2k.jsonl": {
            "instruction": "B",
            "input": "",
            "output": "b",
            "_provenance": {
                "source": "coig_pc",
                "source_index": 2,
                "revision": "rev-b",
            },
        },
        "sharegpt_zh_1k.jsonl": {
            "conversations": [
                {"from": "human", "value": "C"},
                {"from": "gpt", "value": "c"},
            ],
            "_provenance": {
                "source": "sharegpt_zh",
                "source_index": 3,
                "revision": "rev-c",
            },
        },
    }
    for filename, record in raw_records.items():
        _write_jsonl(raw_dir / filename, [record])
    _write_jsonl(
        formatted_dir / "week2_5k_alpaca.jsonl",
        [
            {"instruction": "A", "input": "", "output": "a"},
            {"instruction": "B", "input": "", "output": "b"},
            {"instruction": "C", "input": "", "output": "c"},
        ],
    )
    _write_jsonl(
        formatted_dir / "week2_5k_sharegpt.jsonl",
        [
            {
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer},
                ]
            }
            for prompt, answer in [("A", "a"), ("B", "b"), ("C", "c")]
        ],
    )
    _write_jsonl(
        interim_dir / "day6_provenance.jsonl",
        [
            {
                "sample_id": f"id-{name}",
                "source": name,
                "source_index": index,
                "revision": revision,
                "raw_record_sha256": "a" * 64,
            }
            for name, index, revision in [
                ("alpaca_gpt4_zh", 1, "rev-a"),
                ("coig_pc", 2, "rev-b"),
                ("sharegpt_zh", 3, "rev-c"),
            ]
        ],
    )


def test_validate_delivery_accepts_aligned_valid_records(tmp_path):
    validator = load_script("validate_day6")
    _build_valid_small_delivery(tmp_path)

    report = validator.validate_delivery(
        tmp_path,
        expected_counts={
            "alpaca_gpt4_zh": 1,
            "coig_pc": 1,
            "sharegpt_zh": 1,
        },
        require_manifest=False,
    )

    assert report["valid"] is True
    assert report["counts"]["formatted_alpaca"] == 3
    assert report["counts"]["formatted_sharegpt"] == 3
    assert report["counts"]["provenance"] == 3


def test_validate_delivery_rejects_raw_record_without_provenance(tmp_path):
    validator = load_script("validate_day6")
    _build_valid_small_delivery(tmp_path)
    raw_path = (
        tmp_path
        / "source"
        / "data"
        / "raw"
        / "alpaca_gpt4_zh_2k.jsonl"
    )
    raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_record.pop("_provenance")
    _write_jsonl(raw_path, [raw_record])

    report = validator.validate_delivery(
        tmp_path,
        expected_counts={
            "alpaca_gpt4_zh": 1,
            "coig_pc": 1,
            "sharegpt_zh": 1,
        },
        require_manifest=False,
    )

    assert report["valid"] is False
    assert any("missing _provenance" in error for error in report["errors"])


def test_validate_delivery_rejects_manifest_without_dataset_rows(tmp_path):
    validator = load_script("validate_day6")
    _build_valid_small_delivery(tmp_path)
    manifest_dir = tmp_path / "source" / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "raw_data_manifest.csv").write_text(
        ",".join(validator.MANIFEST_FIELDS) + "\n", encoding="utf-8"
    )
    (manifest_dir / "raw_data_manifest.md").write_text(
        "# Empty manifest\n", encoding="utf-8"
    )

    report = validator.validate_delivery(
        tmp_path,
        expected_counts={
            "alpaca_gpt4_zh": 1,
            "coig_pc": 1,
            "sharegpt_zh": 1,
        },
        require_manifest=True,
    )

    assert report["valid"] is False
    assert any("manifest dataset rows" in error for error in report["errors"])


def test_write_sha256_manifest_hashes_files_by_delivery_relative_path(tmp_path):
    validator = load_script("validate_day6")
    first = tmp_path / "source" / "data" / "a.jsonl"
    second = tmp_path / "source" / "manifests" / "b.csv"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")

    output = validator.write_sha256_manifest(
        tmp_path,
        [first, second],
        tmp_path / "source" / "results" / "day6_files.sha256",
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "b6a98d9ce9a2d9149288fa3df42d377c3e42737afdcdaf7"
        "14e33c0a100b51060  source/data/a.jsonl",
        "f2c82decdd7181cf98945929a62598db7e6b477e11f6e0e"
        "b0ae97020eff151ad  source/manifests/b.csv",
    ]
