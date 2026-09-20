from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


DAY18_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = DAY18_ROOT / "source" / "scripts" / "build_self_built_preferences.py"

EXPECTED_TYPES = {
    "factuality",
    "safety",
    "completeness",
    "helpfulness",
    "format",
}


def load_builder():
    assert SCRIPT_PATH.is_file(), f"builder is missing: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("build_self_built_preferences", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_freezes_210_balanced_preference_pairs():
    builder = load_builder()

    records = builder.build_self_built_records()

    assert len(records) == 210
    assert Counter(record["preference_type"] for record in records) == {
        category: 42 for category in EXPECTED_TYPES
    }
    assert Counter(record["scene_kind"] for record in records) == {
        "business": 105,
        "general": 105,
    }
    assert Counter(
        (record["preference_type"], record["scene_kind"]) for record in records
    ) == {
        (category, scene_kind): 21
        for category in EXPECTED_TYPES
        for scene_kind in ("business", "general")
    }


def test_builder_records_are_unique_reviewed_sharegpt_pairs():
    builder = load_builder()

    records = builder.build_self_built_records()

    assert len({record["id"] for record in records}) == 210
    prompts = [record["conversations"][-1]["value"].strip() for record in records]
    assert len(set(prompts)) == 210
    for record in records:
        assert record["conversations"][-1]["from"] == "human"
        assert record["chosen"]["from"] == "gpt"
        assert record["rejected"]["from"] == "gpt"
        assert record["chosen"]["value"].strip()
        assert record["rejected"]["value"].strip()
        assert record["chosen"]["value"].strip() != record["rejected"]["value"].strip()
        assert len(record["construction_reason"].strip()) >= 12
        assert record["quality_review"] == {
            "single_dimension": True,
            "safe_to_publish": True,
            "status": "approved",
            "reviewer_kind": "codex_review",
            "reviewed_at": "2026-08-11",
        }
        assert record["source"] == {
            "type": "self_built",
            "source_id": record["id"],
            "revision": "week4-day18-v1",
            "license": "project-authored",
            "retrieved_at": "2026-08-11",
        }


def test_safety_pairs_cover_refusal_and_benign_assistance_equally():
    builder = load_builder()

    safety = [
        record
        for record in builder.build_self_built_records()
        if record["preference_type"] == "safety"
    ]

    assert Counter(record["safety_behavior"] for record in safety) == {
        "refuse_harmful": 21,
        "assist_benign": 21,
    }


def test_cli_writes_deterministic_json_and_complete_review_csv(tmp_path: Path):
    first_json = tmp_path / "first.json"
    second_json = tmp_path / "second.json"
    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"

    for output_json, review_csv in (
        (first_json, first_csv),
        (second_json, second_csv),
    ):
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--output",
                str(output_json),
                "--review-csv",
                str(review_csv),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert b"\r\n" not in first_csv.read_bytes()
    assert len(json.loads(first_json.read_text(encoding="utf-8"))) == 210
    with first_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 210
    assert {row["status"] for row in rows} == {"approved"}
    assert {row["reviewer_kind"] for row in rows} == {"codex_review"}
