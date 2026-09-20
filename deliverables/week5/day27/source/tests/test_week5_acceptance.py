from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_day27_assets import build_acceptance_rows, validate_model_manifest


def test_acceptance_requires_terminal_v8_final_pass():
    evidence = {
        "day22": {"status": "PASS", "image_count": 5},
        "day23": {"status": "PASS", "record_count": 25},
        "day24": {"status": "PASS", "heatmap_count": 4},
        "day25": {"status": "PASS", "case_count": 10},
        "day26": {
            "status": "PASS",
            "stage": "final",
            "case_count": 20,
            "mean_gain": 0.41,
            "direct_gain": 0.20,
            "opportunity_win_rate": 0.60,
            "hallucination_not_worse": True,
        },
        "day27": {"report_complete": True, "model_archive_complete": True},
    }

    rows = build_acceptance_rows(evidence)

    assert [row["status"] for row in rows] == ["PASS"] * 4
    quality = next(row for row in rows if row["requirement_id"] == "W5-QUALITY")
    assert "mean_gain=0.4100" in quality["actual"]


def test_development_only_result_cannot_pass_teacher_quality_acceptance():
    evidence = {
        "day22": {"status": "PASS", "image_count": 5},
        "day23": {"status": "PASS", "record_count": 25},
        "day24": {"status": "PASS", "heatmap_count": 4},
        "day25": {"status": "PASS", "case_count": 10},
        "day26": {
            "status": "PASS",
            "stage": "dev",
            "case_count": 20,
            "mean_gain": 0.80,
            "direct_gain": 0.50,
            "opportunity_win_rate": 0.90,
            "hallucination_not_worse": True,
        },
        "day27": {"report_complete": True, "model_archive_complete": True},
    }

    rows = build_acceptance_rows(evidence)

    quality = next(row for row in rows if row["requirement_id"] == "W5-QUALITY")
    assert quality["status"] == "FAIL"
    assert "stage=dev" in quality["actual"]


def test_model_manifest_binds_adapter_weight_and_loading_smoke():
    valid = {
        "model_type": "base_plus_lora_adapter",
        "base_model": {
            "repository": "Qwen/Qwen2-VL-7B-Instruct",
            "revision": "e" * 40,
        },
        "adapter": {
            "remote_path": "/root/autodl-tmp/model/adapter",
            "weight_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "total_bytes": 123,
        },
        "training_dataset_sha256": "c" * 64,
        "training_config_sha256": "d" * 64,
        "load_smoke": {"status": "PASS", "return_code": 0},
    }

    assert validate_model_manifest(valid) == []

    valid["adapter"].pop("weight_sha256")
    errors = validate_model_manifest(valid)
    assert "adapter weight SHA-256 is missing or invalid" in errors
