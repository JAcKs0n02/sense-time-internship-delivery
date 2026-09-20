import json
import subprocess
import sys
from pathlib import Path


DAY8_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = DAY8_ROOT / "source" / "scripts" / "validate_day8_config.py"
CONFIG = DAY8_ROOT / "configs" / "qwen25_7b_week2_lora_annotated.yaml"
DATASET_INFO = DAY8_ROOT / "source" / "data" / "dataset_info.json"


def test_annotated_config_passes_teacher_requirements(tmp_path: Path) -> None:
    output = tmp_path / "validation.json"
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--config",
            str(CONFIG),
            "--dataset-info",
            str(DATASET_INFO),
            "--expected-count",
            "4999",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["valid"] is True
    assert payload["active_parameter_count"] >= 35
    assert payload["all_parameters_fully_annotated"] is True
    assert payload["dataset_name"] == "week2_clean_alpaca"


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "duplicate.yaml"
    bad_config.write_text(
        "# 作用：a\n# 当前值：a\n# 调整影响：a\nseed: 42\n"
        "# 作用：b\n# 当前值：b\n# 调整影响：b\nseed: 7\n",
        encoding="utf-8",
    )
    output = tmp_path / "validation.json"
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--config",
            str(bad_config),
            "--dataset-info",
            str(DATASET_INFO),
            "--expected-count",
            "4999",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "duplicate" in (result.stdout + result.stderr).lower()


def test_parameter_without_three_part_comment_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "missing_comment.yaml"
    bad_config.write_text("# 作用：only one part\nseed: 42\n", encoding="utf-8")
    output = tmp_path / "validation.json"
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--config",
            str(bad_config),
            "--dataset-info",
            str(DATASET_INFO),
            "--expected-count",
            "4999",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "annotation" in (result.stdout + result.stderr).lower()
