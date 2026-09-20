import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml


DAY11_ROOT = Path(__file__).resolve().parents[2]
MATRIX = DAY11_ROOT / "source" / "manifests" / "experiment_matrix.json"
BASE_CONFIG = DAY11_ROOT / "configs" / "qwen25_7b_week3_base.yaml"
GENERATOR = DAY11_ROOT / "source" / "scripts" / "generate_experiment_configs.py"
VALIDATOR = DAY11_ROOT / "source" / "scripts" / "validate_experiment_matrix.py"
SMOKE_CONFIG = DAY11_ROOT / "configs" / "qwen25_7b_week3_smoke.yaml"
SMOKE_RUNNER = DAY11_ROOT / "source" / "scripts" / "run_day11_smoke.sh"


def load_runs(path: Path = MATRIX) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["runs"]


def run_generator(output_dir: Path, matrix: Path = MATRIX) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--matrix",
            str(matrix),
            "--base",
            str(BASE_CONFIG),
            "--output",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )


def run_validator(
    config_dir: Path,
    output_path: Path,
    matrix: Path = MATRIX,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--matrix",
            str(matrix),
            "--base",
            str(BASE_CONFIG),
            "--config-dir",
            str(config_dir),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )


def test_matrix_has_required_groups_values_and_unique_ids() -> None:
    assert MATRIX.exists(), "experiment matrix is missing"
    runs = load_runs()

    assert len(runs) == 9
    assert len({row["run_id"] for row in runs}) == 9
    assert Counter(row["group"] for row in runs) == {
        "rank": 3,
        "learning_rate": 3,
        "epoch": 3,
    }
    assert {
        row["lora_rank"] for row in runs if row["group"] == "rank"
    } == {8, 32, 64}
    assert {
        row["learning_rate"]
        for row in runs
        if row["group"] == "learning_rate"
    } == {1e-4, 2e-4, 5e-5}
    assert {
        row["num_train_epochs"] for row in runs if row["group"] == "epoch"
    } == {2.0, 3.0, 5.0}


def test_three_repeated_baselines_are_explicit_and_identical() -> None:
    rows = {row["run_id"]: row for row in load_runs()}
    baseline_ids = ["rank-r8", "lr-1e-4", "epoch-e3"]

    for run_id in baseline_ids:
        row = rows[run_id]
        assert (
            row["lora_rank"],
            row["learning_rate"],
            row["num_train_epochs"],
        ) == (8, 1e-4, 3.0)
        assert row["repeated_baseline"] is True


def test_generator_creates_nine_isolated_configs(tmp_path: Path) -> None:
    output_dir = tmp_path / "configs"
    result = run_generator(output_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    paths = sorted(output_dir.glob("*.yaml"))
    assert len(paths) == 9

    configs = {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in paths
    }
    assert len({config["output_dir"] for config in configs.values()}) == 9
    assert len({config["run_name"] for config in configs.values()}) == 9
    assert configs["rank-r32"]["lora_rank"] == 32
    assert configs["lr-2e-4"]["learning_rate"] == 2e-4
    assert configs["epoch-e5"]["num_train_epochs"] == 5.0


def test_generator_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload["runs"].append(dict(payload["runs"][0]))
    duplicate_matrix = tmp_path / "duplicate.json"
    duplicate_matrix.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = run_generator(tmp_path / "configs", duplicate_matrix)

    assert result.returncode != 0
    assert "duplicate" in (result.stdout + result.stderr).lower()


def test_validator_accepts_generated_matrix(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    generation = run_generator(config_dir)
    assert generation.returncode == 0, generation.stdout + generation.stderr

    output = tmp_path / "validation.json"
    result = run_validator(config_dir, output)

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["run_count"] == 9
    assert report["group_counts"] == {
        "rank": 3,
        "learning_rate": 3,
        "epoch": 3,
    }
    assert report["unique_hyperparameter_combinations"] == 7
    assert report["repeated_baseline_run_ids"] == [
        "rank-r8",
        "lr-1e-4",
        "epoch-e3",
    ]


def test_validator_rejects_non_target_learning_difference(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    generation = run_generator(config_dir)
    assert generation.returncode == 0, generation.stdout + generation.stderr

    rank_r32 = config_dir / "rank-r32.yaml"
    config = yaml.safe_load(rank_r32.read_text(encoding="utf-8"))
    config["learning_rate"] = 2e-4
    rank_r32.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    output = tmp_path / "validation.json"
    result = run_validator(config_dir, output)

    assert result.returncode != 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert any("rank-r32" in error and "learning_rate" in error for error in report["errors"])


def test_smoke_config_is_small_and_isolated() -> None:
    assert SMOKE_CONFIG.exists(), "smoke config is missing"
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    smoke = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))

    expected_differences = {
        "max_samples",
        "num_train_epochs",
        "save_steps",
        "output_dir",
        "run_name",
    }
    actual_differences = {
        key for key in set(base) | set(smoke) if base.get(key) != smoke.get(key)
    }
    assert actual_differences == expected_differences
    assert smoke["max_samples"] == 8
    assert smoke["num_train_epochs"] == 1.0
    assert smoke["save_steps"] == 1000
    assert smoke["output_dir"].startswith("/root/autodl-tmp/qwen25-week3/smoke/")


def test_smoke_runner_cannot_target_formal_run_directory() -> None:
    assert SMOKE_RUNNER.exists(), "smoke runner is missing"
    script = SMOKE_RUNNER.read_text(encoding="utf-8")

    assert "qwen25_7b_week3_smoke.yaml" in script
    assert 'WEEK3_ROOT="/root/autodl-tmp/qwen25-week3"' in script
    assert '${WEEK3_ROOT}/smoke/day11-qlora-smoke' in script
    assert '${WEEK3_ROOT}/runs/' not in script
    assert "Refusing to overwrite non-empty smoke output" in script
