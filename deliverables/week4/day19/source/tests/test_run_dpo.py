from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_dpo.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_dpo", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_next_attempt_directory_never_reuses_an_existing_run(tmp_path: Path):
    module = load_module()
    root = tmp_path / "runs"
    (root / "attempt_001").mkdir(parents=True)
    (root / "attempt_002").mkdir()

    assert module.next_attempt_dir(root).name == "attempt_003"


def test_create_attempt_directory_is_atomic(tmp_path: Path):
    module = load_module()
    target = tmp_path / "attempt_001"
    target.mkdir()

    with pytest.raises(FileExistsError):
        module.create_attempt_dir(target)


def test_build_runtime_config_only_changes_runtime_fields(tmp_path: Path):
    module = load_module()
    source = {
        "stage": "dpo",
        "pref_beta": 0.1,
        "model_name_or_path": "/models/week3",
        "ref_model": "/models/week3",
        "output_dir": "/placeholder",
        "run_name": "placeholder",
    }
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()

    runtime = module.build_runtime_config(source, attempt)

    assert runtime["output_dir"] == str(attempt / "trainer_output")
    assert runtime["run_name"] == "day19_attempt_001"
    for key in ("stage", "pref_beta", "model_name_or_path", "ref_model"):
        assert runtime[key] == source[key]
    assert source["output_dir"] == "/placeholder"


def test_write_runtime_config_refuses_to_overwrite(tmp_path: Path):
    module = load_module()
    destination = tmp_path / "runtime.yaml"
    destination.write_text("existing: true\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        module.write_yaml_exclusive(destination, {"new": True})


def test_load_yaml_requires_mapping(tmp_path: Path):
    module = load_module()
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        module.load_yaml(path)
