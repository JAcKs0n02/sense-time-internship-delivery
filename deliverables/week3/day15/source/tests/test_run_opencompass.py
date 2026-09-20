import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_opencompass.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_opencompass", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_commands_freezes_two_models_and_identical_benchmark_options(tmp_path):
    module = load_module()
    config = {
        "opencompass_executable": "/env/bin/opencompass",
        "dataset_configs": ["ceval_gen", "cmmlu_gen"],
        "models": [
            {"label": "base", "path": "/models/base"},
            {"label": "best_sft", "path": "/models/best"},
        ],
        "max_seq_len": 2048,
        "max_out_len": 32,
        "batch_size": 4,
        "dataset_source": "ModelScope",
    }

    plans = module.build_run_plans(config, tmp_path / "formal")

    assert [plan.label for plan in plans] == ["base", "best_sft"]
    assert plans[0].command[:4] == [
        "/env/bin/opencompass",
        "--datasets",
        "ceval_gen",
        "cmmlu_gen",
    ]
    assert "--dump-eval-details" in plans[0].command
    base_normalized = ["MODEL" if item == "/models/base" else item for item in plans[0].command]
    best_normalized = ["MODEL" if item == "/models/best" else item for item in plans[1].command]
    base_normalized = ["WORK" if str(tmp_path) in item else item for item in base_normalized]
    best_normalized = ["WORK" if str(tmp_path) in item else item for item in best_normalized]
    assert base_normalized == best_normalized
    assert plans[0].work_dir.name == "attempt-001"
    assert plans[1].work_dir.name == "attempt-001"


def test_validate_config_rejects_duplicate_labels_or_non_frozen_datasets():
    module = load_module()
    config = {
        "opencompass_executable": "/env/bin/opencompass",
        "dataset_configs": ["ceval_gen"],
        "models": [
            {"label": "base", "path": "/models/base"},
            {"label": "base", "path": "/models/best"},
        ],
        "max_seq_len": 2048,
        "max_out_len": 32,
        "batch_size": 4,
        "dataset_source": "ModelScope",
    }

    errors = module.validate_config(config)

    assert any("dataset_configs" in error for error in errors)
    assert any("labels" in error for error in errors)
