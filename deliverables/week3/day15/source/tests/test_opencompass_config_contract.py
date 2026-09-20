from pathlib import Path
import runpy


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "opencompass_week3.py"


def config() -> dict:
    return runpy.run_path(str(CONFIG_PATH))["CONFIG"]


def test_config_freezes_models_datasets_and_generation_limits():
    payload = config()

    assert payload["opencompass_version"] == "0.5.3"
    assert payload["dataset_configs"] == ["ceval_gen", "cmmlu_gen"]
    assert payload["models"] == [
        {
            "label": "base",
            "path": "/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct",
        },
        {
            "label": "best_sft",
            "path": "/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged",
        },
    ]
    assert payload["max_seq_len"] == 2048
    assert payload["max_out_len"] == 32
    assert payload["batch_size"] == 4


def test_config_uses_pip_cli_compatibility_mode_without_ranking_feedback():
    payload = config()

    assert payload["dataset_source"] == "ModelScope"
    assert payload["hf_type"] == "chat"
    assert payload["opencompass_used_for_selection"] is False
    assert payload["execution_mode"] == "pip_cli_generated_config"
