from pathlib import Path

import yaml


CONFIG = Path(__file__).parents[2] / "configs" / "qwen25_7b_week4_dpo_corrective_export.yaml"


def test_corrective_export_uses_promoted_adapter_and_independent_output():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["model_name_or_path"].endswith("qwen25-7b-week3-best-merged")
    assert config["adapter_name_or_path"].endswith(
        "reward-corrective-40step/attempt_001/trainer_output"
    )
    assert config["export_dir"].endswith("qwen25-7b-week4-dpo-corrective-merged")
    assert config["export_size"] == 4
    assert config["export_device"] == "cpu"
    assert "quantization_bit" not in config
