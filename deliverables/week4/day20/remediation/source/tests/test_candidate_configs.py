from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "safety_v2_lr2e6_e2.yaml": (2e-6, 2.0),
    "safety_v2_lr5e6_e2.yaml": (5e-6, 2.0),
    "safety_v2_lr2e6_e3.yaml": (2e-6, 3.0),
}
CORRECTIVE = ROOT / "configs" / "reward_corrective_40step.yaml"


def test_candidate_configs_freeze_shared_contract():
    for name, (learning_rate, epochs) in CONFIGS.items():
        config = yaml.safe_load((ROOT / "configs" / name).read_text())
        assert config["stage"] == "dpo"
        assert config["model_name_or_path"].endswith("qwen25-7b-week3-best-merged")
        assert config["dataset"] == "week4_dpo_train_v2"
        assert config["eval_dataset"] == "week4_dpo_validation_v2"
        assert config["pref_beta"] == 0.1
        assert config["lora_rank"] == 8
        assert config["quantization_bit"] == 4
        assert config["quantization_type"] == "nf4"
        assert config["bf16"] is True
        assert config["per_device_train_batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 8
        assert config["cutoff_len"] == 2048
        assert config["eval_steps"] == config["save_steps"] == 40
        assert config["resume_from_checkpoint"] is None
        assert config["learning_rate"] == learning_rate
        assert config["num_train_epochs"] == epochs


def test_reward_corrective_config_targets_teacher_reward_trend_without_changing_beta():
    config = yaml.safe_load(CORRECTIVE.read_text(encoding="utf-8"))
    assert config["stage"] == "dpo"
    assert config["pref_beta"] == 0.1
    assert config["model_name_or_path"].endswith("qwen25-7b-week3-best-merged")
    assert config["dataset"] == "week4_dpo_train_v2"
    assert config["eval_dataset"] == "week4_dpo_validation_v2"
    assert config["max_steps"] == 40
    assert config["logging_steps"] == 1
    assert config["save_steps"] == 40
    assert config["eval_steps"] == 20
    assert config["lr_scheduler_type"] == "constant_with_warmup"
    assert config["warmup_steps"] == 29
    assert config["resume_from_checkpoint"] is None
