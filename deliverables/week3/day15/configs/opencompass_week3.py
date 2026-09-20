"""Frozen OpenCompass 0.5.3 CLI plan for the Week 3 comparison.

The pip distribution does not ship the repository-level ``run.py`` and
``tools/list_configs.py`` assumed by the initial plan.  The official
``opencompass`` CLI resolves ``ceval_gen`` and ``cmmlu_gen`` and writes the
fully expanded executable config into each work directory.  The companion
``run_opencompass.py`` consumes this dictionary and preserves those generated
configs as raw evidence.
"""


CONFIG = {
    "opencompass_version": "0.5.3",
    "opencompass_executable": "/root/autodl-tmp/conda/envs/llm_exp/bin/opencompass",
    "dataset_configs": ["ceval_gen", "cmmlu_gen"],
    "dataset_source": "ModelScope",
    "hf_type": "chat",
    "models": [
        {
            "label": "base",
            "path": "/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct",
        },
        {
            "label": "best_sft",
            "path": "/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged",
        },
    ],
    "max_seq_len": 2048,
    "max_out_len": 32,
    "batch_size": 4,
    "dump_eval_details": True,
    "execution_mode": "pip_cli_generated_config",
    "opencompass_used_for_selection": False,
}
