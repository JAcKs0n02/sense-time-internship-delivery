#!/usr/bin/env python3
"""根据 Qwen2 配置统计各参数组的参数量，不加载模型权重。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "model_type",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "tie_word_embeddings",
)


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} 必须是正整数，实际为 {value!r}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    """读取 JSON 配置并检查统计所需字段。"""
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    missing = [key for key in REQUIRED_FIELDS if key not in config]
    if missing:
        raise ValueError(f"config.json 缺少字段: {', '.join(missing)}")
    return config


def analyze_config(config: dict[str, Any]) -> dict[str, Any]:
    """按 Transformers 中 Qwen2ForCausalLM 的参数形状计算参数量。"""
    missing = [key for key in REQUIRED_FIELDS if key not in config]
    if missing:
        raise ValueError(f"配置缺少字段: {', '.join(missing)}")
    if config["model_type"] != "qwen2":
        raise ValueError("本脚本仅支持 model_type=qwen2")
    if not isinstance(config["tie_word_embeddings"], bool):
        raise ValueError("tie_word_embeddings 必须是布尔值")

    vocab_size = _positive_int(config, "vocab_size")
    hidden_size = _positive_int(config, "hidden_size")
    intermediate_size = _positive_int(config, "intermediate_size")
    num_layers = _positive_int(config, "num_hidden_layers")
    num_q_heads = _positive_int(config, "num_attention_heads")
    num_kv_heads = _positive_int(config, "num_key_value_heads")

    if hidden_size % num_q_heads != 0:
        raise ValueError(
            "hidden_size 必须能被 num_attention_heads 整除: "
            f"{hidden_size} % {num_q_heads} != 0"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "num_attention_heads 必须能被 num_key_value_heads 整除: "
            f"{num_q_heads} % {num_kv_heads} != 0"
        )

    head_dim = hidden_size // num_q_heads
    q_out_features = num_q_heads * head_dim
    kv_out_features = num_kv_heads * head_dim

    # Qwen2Attention: q_proj/k_proj/v_proj 带 bias，o_proj 不带 bias。
    q_projection = hidden_size * q_out_features + q_out_features
    k_projection = hidden_size * kv_out_features + kv_out_features
    v_projection = hidden_size * kv_out_features + kv_out_features
    o_projection = q_out_features * hidden_size
    attention_parameters = q_projection + k_projection + v_projection + o_projection

    # SwiGLU 包含 gate_proj、up_proj、down_proj，三个线性层均不带 bias。
    mlp_parameters = 3 * hidden_size * intermediate_size
    layer_norm_parameters = 2 * hidden_size
    decoder_layer_parameters = (
        attention_parameters + mlp_parameters + layer_norm_parameters
    )

    token_embeddings = vocab_size * hidden_size
    final_norm = hidden_size
    lm_head = 0 if config["tie_word_embeddings"] else vocab_size * hidden_size

    group_counts: list[tuple[str, int]] = [("token_embeddings", token_embeddings)]
    group_counts.extend(
        (f"decoder_layer_{index:02d}", decoder_layer_parameters)
        for index in range(num_layers)
    )
    group_counts.extend((("final_norm", final_norm), ("lm_head", lm_head)))

    total_parameters = sum(parameters for _, parameters in group_counts)
    groups = [
        {
            "group": group,
            "parameters": parameters,
            "percent": parameters / total_parameters * 100,
        }
        for group, parameters in group_counts
    ]

    return {
        "model_type": config["model_type"],
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "queries_per_kv_head": num_q_heads // num_kv_heads,
        "attention_parameters_per_layer": attention_parameters,
        "mlp_parameters_per_layer": mlp_parameters,
        "norm_parameters_per_layer": layer_norm_parameters,
        "decoder_layer_parameters": decoder_layer_parameters,
        "total_parameters": total_parameters,
        "transformer_body_parameters": total_parameters
        - token_embeddings
        - lm_head,
        "groups": groups,
    }


def write_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    """把参数分组写为可复核的 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("group", "parameters", "percent"))
        writer.writeheader()
        for row in groups:
            writer.writerow(
                {
                    "group": row["group"],
                    "parameters": row["parameters"],
                    "percent": f'{row["percent"]:.10f}',
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Qwen2 config.json 统计各参数组参数量（不加载权重）"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="config.json 的路径")
    source.add_argument("--model", type=Path, help="包含 config.json 的模型目录")
    parser.add_argument("--output", required=True, type=Path, help="输出 CSV 路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config is not None else args.model / "config.json"
    config = load_config(config_path)
    result = analyze_config(config)
    write_csv(args.output, result["groups"])

    summary_fields = (
        ("config", str(config_path.resolve())),
        ("model_type", result["model_type"]),
        ("vocab_size", result["vocab_size"]),
        ("hidden_size", result["hidden_size"]),
        ("intermediate_size", result["intermediate_size"]),
        ("num_hidden_layers", result["num_hidden_layers"]),
        ("num_attention_heads", result["num_attention_heads"]),
        ("num_key_value_heads", result["num_key_value_heads"]),
        ("head_dim", result["head_dim"]),
        ("queries_per_kv_head", result["queries_per_kv_head"]),
        (
            "attention_parameters_per_layer",
            result["attention_parameters_per_layer"],
        ),
        ("mlp_parameters_per_layer", result["mlp_parameters_per_layer"]),
        ("norm_parameters_per_layer", result["norm_parameters_per_layer"]),
        ("decoder_layer_parameters", result["decoder_layer_parameters"]),
        ("transformer_body_parameters", result["transformer_body_parameters"]),
        ("total_parameters", result["total_parameters"]),
        ("csv", str(args.output.resolve())),
    )
    for key, value in summary_fields:
        print(f"{key}={value}")
    print("analysis_exit_code=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
