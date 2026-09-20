# Week 3 Day 11: Experiment Plan

本文件对应老师要求的“实验计划表（含假设）”。正式实验固定使用 Qwen2.5-7B-Instruct、Week 2 Day 7 的 4,999 条清洗后 Alpaca 数据和 `seed=42`。

## Frozen Baseline

`rank=8`、`learning_rate=1e-4`、`epochs=3`、`lora_alpha=16`、`seed=42`。

## Experiment Matrix and Hypotheses

| Order | Run ID | Group | Rank | Learning rate | Epochs | Baseline repeat | Pre-registered hypothesis |
|---:|---|---|---:|---:|---:|---|---|
| 1 | `rank-r8` | Rank | 8 | 1e-4 | 3 | Yes | Resource cost is lowest; capacity may be insufficient, but overfitting risk is lower |
| 2 | `rank-r32` | Rank | 32 | 1e-4 | 3 | No | May provide the best balance among capacity, runtime, VRAM and generalization |
| 3 | `rank-r64` | Rank | 64 | 1e-4 | 3 | No | Capacity is largest, but runtime, VRAM and overfitting risk may increase |
| 4 | `lr-1e-4` | Learning rate | 8 | 1e-4 | 3 | Yes | Reproduces the stable Week 2 baseline |
| 5 | `lr-2e-4` | Learning rate | 8 | 2e-4 | 3 | No | May converge faster, with higher oscillation or capability-regression risk |
| 6 | `lr-5e-5` | Learning rate | 8 | 5e-5 | 3 | No | More conservative updates may preserve base capability but underfit in three epochs |
| 7 | `epoch-e2` | Epoch | 8 | 1e-4 | 2 | No | Reduces runtime and overfitting exposure but may train insufficiently |
| 8 | `epoch-e3` | Epoch | 8 | 1e-4 | 3 | Yes | Reproduces the Week 2 three-epoch baseline in the new run directory |
| 9 | `epoch-e5` | Epoch | 8 | 1e-4 | 5 | No | Training fit may improve, while overfitting and general-capability drift may increase |

九次真实训练包含七个唯一超参数组合。`rank-r8`、`lr-1e-4` 和 `epoch-e3` 是相同学习参数的三次独立运行，用于测量固定随机种子下 CUDA kernel 非确定性和调度差异造成的实际运行波动，不构成新的超参数比较。

## Controlled Variables and Rank Limitation

- Rank 组只修改 `lora_rank`；
- Learning-rate 组只修改 `learning_rate`；
- Epoch 组只修改 `num_train_epochs`；
- 其余模型、数据、随机种子、模板、量化、batch、优化器和精度配置保持一致。

本实验遵循老师要求，只修改 rank 配置项并固定 `lora_alpha=16`。因此 LoRA 缩放比例 `alpha/rank` 会随 rank 改变，Rank 组结果反映不同 rank 配置下的整体训练表现，而不是严格意义上仅比较模型容量。

完整工程配置、机器可读矩阵和验证记录见 [Day 11 engineering archive](../../../deliverables/week3/day11/README.md)。
