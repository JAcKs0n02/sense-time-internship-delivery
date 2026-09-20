# Week 3 Day 12–13: Nine-Run Batch Experiments

九组冻结实验已在同一 RTX 3090 环境中串行完成。所有运行均为 `attempt-001`、退出码为 0，并通过配置哈希、有限 Loss、预期步数、adapter 文件和终态检查。

## Required Metric Summary

| Run ID | Final Loss | Last-100 Mean | Train Time | Peak VRAM MiB | Steps | Status |
|---|---:|---:|---:|---:|---:|---|
| `rank-r8` | 0.9714 | 1.061565 | 03:11:36 | 21096 | 3747 | completed |
| `rank-r32` | 0.9925 | 1.089147 | 03:12:11 | 21836 | 3747 | completed |
| `rank-r64` | 0.9968 | 1.101392 | 03:13:46 | 22682 | 3747 | completed |
| `lr-1e-4` | 0.9701 | 1.063174 | 03:10:58 | 21096 | 3747 | completed |
| `lr-2e-4` | 0.9056 | 0.876906 | 03:11:27 | 21096 | 3747 | completed |
| `lr-5e-5` | 1.0299 | 1.166453 | 03:11:22 | 21096 | 3747 | completed |
| `epoch-e2` | 1.3368 | 1.250650 | 02:07:24 | 21096 | 2498 | completed |
| `epoch-e3` | 0.9744 | 1.061932 | 03:11:11 | 21096 | 3747 | completed |
| `epoch-e5` | 0.9867 | 0.627596 | 05:18:57 | 21096 | 6245 | completed |

Final Loss 是最后一个有效 logging step 的 Loss，直接对应老师要求；Last-100 Mean 只作辅助分析，不替代 Final Loss。Train Time 是训练框架报告的训练耗时。Peak VRAM 是 `nvidia-smi` 每秒采样所得峰值，不宣称是 CUDA 瞬时精确峰值。

训练 Loss 只反映训练目标拟合程度，不能单独决定回答质量。因此即使 `lr-2e-4` 的 Final Loss 最低，本阶段也不标记“最优模型”；最终排序等待 Day 14 的固定 20 题测试和两位真实评分者盲评。

## Required Deliverables

- [Experiment Summary CSV](Results/Experiment_Summary.csv)
- [Experiment Summary JSON](Results/Experiment_Summary.json)
- [Raw Logs](Raw_Logs/)
- [Raw Log SHA-256 Manifest](Raw_Logs.sha256)

`Raw_Logs/` 共包含 108 个白名单文件，包括 source/runtime 配置、环境、预检、状态、训练日志、显存采样、Trainer state 与训练结果。该目录不包含模型权重、checkpoint、optimizer state、训练数据、缓存或符号链接。

完整调度器、严格门禁、测试和工程辅助汇总见 [Day 12–13 engineering archive](../../../deliverables/week3/day12_13/README.md)。
