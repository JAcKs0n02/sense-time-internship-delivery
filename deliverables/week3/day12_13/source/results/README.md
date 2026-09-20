# Day 12–13 Results

本目录保存九次正式 QLoRA 训练的指标汇总和小型原始审计文件。完整 adapter 继续保留在 AutoDL 数据盘；Git 只保存复核实验状态所需的文本与 JSON/CSV 证据。

## 顶层文件

| 文件 | 内容 |
|---|---|
| `experiment_summary.csv` | 老师优先检查的九行表格：Final Loss、最近 100 步平均 Loss、训练耗时、显存、步数和状态 |
| `experiment_summary.json` | 与 CSV 等价的机器可读完整记录，额外包含配置哈希、adapter 路径和审计耗时 |
| `baseline_variability.json` | 三次相同基线运行的实际波动统计，只用于一致性分析 |
| `completed_adapters.json` | 九个已完成 adapter 的远端路径与文件哈希清单路径，供 Day 14 加载 |
| `raw_logs.sha256` | 108 个回收文件的 SHA-256 完整性清单 |
| `raw_logs/<run-id>/attempt-001/` | 每次运行的配置、环境、预检、状态、训练日志、显存采样和 Trainer 结果 |

## 完整性复核

从仓库根目录执行：

```bash
cd deliverables/week3/day12_13/source/results
shasum -a 256 -c raw_logs.sha256
```

预期输出 108 行 `OK`。九组运行集合固定为 `rank-r8`、`rank-r32`、`rank-r64`、`lr-1e-4`、`lr-2e-4`、`lr-5e-5`、`epoch-e2`、`epoch-e3` 和 `epoch-e5`。

## 排除项

本目录不得加入 `.safetensors`、`.bin`、`checkpoint-*`、`optimizer.pt`、训练数据、缓存或符号链接。训练权重不属于 Day 12–13 的 Git 交付，最优 adapter 将在 Day 16 按老师要求单独归档。
