# Week 3 Day 12–13 分支与教师提交设计

## 1. 目标

在不修改 `main` 的前提下，为已经完成的 Week 3 Day 12–13 批量实验建立独立分支，更新仓库导航与实际进度，并创建只包含老师当前要求交付内容的 `Submission/Week3`。

目标分支为 `codex/week3-day12-13-experiments`。该分支从已经包含 Day 11 设计及 Day 12–13 九组实验结果的当前提交创建，因此保留完整的顺序依赖和审计历史。Week 3 全部完成后，再统一决定是否合并并推送 `main`。

## 2. 需求依据

以仓库根目录的 `商汤实习需求.pages` 为教师要求原文。当前已完成部分的正式交付只有：

- Day 11：实验计划表，必须包含三组单变量实验及预注册假设；
- Day 12–13：九组实验的原始日志文件夹，并记录每组 Final Loss、训练耗时和显存占用。

Day 14–16 尚未完成，因此本次不创建对应空目录、占位报告或伪结果。

## 3. 分支策略

1. 保留现有 `codex/week3-day11-experiment-design`，作为 Day 11 阶段节点；
2. 从其当前 HEAD 创建 `codex/week3-day12-13-experiments`；
3. 所有 README 与 `Submission/Week3` 补全工作都提交到新分支；
4. 完成验证后推送新分支并设置远端跟踪；
5. 不切换、不合并、不推送 `main`。

不采用从 `main` 单独摘取 Day 12–13 提交的方式，因为训练调度、配置哈希和九组矩阵依赖 Day 11 的冻结设计。也不重命名 Day 11 分支，以免丢失清晰的阶段节点。

## 4. README 更新范围

### 4.1 根目录 `README.md`

- 将 Week 3 概述从“Day 11 已完成”更新为“Day 11–13 已完成”；
- 在 Week 3 进度表增加 Day 12–13；
- 增加九组实验全部完成、9/9 门禁通过、日志哈希通过的核心结果；
- 更新仓库结构，加入 `deliverables/week3/day12_13` 与 `Submission/Week3`；
- 增加 Day 12–13 工程交付和教师提交入口链接。

### 4.2 `deliverables/week3/README.md`

新增 Week 3 工程归档索引，只列已经完成的 Day 11 与 Day 12–13，说明 `deliverables` 与 `Submission` 的职责边界，并明确 Day 14–16 完成后再追加。

### 4.3 `deliverables/week3/day12_13/README.md`

将当前偏执行设计的说明补成实际完成说明：

- 保留调度、attempt、完成门禁和指标口径；
- 增加九组真实结果表；
- 记录 9/9 completed、退出码均为 0、108 个原始日志文件哈希验证通过；
- 明确本阶段不根据训练 Loss 选出“最优模型”，最终选择留给 Day 14 的人工盲评；
- 指向原始日志、汇总 JSON/CSV 和完整性清单。

### 4.4 `deliverables/week3/day12_13/source/results/README.md`

新增结果文件夹索引，解释五个顶层汇总文件和 `raw_logs/<run-id>/attempt-001/` 的内容、复核方法及禁止纳入 Git 的模型文件类型。

## 5. 教师提交结构

`Submission/Week3` 只包含当前老师要求检查的成果：

```text
Submission/Week3/
├── README.md
├── Day11_Experiment_Design/
│   └── README.md
└── Day12_13_Batch_Experiments/
    ├── README.md
    ├── Results/
    │   ├── Experiment_Summary.csv
    │   └── Experiment_Summary.json
    ├── Raw_Logs/
    │   └── <run-id>/attempt-001/...
    └── Raw_Logs.sha256
```

其中：

- Day 11 README 提供完整实验计划表、假设、固定变量、Rank 缩放限制和三次基线复跑解释；
- Day 12–13 README 提供老师可直接查看的九组指标表、结果口径和日志结构；
- `Experiment_Summary.csv` 是优先检查入口，JSON 用于机器复核；
- `Raw_Logs` 原样保留已经收集的 108 个白名单审计文件；
- `Raw_Logs.sha256` 按教师提交中的 `Raw_Logs/` 路径重新生成，用于逐文件完整性验证；生成后还要将每个文件的摘要与工程归档 `raw_logs.sha256` 交叉核对，确保只改变路径前缀和大小写，不改变日志内容。

本次明确不复制以下内容到 `Submission/Week3`：

- 调度、生成、汇总和测试脚本；
- pytest 测试、缓存和 `__pycache__`；
- adapter、模型权重、checkpoint、optimizer state 和训练数据；
- `baseline_variability.json` 与 `completed_adapters.json` 等工程辅助文件；
- Day 14–16 的任何空目录或占位内容。

同时更新 `Submission/README.md` 的 Week 3 入口，并重新生成 `Submission/SHA256SUMS.txt`，覆盖当前提交区的所有正式文件但不把哈希文件自身写入自身。

## 6. 数据一致性

Submission 中的实验指标不得手工重新计算或改写，必须直接复制自：

- `deliverables/week3/day12_13/source/results/experiment_summary.csv`；
- `deliverables/week3/day12_13/source/results/experiment_summary.json`。

九组运行必须保持以下集合：

`rank-r8`、`rank-r32`、`rank-r64`、`lr-1e-4`、`lr-2e-4`、`lr-5e-5`、`epoch-e2`、`epoch-e3`、`epoch-e5`。

所有运行必须满足 `status=completed`、`exit_code=0`，并同时存在 Final Loss、训练耗时和显存采样峰值。README 表格只能引用这些冻结结果。

## 7. 验证与完成条件

实施完成前必须通过：

1. Day 11 与 Day 12–13 的全部本地测试；
2. 仓库回归测试（明确排除仅因本机未安装 PyTorch 而无法收集的历史 Day 2 GPU 测试）；
3. 工程归档与 Submission 的九组运行集合、关键指标逐字段一致；
4. 工程归档和 Submission 各自的 108 条原始日志 SHA-256 全部通过；
5. Submission 中不存在权重、checkpoint、optimizer state、缓存或未完成日期目录；
6. 所有 README 相对链接有效；
7. `git diff --check` 通过且工作区只包含本次范围内的修改；
8. 提交成功，并将 `codex/week3-day12-13-experiments` 推送到 `origin`；
9. `main` 与 `origin/main` 均保持不变。
