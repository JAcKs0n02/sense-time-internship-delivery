# Week 3 Day 12–13：九次 QLoRA 批量训练

Day 12–13 是一个连续执行阶段：Day 12 完成调度、汇总和证据收集工具并启动冻结的九次实验；Day 13 继续监控同一个矩阵，核验九个终态，汇总老师要求的 Final Loss、训练耗时和显存，并回收原始日志。两天不会启动两套独立矩阵。

本阶段现已完成：九组正式运行均为 `completed`，退出码均为 0；老师要求的三项指标和辅助指标已汇总，108 个原始日志与状态文件全部通过 SHA-256 校验。

## 1. 冻结输入

- 实验矩阵：`../day11/source/manifests/experiment_matrix.json`
- Day 11 哈希验证：`../day11/source/results/day11_validation.json`
- 基线配置：`../day11/configs/qwen25_7b_week3_base.yaml`
- 九份正式配置：`../day11/configs/experiments/*.yaml`
- 基座模型：`/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct`
- 数据：Week 2 Day 7 清洗后的 4,999 条 Alpaca，`seed=42`

九次运行严格按 manifest 顺序串行执行。`rank-r8`、`lr-1e-4` 和 `epoch-e3` 是相同学习参数的三次独立基线复跑，只测量实际运行波动。

## 2. 工具

| 文件 | 职责 |
|---|---|
| `source/scripts/run_experiments.py` | 校验冻结哈希、创建 attempt、执行资源门禁、串行训练、采样显存和写原子状态 |
| `source/scripts/summarize_experiments.py` | 严格核验退出码、步数、loss、耗时和 adapter，生成九行汇总与基线波动 |
| `source/scripts/collect_experiment_evidence.py` | 只回收日志和状态白名单，拒绝权重、checkpoint、optimizer state、训练数据和符号链接 |
| `source/tests/` | 覆盖哈希篡改、自动重跑、跨 run 恢复、资源不足、硬超时、NaN/Inf、步数不足和权重篡改 |

## 3. 运行安全边界

- Day 11 YAML 是不可修改的 source config；每次 attempt 使用 runtime config。
- 普通 runtime config 只改变 `output_dir` 和 `run_name`。
- 首次运行使用 `attempt-001`；失败不会自动重跑或覆盖，显式重跑使用下一 attempt。
- 每个 run 的硬超时默认为 12 小时。
- 启动门禁：无其他 GPU compute 进程、空闲显存至少 20,000 MiB、数据盘至少 15 GiB 可用。
- 显存为 `nvidia-smi` 每秒一次的采样峰值，不宣称是 CUDA 精确瞬时峰值。

## 4. 成功判定

调度器的 `completed` 只是初步状态。汇总器仅在下列条件全部满足时保留 `completed`：

1. 退出码为 0；
2. source/runtime 配置哈希与状态记录一致；
3. `trainer_state.json`、`train_results.json` 可解析；
4. 所有 logging loss 均为有限值；
5. optimizer steps 达到冻结预期：2 epochs = 2,498、3 epochs = 3,747、5 epochs = 6,245；
6. adapter 配置和非空权重存在；
7. `adapter_files.sha256` 校验通过；
8. 没有中断或失败原因。

Final Loss 是最后一个有效 logging step 的 loss。最近 100 个 logging steps 平均 loss 只作辅助，不替代 Final Loss。`train_runtime_seconds` 是老师表格使用的框架训练耗时；`wall_clock_runtime_seconds` 是包含加载、预处理和保存的审计耗时。

## 5. 实际结果

| Run ID | Final Loss | 最近 100 步平均 Loss | 训练耗时 | 采样峰值显存（MiB） | Optimizer steps | 状态 |
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

完整机器可读结果见 [`experiment_summary.csv`](source/results/experiment_summary.csv) 和 [`experiment_summary.json`](source/results/experiment_summary.json)。Loss 是训练拟合指标，不能单独决定回答质量；即使 `lr-2e-4` 的 Final Loss 最低，本阶段也不据此标记“最优模型”。最终超参数选择仍由 Day 14 固定 20 题测试和两位真实评分者的盲评决定。

## 6. 完成门禁与完整性

- 9/9 run 满足严格 `completed` 条件，退出码均为 0；
- optimizer steps 与冻结预期完全一致；
- 108/108 个回收文件通过 [`raw_logs.sha256`](source/results/raw_logs.sha256)；
- 原始日志位于 [`source/results/raw_logs/`](source/results/raw_logs/)；
- 回收目录不含权重、checkpoint、optimizer state、训练数据或符号链接。

## 7. 本地测试

```bash
python -m pytest -q deliverables/week3/day12_13/source/tests
python -m py_compile deliverables/week3/day12_13/source/scripts/*.py
```

本地测试不启动 GPU 或 LLaMA-Factory 正式训练。

## 8. 远端启动接口

```bash
nohup python /root/autodl-tmp/qwen25-week3/scripts/run_experiments.py \
  --matrix /root/autodl-tmp/qwen25-week3/manifests/experiment_matrix.json \
  --validation /root/autodl-tmp/qwen25-week3/manifests/day11_validation.json \
  --base-config /root/autodl-tmp/qwen25-week3/configs/qwen25_7b_week3_base.yaml \
  --config-dir /root/autodl-tmp/qwen25-week3/configs/experiments \
  --run-root /root/autodl-tmp/qwen25-week3/runs \
  --timeout-hours 12 \
  > /root/autodl-tmp/qwen25-week3/logs/matrix_scheduler.log 2>&1 &
```

状态查询使用同一调度器的 `--status-only`，不会启动训练。

## 9. 实际交付

```text
source/results/
├── experiment_summary.csv
├── experiment_summary.json
├── baseline_variability.json
├── completed_adapters.json
├── raw_logs.sha256
└── raw_logs/<run-id>/attempt-<NNN>/
```

Git 只保存小型审计文件和原始文本日志，不保存 `.safetensors`、`.bin`、checkpoint、optimizer state 或训练数据。完整 adapter 保留在 AutoDL 数据盘，供 Day 14 加载。
