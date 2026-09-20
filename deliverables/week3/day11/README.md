# Week 3 Day 11：单变量对比实验设计

Day 11 已完成 9 次 QLoRA 正式训练的实验矩阵、基线配置、配置生成器、自动校验器和 AutoDL RTX 3090 smoke test。本日只冻结实验设计，不把 smoke test 当作正式实验，也不提前声称任何超参数更优。

## 1. 老师要求与完成方式

| 老师要求 | 本次实现 |
|---|---|
| Rank：`r=8 / 32 / 64` | Rank 组 3 次，其他学习参数固定为基线 |
| 学习率：`1e-4 / 2e-4 / 5e-5` | Learning-rate 组 3 次，Rank 和 epoch 固定为基线 |
| 训练轮数：`2 / 3 / 5` | Epoch 组 3 次，Rank 和学习率固定为基线 |
| 固定随机种子和数据集 | 所有正式配置固定 `seed=42`，使用 Week 2 Day 7 的 4,999 条清洗后 Alpaca 数据 |
| 交付实验计划表（含假设） | 下表、机器可读 manifest、9 份生成后的 YAML 和验证报告共同构成交付 |

## 2. 实验计划表

基线为 `rank=8`、`learning_rate=1e-4`、`epochs=3`、`lora_alpha=16`、`seed=42`。

| 顺序 | Run ID | 实验组 | Rank | 学习率 | Epoch | 基线复跑 | 预注册假设 |
|---:|---|---|---:|---:|---:|---|---|
| 1 | `rank-r8` | Rank | 8 | 1e-4 | 3 | 是 | 参数量和资源开销最低；可能欠拟合，但过拟合风险较低 |
| 2 | `rank-r32` | Rank | 32 | 1e-4 | 3 | 否 | 可能在容量、耗时、显存和泛化之间取得较好平衡 |
| 3 | `rank-r64` | Rank | 64 | 1e-4 | 3 | 否 | 容量最大，但耗时、显存和过拟合风险可能更高 |
| 4 | `lr-1e-4` | Learning rate | 8 | 1e-4 | 3 | 是 | 复现 Week 2 已成功的稳定基线 |
| 5 | `lr-2e-4` | Learning rate | 8 | 2e-4 | 3 | 否 | 可能收敛更快，但 loss 波动和能力退化风险更高 |
| 6 | `lr-5e-5` | Learning rate | 8 | 5e-5 | 3 | 否 | 更新更保守，可能更好保留基座能力，但三轮内可能欠拟合 |
| 7 | `epoch-e2` | Epoch | 8 | 1e-4 | 2 | 否 | 耗时和过拟合暴露更少，但可能训练不足 |
| 8 | `epoch-e3` | Epoch | 8 | 1e-4 | 3 | 是 | 在新目录中复现 Week 2 三轮基线 |
| 9 | `epoch-e5` | Epoch | 8 | 1e-4 | 5 | 否 | 训练拟合可能改善，但过拟合和通用能力漂移风险增加 |

这里有 9 次真实训练，但只有 7 个唯一超参数组合。`rank-r8`、`lr-1e-4` 和 `epoch-e3` 是相同基线的三次独立运行。重复运行用于评估训练过程在固定随机种子下的实际运行波动，例如 CUDA kernel 非确定性和运行调度差异，不参与新的超参数比较。

## 3. 单变量控制与 Rank 限制

三个实验组分别只允许修改一个学习参数：

- Rank 组只修改 `lora_rank`；
- Learning-rate 组只修改 `learning_rate`；
- Epoch 组只修改 `num_train_epochs`。

`run_name` 和 `output_dir` 是用于隔离运行的元数据，不是学习参数。校验器会把每份 YAML 与基线和 manifest 逐字段比较，任何未授权差异都会使验证失败。

本实验遵循老师要求，仅修改 rank 配置项，因此固定 `lora_alpha=16`。这意味着 LoRA 缩放比例 `alpha/rank` 会随 rank 改变，因此实验结果主要反映不同 rank 配置下的整体训练表现，而不是严格意义上仅比较模型容量。这一限制将在结果分析中明确说明。

## 4. 固定控制变量

| 类别 | 固定值 |
|---|---|
| 基座模型 | 本地固定的 Qwen2.5-7B-Instruct |
| 训练数据 | Week 2 清洗后的 4,999 条 Alpaca 数据；不重复加入等价 ShareGPT 副本 |
| 随机性 | `seed=42` |
| 输入 | `template=qwen`，`cutoff_len=2048` |
| QLoRA | 4-bit bitsandbytes、NF4、double quantization、`lora_target=all`、`lora_alpha=16`、dropout 0.05 |
| Batch | micro batch 1、gradient accumulation 4，有效 batch size 4 |
| 优化 | cosine scheduler、warmup ratio 0.1、weight decay 0.01、max grad norm 1.0 |
| 精度 | BF16，关闭 FP16，启用 gradient checkpointing |
| 记录 | 每个 optimizer step 记录 loss；每 500 steps 保存 checkpoint；TensorBoard 开启 |

固定随机种子意味着伪随机抽样从同一初始状态开始，但不能保证 GPU 训练逐位完全一致。某些 CUDA kernel、并行归约和系统调度仍可能产生微小波动，这正是三个基线复跑需要测量的内容。

## 5. 结果指标的口径

Day 12–13 对每个正式 run 统一记录：

- **Final Loss**：最后一个有效 logging step 的 loss，直接满足老师“最终 Loss”的要求；
- **最近 100 个 logging steps 平均 loss**：仅用于降低末步随机波动对观察的影响，不替代 Final Loss；
- 训练总耗时；
- 1 秒采样的峰值显存；
- optimizer steps、退出码、adapter 路径和文件清单。

训练 loss 只反映训练目标的拟合程度，不能单独代表回答质量。Day 14 的自动评测仅提供数学答案、代码测试和格式检查等客观辅助证据，不直接决定模型排序；真正的模型排序由两位真实评分者按冻结评分卡盲评完成。OpenCompass 也不参与超参数选择，只用于验证 Day 14 已选模型在公开基准上的泛化能力。

## 6. 交付文件

| 文件 | 用途 |
|---|---|
| `source/manifests/experiment_matrix.json` | 9 行实验顺序、假设、固定控制和解释限制的唯一机器可读来源 |
| `configs/qwen25_7b_week3_base.yaml` | 从 Week 2 成功配置派生的不可变基线 |
| `configs/experiments/*.yaml` | 由生成器创建的 9 份正式训练配置 |
| `source/scripts/generate_experiment_configs.py` | 从基线和 manifest 确定性生成正式 YAML |
| `source/scripts/validate_experiment_matrix.py` | 校验取值、重复基线、单变量差异、文件名和输出目录 |
| `source/results/day11_validation.json` | 校验结论及 manifest、基线、9 份配置的 SHA-256 |
| `source/results/day11_files.sha256` | Day 11 本地交付文件的冻结哈希清单 |
| `source/results/remote_evidence/` | 新实例连续性审计、远端预检、完整 smoke 日志、状态、指标和证据包 |
| `configs/qwen25_7b_week3_smoke.yaml` | 仅使用 8 条数据的独立 GPU smoke 配置 |
| `source/scripts/run_day11_smoke.sh` | AutoDL 预检、短训练、日志和状态采集脚本 |

## 7. 本地复现与验证

重新生成正式配置时，输出目录必须为空；生成器默认拒绝覆盖已有 YAML：

```bash
python deliverables/week3/day11/source/scripts/generate_experiment_configs.py \
  --matrix deliverables/week3/day11/source/manifests/experiment_matrix.json \
  --base deliverables/week3/day11/configs/qwen25_7b_week3_base.yaml \
  --output deliverables/week3/day11/configs/experiments
```

验证矩阵和配置：

```bash
python deliverables/week3/day11/source/scripts/validate_experiment_matrix.py \
  --matrix deliverables/week3/day11/source/manifests/experiment_matrix.json \
  --base deliverables/week3/day11/configs/qwen25_7b_week3_base.yaml \
  --config-dir deliverables/week3/day11/configs/experiments \
  --output deliverables/week3/day11/source/results/day11_validation.json

pytest -q deliverables/week3/day11/source/tests/test_experiment_matrix.py
```

当前本地结果为 `valid: true`、9 个 run、三个实验组各 3 个 run、7 个唯一组合，专项测试 8/8 通过。

## 8. AutoDL smoke test

Smoke test 只检查数据注册、模型加载、4-bit QLoRA 初始化和短训练链路，不参与 9 组实验。它使用 `max_samples=8`、1 epoch、独立路径 `/root/autodl-tmp/qwen25-week3/smoke/day11-qlora-smoke`，并拒绝覆盖非空输出目录。正式路径 `/root/autodl-tmp/qwen25-week3/runs/` 不会被 smoke 脚本写入。

将配置、脚本和固定数据同步到 AutoDL 后执行：

```bash
bash /root/autodl-tmp/qwen25-week3/scripts/run_day11_smoke.sh
```

脚本已于 2026-08-03 在新克隆的 AutoDL 实例
`78fb4ea487-e700080a` 上完成。开始训练前先等待数据盘复制完成，再验证
Week 1/2 路径、训练数据、Day 8 adapter 和 Day 9 合并权重；关键权重哈希
与本地归档完全一致。

| Smoke 指标 | 实际结果 |
|---|---:|
| GPU | NVIDIA GeForce RTX 3090，24 GB |
| 状态 / 退出码 | `completed` / 0 |
| 样本数 / Epoch | 8 / 1 |
| Optimizer steps | 2 / 2 |
| Train loss | 1.8717711567878723 |
| Train runtime | 5.3726 秒 |
| Adapter 大小 | 80,792,096 bytes |
| Adapter SHA-256 | `62cc225bfe327cee9398c9e61bb277c480ed395544e3c2b59a132a28f03f6fe9` |
| 远端输出 | `/root/autodl-tmp/qwen25-week3/smoke/day11-qlora-smoke` |

完整审计和运行证据见
[`source/results/remote_evidence/README.md`](source/results/remote_evidence/README.md)。
Smoke 输出与正式路径 `/root/autodl-tmp/qwen25-week3/runs/` 完全隔离，不能计入
九次实验。

## 9. 专业词汇解释

- **超参数（hyperparameter）**：训练开始前由实验者设定的参数，例如 Rank、学习率和 epoch；它们不是模型从数据中自动学习出来的权重。
- **LoRA Rank**：低秩适配矩阵的秩，可近似理解为 adapter 用于表达任务变化的容量。Rank 越大，通常可训练参数、显存和计算开销越高。
- **`lora_alpha` 与 `alpha/rank`**：`lora_alpha` 控制 LoRA 更新的缩放；在当前实现中缩放与 `alpha/rank` 相关，所以只改 Rank 也会改变缩放比例。
- **学习率（learning rate）**：每次优化器更新权重的步长。过大可能震荡或破坏原能力，过小可能在有限训练时间内学不够。
- **Epoch**：完整遍历一次训练数据。5 epochs 表示每条样本大致被模型看到五次。
- **随机种子（seed）**：初始化伪随机过程的数字，用于提高重复实验的可比性；它不等于 GPU 逐位确定性。
- **控制变量**：为了把结果差异主要归因于目标超参数，实验中保持不变的模型、数据、batch、精度等条件。
- **Smoke test**：正式运行前的小样本短测试，用来尽早发现路径、依赖、显存或数据格式错误，不产生正式结论。
- **Adapter**：LoRA 训练得到的小型增量权重；基座模型保持冻结，推理时将 adapter 与同一基座一起加载。

## 10. Day 11 验收结论

- [x] 三组取值与老师要求完全一致；
- [x] 9 次正式运行、7 个唯一组合和 3 次基线复跑均已明确；
- [x] 数据、seed 和其余训练条件已冻结；
- [x] 每一行都有预注册假设；
- [x] Rank 缩放限制、基线复跑目的、Final Loss 口径、自动评测和 OpenCompass 边界均已写明；
- [x] 生成器、验证器、9 份 YAML 和自动测试已完成；
- [x] AutoDL 真实 GPU smoke 退出码为 0，环境、日志、状态、adapter 哈希和远端路径均已归档。
