# Week 3 SFT 超参数优化与评估设计

## 1. 设计目的

本设计把老师的 Week 3 原始要求转成一套可执行、可审计、可复现的工程方案。范围严格限定为 Day 11–Day 16：完成 9 次单变量 SFT 对比训练，用固定 20 题评测集和双人盲评选出最优 SFT 模型，再用 OpenCompass 0.5.3 对基座与最优模型运行 CEval、CMMLU，最后形成周报和下周 DPO 输入归档。

本周不把训练 loss 当成模型质量结论，不把重复配置隐藏成不同超参数，也不为追求更大的实验数量扩展到 27 次全因子实验。

## 2. 原始要求核对结论

老师要求的三个实验因素各有三个取值：

- LoRA rank：`8`、`32`、`64`。
- 学习率：`1e-4`、`2e-4`、`5e-5`。
- 训练轮数：`2`、`3`、`5`。

Day 12–13 明确要求“依次运行上述 9 组实验”，因此本设计执行 9 次真实训练。三组实验都围绕 Week 2 已验证基线 `rank=8, learning_rate=1e-4, epochs=3`，所以该基线会以不同 run ID 重复三次。重复运行不是三个不同超参数配置，而是三个实际实验重复，用于检查固定 seed 后仍可能存在的运行时非确定性。

这比只跑 7 个唯一配置更忠实于“9 组实验”，也比 27 次全因子实验更符合本周范围和单张 RTX 3090 的时间预算。

## 3. 已知基线与资源约束

沿用 Week 2 已验证事实：

| 项目 | 固定值或事实 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090，24GB |
| 基座模型 | Qwen2.5-7B-Instruct |
| 训练框架 | LLaMA-Factory 0.9.3 |
| 训练方式 | 4-bit NF4 QLoRA，`lora_target=all` |
| 数据 | Day 7 清洗后的 4,999 条 Alpaca 表示 |
| 数据长度 | Qwen chat template 后最多 2,048 tokens |
| Week 2 基线 | rank 8、alpha 16、learning rate `1e-4`、3 epochs |
| 基线训练结果 | 3,747 optimizer steps，耗时 3:12:12，最终 adapter 80,792,096 bytes |
| 基线质量限制 | 三题盲评基座 25/30、Week 2 合并模型 22/30，未证明优于基座 |
| 数据盘 | 100GB；Day 9 导出前约 75GB 可用 |

按 Week 2 实测速度估算，9 次训练约需 30 个 GPU 小时，rank 增大还可能略微增加耗时与显存。九个完整合并模型约需 137GB，仅权重就会超过数据盘余量。因此 Day 12–14 保留九个 adapter，Day 14 以“基座 + adapter”方式逐个评测，Day 15 前只合并一个已选出的最优 adapter。

## 4. 九次训练矩阵

除表中实验变量、`run_name`、`output_dir` 和日志路径外，其余训练条件保持一致。三个路径字段只负责产物隔离，不属于学习变量。

| Run ID | 实验组 | rank | learning rate | epochs | 与组内基线相比的唯一学习变量 | 假设 |
|---|---|---:|---:|---:|---|---|
| `rank-r8` | Rank | 8 | `1e-4` | 3 | rank | 资源最低，可能容量不足，但更不易过拟合 |
| `rank-r32` | Rank | 32 | `1e-4` | 3 | rank | 容量与资源之间可能取得更好平衡 |
| `rank-r64` | Rank | 64 | `1e-4` | 3 | rank | 训练拟合能力最强，但显存、耗时和过拟合风险最高 |
| `lr-1e-4` | Learning rate | 8 | `1e-4` | 3 | learning rate | Week 2 已验证稳定，是学习率组基线 |
| `lr-2e-4` | Learning rate | 8 | `2e-4` | 3 | learning rate | 收敛可能更快，也更容易振荡或损伤通用能力 |
| `lr-5e-5` | Learning rate | 8 | `5e-5` | 3 | learning rate | 更新更保守，可能保留基座能力但出现欠拟合 |
| `epoch-e2` | Epoch | 8 | `1e-4` | 2 | epochs | 时间较短，可能减少过拟合但学习不足 |
| `epoch-e3` | Epoch | 8 | `1e-4` | 3 | epochs | Week 2 已验证基线 |
| `epoch-e5` | Epoch | 8 | `1e-4` | 5 | epochs | 可能进一步拟合，也可能加重能力偏移和过拟合 |

本实验遵循老师要求，仅修改 rank 配置项，因此固定 `lora_alpha=16`。这意味着 LoRA 缩放比例 `alpha/rank` 会随 rank 改变，因此实验结果主要反映不同 rank 配置下的整体训练表现，而不是严格意义上仅比较模型容量。这一限制将在结果分析中明确说明。

三个基线运行的学习参数完全相同，但输出目录和 run ID 不同。重复运行用于评估训练过程在固定随机种子下的实际运行波动，例如 CUDA kernel 非确定性和运行调度差异，不参与新的超参数比较。分析时既分别报告，也额外计算三次基线的 loss、耗时和峰值显存离散程度。不得把它们误报为三个唯一超参数组合。

## 5. 实验控制与可复现性

所有运行固定：

- 同一个基座模型目录及文件哈希。
- 同一个 4,999 条训练数据快照及 SHA-256。
- 同一个 `dataset_info.json`、Qwen template 和 `cutoff_len=2048`。
- `seed=42`，并固定当前版本支持的 Python、NumPy、PyTorch 和数据顺序 seed。
- 相同的 micro batch、gradient accumulation、scheduler、warmup、weight decay、dropout、LoRA target、量化方式、BF16 和 gradient checkpointing。
- 相同的 LLaMA-Factory、Transformers、PyTorch、CUDA、bitsandbytes 和驱动版本。
- 同一张 GPU 串行运行，不并发训练，不在训练中启动模型推理。

Day 11 生成一份不可变实验清单。验证器逐组比较 YAML：只有目标变量和隔离产物所需的名称、目录字段可以变化。每个配置启动前保存规范化参数快照和 SHA-256，训练结束后再从 `trainer_state.json`、adapter 配置和日志反查实际值。

固定 seed 不能保证所有 GPU kernel 位级确定，因此“可复现”定义为输入、配置、版本和调度可复核，并对重复基线的实际差异如实报告，而不是预先宣称三个 adapter 必须逐字节一致。

## 6. 批量训练架构

Day 12–13 作为一个连续阶段完成：Day 12 实现并验证调度器、日志解析器，冻结输入后启动矩阵；Day 13 继续监控同一个后台任务，完成终态核验、原始日志回收和指标汇总。两天共用 `day12_13/` 目录和同一套证据契约，不启动两套相互独立的矩阵。

顺序调度器读取实验清单，每次只启动一个 LLaMA-Factory 训练进程。调度器必须同时读取 Day 11 的 `day11_validation.json`，以其中的 `matrix_sha256`、`base_config_sha256` 和逐 run `config_sha256` 作为冻结基准；只重新计算哈希而没有预期值不构成验证。任何 manifest、基线或正式 YAML 哈希不一致都在启动 GPU 前失败，不允许边训练边修正输入。

每个正式 run 使用两层配置：Day 11 YAML 是不可修改的 **source config**；调度器为每次尝试生成 **runtime config**，普通 attempt 只允许把 `output_dir` 和 `run_name` 改到独立的 `runs/<run-id>/attempt-<NNN>/`。`status.json` 同时记录 source config 路径与 SHA-256、runtime config 路径与 SHA-256、两者规范化差异。首次运行是 `attempt-001`；失败或中断不会被自动重试，也不会被覆盖，新的 attempt 必须在诊断后显式启动。若显式批准从兼容 checkpoint 恢复，runtime config 才可额外修改 `resume_from_checkpoint`；checkpoint 必须来自同一 run、同一 source config，并记录路径、SHA-256、中断 step 和恢复边界。新鲜重跑不得带入该字段。

每个 attempt 独立保存：

- source/runtime YAML、两层配置哈希及允许差异报告。
- 启动、结束时间及退出码。
- 未裁剪的标准输出和标准错误日志。
- 1 秒间隔的 `nvidia-smi` 显存采样日志及采样峰值。
- `trainer_state.json` 导出的逐步 loss、learning rate、epoch 和 grad norm。
- 最后一个有效 loss、末 100 步均值、全局均值、最小值、最大值、斜率和 NaN/Inf 计数。
- adapter 逐文件 SHA-256 清单、总大小和服务器绝对路径。
- 数据、模型、软件环境和命令快照。

一个 attempt 只有在以下条件全部满足时才能标记为 `completed`：训练进程退出码为 0；`trainer_state.json` 和训练结果文件存在且可解析；所有已记录 loss 均为有限值；`global_step` 达到由样本数、有效 batch 和 epoch 计算的预期 optimizer steps；adapter 配置与至少一个非空权重文件存在；状态中没有中断或失败原因。任何一项不满足都保留真实状态，不以最后一个有限 loss 或部分 adapter 冒充成功。

为满足老师要求，实验记录中保留“最后一个有效 logging step 的 loss”作为 **Final Loss**；同时额外报告最近 100 个 logging step 的平均 loss，仅作为辅助分析指标，不替代 Final Loss。耗时同时记录两种口径：`train_runtime_seconds` 取自训练框架结果，作为正式“训练耗时”；`wall_clock_runtime_seconds` 由调度器的单调时钟计算，覆盖加载、预处理、训练和保存，作为审计辅助。显存占用明确写成“1 秒采样峰值”，不伪装成 CUDA 分配器的精确瞬时峰值。

每个 attempt 启动前必须通过资源门禁：目标 GPU 上没有其他计算进程；空闲显存不少于 20,000 MiB；`/root/autodl-tmp` 可用空间不少于 15 GiB；`nvidia-smi` 和数据、模型路径检查成功。显存采样使用 `--format=csv,noheader,nounits`，记录 GPU 总显存使用而不是宣称进程级精确值。任一门禁失败时停止启动后续训练并写入 `blocked_preflight`，不得自动删除 checkpoint 或既有证据。

老师要求的“实验原始日志文件夹”既保留在 AutoDL，也回收到仓库的 `deliverables/week3/day12_13/source/results/raw_logs/<run-id>/attempt-<NNN>/`。本地副本包含日志、状态、显存采样、trainer 状态/结果、实际配置、环境快照和 adapter 文件清单，但不包含 adapter 权重、optimizer state、checkpoint 或训练数据；全部回收文件进入 `raw_logs.sha256`。

调度器支持识别已完成 run 并继续清单中尚未开始的 run；它不会覆盖失败 attempt，也不会静默重试崩溃实验。某个 run 失败后，只要 GPU 和磁盘门禁仍通过，调度器可继续执行下一个不同 run；该失败 run 是否创建新 attempt 需要显式决定。smoke test、失败 attempt 和正式成功 attempt 分区保存，smoke test 不计入九次正式实验。目标仍是九次成功训练；“至少三组有效对比”是最终最低验收线，不是提前停止剩余计划运行的理由。

## 7. Day 14 固定评测与盲评

### 7.1 二十题评测集

在读取任何 Week 3 模型回答前冻结 20 题：数学 7 题、逻辑与约束推理 7 题、代码 6 题。每题包含唯一 ID、类别、难度说明、原始 messages、参考答案或关键点、自动判定规则和人工评分注意事项。

数学题优先采用可确定验证的数值或推导结果；推理题包含多约束、反事实和格式遵循；代码题要求生成可执行函数，并配套隔离、限时的单元测试。对 4,999 条训练数据执行精确匹配和相似度检查，保存题目未进入训练集的证据。

### 7.2 推理一致性

基座和九个 adapter 使用同一个 Qwen chat template、system prompt、确定性解码、`max_new_tokens`、停止条件、精度和设备。每次保存输入 token 数、原始输出、输出 token 数、耗时、退出状态和模型标识。评测失败保留错误记录，不以空回答替换，也不只重跑表现差的模型。

### 7.3 五维评分卡

每题的五个维度均按 0–5 分评分，再使用老师给定权重：

- 准确性 30%。
- 完整性 25%。
- 逻辑性 20%。
- 安全性 15%。
- 格式 10%。

评分指南为每个分值提供行为锚点。自动评测仅提供客观辅助证据，例如数学答案核对、代码测试和格式检查，不直接决定模型排序，也不冒充第二位人工评分者。真正决定模型排序的是两位真实评分者的人工评分。

由两位真实评分者独立完成盲评，其中一位必须是同事或导师。盲评文件只显示随机化模型代号；代号映射在两份评分均冻结后才解盲。若第二位评分者未完成，不生成虚构分数，Day 14 的人工盲评交付保持未通过状态。

### 7.4 最优模型规则

最优 SFT 模型在查看结果前按以下规则确定：

1. 第一排序键：两位评分者的加权总分平均值。
2. 分数相同：两位评分者在准确性维度的平均分更高者优先。
3. 仍相同：两位评分者的总分分歧更小者优先。
4. 仍相同：adapter 更小、训练耗时更短者优先。

自动评测结果不会进入上述排序键，只作为人工评分和结果解释的辅助证据。最终 loss 也不参与质量主排序，只用于解释训练行为。最优模型必须从九个 SFT run 中选择；同时单独比较它与基座。如果所有 SFT 模型都不如基座，报告必须如实写出，不能把“组内最优”改写为“优于基座”。

雷达图展示每个模型在五个维度上的两人平均分，并同时提供 CSV/JSON 原始数值，避免只有图片无法复核。

## 8. Day 15 OpenCompass 评测

运行前固定 OpenCompass 0.5.3、数据集配置名、数据来源和 revision。先用当前安装版本的 `tools/list_configs.py` 确认 CEval、CMMLU 的实际配置名，再执行 dry run、最小 smoke test 和正式评测。Qwen2.5-Instruct 使用生成式 chat 配置，不把面向 base 模型的 PPL 配置与 chat 评测混用。

OpenCompass 0.5.3 官方文档支持 HuggingFace 本地模型路径、`--hf-type chat`、CEval/CMMLU 配置以及 CSV/TXT 汇总：

- [OpenCompass 0.5.3 Quick Start](https://opencompass.readthedocs.io/en/stable/get_started/quick_start.html)
- [Prepare Models](https://opencompass.readthedocs.io/en/latest/user_guides/models.html)
- [Task Execution and Monitoring](https://opencompass.readthedocs.io/en/latest/user_guides/experimentation.html)

正式评测对象只有：

1. 原始 Qwen2.5-7B-Instruct 基座模型。
2. Day 14 规则选出的最优 SFT adapter 合并模型。

在 OpenCompass 前只合并最优 adapter，并验证合并目录可独立加载。两个模型使用相同 CEval、CMMLU 配置、推理参数、数据快照和评测版本。保存完整命令、配置、原始预测、每学科分数、聚合分数、CSV/TXT 汇总、退出码和路径清单。

OpenCompass 结果是外部通用能力验证，不回头改变 Day 14 已冻结的选模规则。若最优 SFT 在 CEval 或 CMMLU 下降，报告分析能力迁移与退化，不临时换另一个模型重跑后只展示更好结果。

**OpenCompass 不参与超参数选择，仅用于验证 Day 14 已选模型在公开基准上的泛化能力。**

## 9. Day 16 报告与归档

《第3周：SFT优化与评估报告》覆盖：

1. 原始要求和验收矩阵。
2. Week 2 基线、资源与实验控制。
3. 九次实验矩阵、假设和配置差异验证。
4. loss、耗时、显存及重复基线的复现性结果。
5. 20 题设计、训练集污染检查和自动评测方法。
6. 五维双人盲评、评分一致性、雷达图和选模规则。
7. 最优 SFT 与基座的公平对比。
8. OpenCompass CEval、CMMLU 配置与结果。
9. 失败实验、异常、限制和不支持的结论。
10. 最优模型归档、复现命令和下周 DPO 输入接口。

最优模型归档包含基座标识、adapter、独立合并模型、训练配置、数据哈希、环境清单、Day 14 评测结果、OpenCompass 结果、文件清单和 SHA-256。Git 仓库只保存可审计的小文件、脚本、报告和大文件清单；完整 adapter 与合并权重保留在 AutoDL 数据盘。

下周 DPO 输入说明必须明确究竟使用“基座 + 最优 adapter”还是已合并模型，并记录绝对路径和哈希，禁止只写“best model”这种无法复现的别名。

## 10. 产物边界

计划中的本地工程入口按现有仓库模式建立：

```text
deliverables/week3/
├── day11/      # 实验矩阵、假设、配置生成与验证
├── day12_13/   # 调度器、九次原始日志、指标和 adapter 清单
├── day14/      # 20 题、eval_harness、自动结果、盲评、雷达图
├── day15/      # OpenCompass 配置、日志、原始结果和分数表
└── day16/      # 周报、验收矩阵、最优模型路径和最终清单

Submission/Week3/  # 面向老师的精简正式提交
```

远端统一使用 `/root/autodl-tmp/qwen25-week3/`，并按 `configs/`、`runs/`、`eval/`、`opencompass/`、`best_model/`、`logs/` 分区。任何正式 run 都不能写入 Week 1、Week 2 的目录。

## 11. 故障与恢复原则

- 训练 OOM：先检查残留进程和实际配置，只改变一个资源变量并保留失败日志；如果改变学习条件，该 run 作废并从原始配置重新开始。
- 训练中断：仅从同一 run 的兼容 checkpoint 恢复；记录中断和恢复边界。
- loss 为 NaN/Inf：该 run 标记无效，不以最后一个有限值冒充最终 loss。
- 磁盘不足：停止后续 run，先生成清单确认可删除对象；未经确认不删除已有正式证据。
- 评测进程失败：保留逐题状态，按预定义规则整组补跑，不选择性补跑低分回答。
- 代码题：在无网络、限时、限制资源的隔离进程中执行，不直接运行不受控模型代码。
- OpenCompass 下载或配置失败：保留 dry-run 和错误日志，核对 0.5.3 实际配置名与数据路径，不通过升级版本掩盖兼容问题。
- 第二评分者缺席：自动评测可以完成，但人工双盲验收明确保持未完成。

## 12. 测试与验收设计

代码实现阶段至少覆盖以下自动测试：

- 实验清单恰好 9 行，三个组各 3 行，取值集合与老师要求完全一致。
- 每组 YAML 除目标变量和输出标识外无其他差异。
- 三个重复基线的学习参数完全相同，run ID 和目录互不相同。
- 配置、数据和模型哈希在所有 run 中一致。
- 日志解析器正确处理正常完成、缺失 step、NaN/Inf、中断和非零退出码。
- 显存采样、耗时和 loss 汇总都有单位、来源和 run ID。
- 评测集恰好 20 题，数学、推理、代码分别为 7、7、6 题。
- 五维权重精确和为 1.00，分值范围合法，盲评映射在解盲前不进入评分文件。
- 代码评测具备超时和隔离，失败测试不会被计为通过。
- 选模器严格按预注册排序键运行，不读取 OpenCompass 结果。
- OpenCompass 结果解析同时保存基座、最优 SFT、CEval 和 CMMLU 的原始与汇总分数。
- 最终报告中的 run 数、指标和路径能够回链到实际文件。

Week 3 最终通过条件为：至少三组有效单变量对比有真实数据；九次计划运行逐项有状态；最优选择有冻结题集、自动证据和双人盲评分数支持；OpenCompass 0.5.3 对 CEval、CMMLU 自然结束并出分；周报与最优模型路径说明完成。若某项未达到，按实际状态标记，不能用计划值替代结果。

## 13. 明确不做的内容

- 不运行 27 次全因子网格搜索。
- 不同时训练 Alpaca 与其等价 ShareGPT 副本。
- 不为九个 adapter 全部导出 15GB 级完整模型。
- 不用训练 loss 单独选最优模型。
- 不使用模型或 AI 生成的分数冒充“另一位同事或导师”的人工评分。
- 不因 OpenCompass 结果不理想临时更换已冻结的最优模型。
- 不在设计和计划阶段启动 AutoDL GPU 或把预期结果写成已完成事实。
