# 第2周：数据与SFT入门报告

## 1. 本周目标与结论

本周完成了从多源中文指令数据到可训练数据、从 QLoRA SFT 到模型合并与受控
评测的完整闭环。三个固定版本的数据源共抽取 5,000 条语义样本，统一为
Alpaca 和 ShareGPT 两种等价表示；清洗后保留 4,999 条；Qwen2.5-7B-Instruct
在 AutoDL RTX 3090 上完成 3 epoch、3,747 个 optimizer step 的 4-bit QLoRA
SFT；训练 loss 整体下降；LoRA 成功合并为可独立加载的完整模型。

五项最终验收中，清洗脚本、训练数据、loss 趋势和周报四项通过。三题盲评中
基座模型为 25/30，合并模型为 22/30，因此“合并后模型对话质量优于基座”
未通过。本报告保留该负面结果，不用训练 loss 代替真实质量评测。

完整验收入口见
[Week 2 验收矩阵](source/results/week2_acceptance_matrix.md)。

## 2. 实验环境与范围

| 项目 | 实际环境 |
|---|---|
| 远端平台 | AutoDL |
| GPU | NVIDIA GeForce RTX 3090，24 GB |
| 基座模型 | Qwen2.5-7B-Instruct |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| CUDA Runtime | 12.1 |
| Transformers | 4.50.0 |
| LLaMA-Factory | 0.9.3 |
| 训练方法 | SFT + 4-bit QLoRA |
| 训练精度 | 4-bit NF4 基座、BF16 计算 |
| 随机种子 | 42 |
| 最大样本长度 | 2,048 Qwen chat-template tokens |

`SFT`（Supervised Fine-Tuning，监督微调）用“指令—回答”样本训练模型在给定
指令条件下生成目标回答。`QLoRA` 则把冻结的基座权重以 4-bit 加载，只训练
插入线性层的低秩 LoRA 矩阵，从而把 7B 模型的训练显存控制在单张 24 GB
显卡可承受范围内。

## 3. Day 6：数据收集与格式统一

### 3.1 数据源与版本

| 数据源 | 固定 revision | 上游规模 | 本次抽样 | 许可/条件 |
|---|---|---:|---:|---|
| `llamafactory/alpaca_gpt4_zh` | `065394ee242c43928298de8f43a8748ffd16f3e3` | 42,677 | 2,000 | Apache-2.0 |
| `BAAI/COIG-PC` / `Top200PerTask` | `4ca2778d69d7a9c17a0a6c1668c6e989b102176b` | 253,558 | 2,000 | gated；子数据集许可优先 |
| `FreedomIntelligence/sharegpt-chinese` | `ef0a611d7c8c3cd7b17c152d4166c67a92015c2b` | 30,015 | 1,000 | Apache-2.0 |

数据 `provenance`（血缘）记录每条样本来自哪个仓库、revision 和原始索引。
这样后续即使经过转换、清洗和去重，也能回溯原始记录。本次使用固定
`seed=42`，按 `SHA256("42:<source>:<source_index>")` 的优先级确定性抽样。

完整来源、文件大小和哈希见
[原始数据归档清单](../day6/source/manifests/raw_data_manifest.md)；最终计数验证见
[Day 6 验证结果](../day6/source/results/day6_validation.json)。

### 3.2 Alpaca 与 ShareGPT 的区别

Alpaca 结构适合表示单个监督任务：

```json
{"instruction": "任务", "input": "补充上下文", "output": "目标回答"}
```

ShareGPT 结构保存带角色的多轮消息：

```json
{
  "conversations": [
    {"from": "human", "value": "问题"},
    {"from": "gpt", "value": "回答"}
  ]
}
```

本次两份 5,000 条文件是同一批语义样本的等价表示，不能相加后宣称有
10,000 条训练数据。ShareGPT 转为 Alpaca 时，最后一组完整问答作为
instruction/output，更早完整轮次序列化到 input；反向转换时 instruction 与
非空 input 共同构成 human 消息，output 构成 gpt 消息。

实际抽样中有 10 条 ShareGPT 记录以未回答的 human 消息结尾。原始记录保持
不变，训练可见格式仅删除无法形成监督目标的末尾消息，并在 provenance 中
记录调整。详见 [FAQ 2](FAQ.md#faq-2sharegpt-原始对话为什么不能全部原样进入-sft)。

## 4. Day 7：清洗 Pipeline

### 4.1 为什么先清洗、统一和去重

- 结构统一让训练框架知道哪里是提示、上下文和监督回答。
- HTML、脚本、不可见字符会把网页噪声或异常 token 带入训练分布。
- 空回答没有监督目标，无法产生有意义的 SFT loss。
- 超长样本增加显存与计算，并可能被框架以不透明方式截断。
- 重复数据会重复加权某些模式，降低数据有效多样性并增加训练成本。

正式 Pipeline 顺序为：schema 校验 → 初次空值过滤 → HTML/entity 清理 →
控制字符清理 → 空白规范化 → 二次空值过滤 → Qwen Tokenizer 长度统计与
结构化截断 → 精确去重 → SimHash 候选召回 → Jaccard 确认 → 双格式输出。

独立入口为 [clean_pipeline.py](../day7/clean_pipeline.py)。

### 4.2 为什么用训练模型的 Tokenizer 截断

字符数与 token 数并不等价。中英文、代码、特殊符号和角色标记的分词长度差异
很大。Pipeline 使用 Qwen2.5 Tokenizer，并实际调用 chat template 后计算长度，
与 Day 8 的 `template: qwen` 和 `cutoff_len: 2048` 保持一致。

多轮样本先删除最旧的完整 user-assistant 对，始终保留最后一个监督目标；若
最后一轮仍超长，才在消息 token 边界内继续缩短。最终所有样本都不超过 2,048
tokens。

### 4.3 SimHash 与 Jaccard

SimHash 把文本特征映射成定长指纹。本实现提取字符 3-gram，为每个特征生成
64-bit 哈希，再按特征权重对每一位投票。两个指纹不同位的数量称为 Hamming
distance；距离越小，只表示越可能相似，不保证语义重复。

因此正式规则先用 4 个 16-bit band 召回候选，再要求：

- Hamming distance ≤ 3；
- 整体字符 3-gram Jaccard ≥ 0.85；
- user 和 assistant 两侧 Jaccard 的较小值 ≥ 0.90。

`Jaccard(A,B)=|A∩B|/|A∪B|`，用于衡量两个特征集合的重合比例。真实数据召回
77 对候选，但没有一对同时通过确认门槛，因此模糊删除为 0；唯一删除的是一条
精确重复。这里没有为了制造“去重数量”而降低阈值。

### 4.4 清洗统计

| 项目 | 数量 |
|---|---:|
| 原始输入 | 5,000 |
| HTML/entity 修改 | 72 |
| 控制字符修改 | 27 |
| 空白规范化修改 | 1,127 |
| 超过 2,048 tokens | 276 |
| 完成结构化截断 | 276 |
| 精确重复删除 | 1 |
| 模糊重复删除 | 0 |
| 最终保留 | 4,999 |

最终 token 长度由清洗前 mean 539.761、max 29,343，变为 mean 365.411、
max 2,048。数量、长度与校验结果见
[清洗统计](../day7/source/results/cleaning_stats.json)和
[Day 7 验证](../day7/source/results/day7_validation.json)。

## 5. Day 8：逐行理解配置与首次 SFT

### 5.1 官方配置基线

固定的 LLaMA-Factory 0.9.3 中没有字面名为 `qwen_lora.yaml` 的纯文本示例。
本实验没有下载其他版本的同名文件冒充，而是原样归档同版本官方文本 LoRA SFT
示例，再在独立文件中改成 Qwen2.5 正式配置。官方副本与来源的 SHA-256 均为
`ecd8efc4cdf29d52d8a0be25028443cc813f6d82db957a90f124d7dee5095fba`。

[正式 YAML](../day8/configs/qwen25_7b_week2_lora_annotated.yaml)共有 45 个活动
参数；每个参数均解释“作用、当前值、调整影响”。最终验证为 `valid: true`。

### 5.2 六个重点参数

| 参数 | 本次值 | 含义 |
|---|---|---|
| `model_name_or_path` | 本地 Qwen2.5-7B-Instruct | 决定模型结构、Tokenizer 和冻结基座权重；合并时必须使用同一基座 |
| `template` | `qwen` | 把 Alpaca 字段渲染成 Qwen 角色 token，并决定哪些 assistant token 参与 loss |
| `finetuning_type` | `lora` | 冻结基座，只训练低秩增量矩阵 |
| `lora_rank` | 8 | 低秩矩阵的秩，影响 adapter 容量、参数量和显存 |
| `lora_alpha` | 16 | 缩放 LoRA 分支；标准比例为 alpha/rank=2，不等于学习率 |
| `learning_rate` | `1e-4` | 控制优化器每次更新 LoRA 参数的步幅 |

LoRA 用两个较小矩阵近似权重增量 `ΔW = B×A`。rank 越高，可表达的增量空间
通常越大，但可训练参数和显存也随之增加；alpha 调整增量强度，不改变矩阵数量。

Chat Template 不是展示层装饰。它决定 system/user/assistant 的特殊 token、
消息边界和 loss mask，因此属于训练数据分布的一部分。模板错误即使不报错，
也可能让模型学习错误的角色结构。

### 5.3 正式训练配置与结果

| 项目 | 实际值 |
|---|---:|
| 清洗后训练记录 | 4,999 |
| Epoch | 3 |
| Micro batch | 1 |
| Gradient accumulation | 4 |
| 有效 batch | 4 |
| Optimizer steps | 3,747 |
| Cutoff length | 2,048 |
| LoRA rank / alpha | 8 / 16 |
| Learning rate | 1e-4 |
| Scheduler / warmup | cosine / 0.1 |
| Gradient checkpointing | true |
| Loss logging | 每个 optimizer step |

`batch size` 是一次前后向实际处理的样本数；`gradient accumulation` 让多个
micro-batch 的梯度累积后再执行一次 optimizer step。两者共同决定有效 batch，
但累积不能降低单条超长样本本身的峰值显存。`epoch` 是完整遍历训练集的次数；
`learning rate` 过大可能振荡或破坏原能力，过小则学习不足。loss 是当前监督
目标上的误差，不是通用问答质量分数。

正式训练自然结束，退出码为 0，用时 3:12:12。最终 adapter 大小
80,792,096 bytes，SHA-256 为
`39eef2d0647d08cc1119ff2f80aa90224229cc6078d52982ad28a7b09b3d94fa`。
详见 [SFT 摘要](../day8/results/first_sft_summary.json)。

## 6. Day 9：训练监控、权重合并与测试

### 6.1 Loss 与 TensorBoard

TensorBoard 读取训练 event 文件并展示标量曲线。本次把 3,747/3,747 个连续
optimizer step 全部导出到 CSV，同时保留 TensorBoard 原始界面截图。

| 指标 | 数值 |
|---|---:|
| 第一步 loss | 1.4029 |
| 最后一步 loss | 0.9737 |
| 前 100 步均值 | 1.607461 |
| 后 100 步均值 | 1.061506 |
| 线性斜率 | -0.00013193245 |
| 非有限 loss | 0 |

逐步 loss 有正常波动，不能只比较任意两个点；前后移动平均和全局线性斜率共同
支持“整体下降”结论。证据见
[逐步 loss CSV](../day9/source/results/train_loss_by_step.csv)、
[loss 汇总](../day9/source/results/loss_summary.json)和
[TensorBoard 截图](../day9/evidence/tensorboard_train_loss.png)。

### 6.2 LoRA adapter 与合并模型

adapter 只保存低秩增量，加载时仍需要原始基座。`llamafactory-cli export`
把 adapter 增量应用到同一基座并导出完整模型，因此合并目录可以独立加载。

正式导出退出码为 0。合并目录位于：

```text
/root/autodl-tmp/qwen25-week2/merged/qwen25-7b-week2-sft-merged
```

目录共有 14 个文件，其中四个 `safetensors` 权重分片合计
15,231,271,872 bytes；配置、Tokenizer、chat template 和权重索引均存在，
没有 `adapter_model` 权重。验证结果见
[合并模型验证](../day9/source/results/merged_model_validation.json)。

### 6.3 三个新问题的公平比较

三个问题在推理前冻结，基座与合并模型使用同一 Qwen template、BF16、
`do_sample: false`、seed 42 和最多 512 个新 token。与 4,999 条训练 prompt 的
最大 Jaccard 分别为 0.146789、0.116959 和 0.104478，低于 0.80 排除阈值。

盲评按指令完成、正确性、格式约束、相关完整性和生成质量五个维度评分，每题
满分 10 分。解盲结果为：

| 模型 | Q1 | Q2 | Q3 | 总分 |
|---|---:|---:|---:|---:|
| 基座模型 | 8 | 9 | 8 | 25/30 |
| 合并模型 | 7 | 5 | 10 | 22/30 |

合并模型在 Q3 更简洁且明确写出验收标准，但在 Q2 违反“每部分最多 3 点”的
限制；两个模型都超过 Q1 的 150 字限制。预设规则要求合并模型总分更高且没有
生成质量退化，因此本次结果为 `merged_model_better: false`。

完整回答和评分见
[基座/合并模型对比](../day9/source/results/base_vs_merged_comparison.md)。

## 7. 问题复盘

本周真实问题包括：COIG-PC gated 访问条件；10 条 ShareGPT 末尾没有 assistant
回答；老师示例名与固定 LLaMA-Factory 版本不一致；合并模型在本次三题盲评中
没有超过基座。

正式 SFT 没有发生 CUDA OOM，也没有发生 LLaMA-Factory 数据解析异常。报告不
虚构报错，而是在 FAQ 中分别提供了预防性诊断步骤。完整的现象、根因、修复、
验证和预防见 [Week 2 FAQ](FAQ.md)。

## 8. 最终验收矩阵

| 编号 | 老师验收标准 | 结论 | 证据摘要 |
|---:|---|---|---|
| 1 | 清洗脚本可独立运行 | 通过 | 独立 CLI；20 项 Day 7 测试通过；完整复跑的 12 项语义产物逐字节一致 |
| 2 | 训练集 ≥1,500 条且格式正确 | 通过 | 清洗后 4,999 条；Alpaca 与 ShareGPT 对齐；最大长度 2,048；验证无错误 |
| 3 | SFT loss 稳定下降 | 通过 | 3,747 条连续有限 loss；前/后 100 步均值 1.607461/1.061506；斜率为负 |
| 4 | 合并后模型对话质量优于基座 | **未通过** | 三题盲评基座 25/30、合并 22/30；合并与加载本身均成功 |
| 5 | 周报提交 | 通过 | 本报告与 FAQ 均已形成可追溯交付 |

## 9. 实验限制与下一步

1. 三题评测样本量小，只能否定“本次证据证明整体提升”，不能证明模型所有任务
   都退化。
2. 4,999 条训练数据覆盖领域较广，没有围绕单一能力设计；训练目标与三题约束
   遵循评测之间可能存在偏差。
3. 全部清洗数据用于首次训练，没有在训练前单独划出大规模验证集。
4. 训练 loss 下降只说明模型更好地拟合训练目标，不能自动推出事实正确性、
   格式遵循或泛化能力提高。
5. 下一轮应先冻结更大的 held-out 验证集，再增加高质量的格式约束、简洁回答和
   数据工程任务样本，进行只改变一个变量的对照训练。

`held-out` 指训练过程中完全不参与参数更新的数据，用于估计模型面对新样本时的
表现。扩大评测后还应报告逐类能力、平均分、方差和失败案例，而不是只给总分。

## 10. 复现与文件入口

- [Day 6 数据收集与格式统一](../day6/README.md)
- [Day 7 清洗 Pipeline](../day7/README.md)
- [Day 8 配置与首次 SFT](../day8/README.md)
- [Day 9 TensorBoard、合并与评测](../day9/README.md)
- [Week 2 FAQ](FAQ.md)
- [Week 2 文件清单](source/results/week2_file_manifest.csv)
- [Week 2 SHA-256](source/results/week2_files.sha256)
- [最终验证记录](source/results/week2_final_validation.txt)

大模型完整合并权重没有提交到 Git；仓库保存模型清单和哈希，完整目录继续位于
AutoDL 数据盘。这样既能验证文件完整性，也避免把约 15.25 GB 模型文件写入
普通源码仓库。

## 11. 总结

本周完成了可追溯的数据收集、双格式转换、清洗、真实 Tokenizer 截断、保守
模糊去重、QLoRA SFT、逐步 loss 监控、LoRA 合并和公平的基座对比。工程链路
已经跑通，训练指标符合预期；模型质量验收则给出了一个有价值的负面结果：
“训练成功”不等于“对话质量必然提升”。下一轮工作的重点应从继续堆叠训练步数，
转向更明确的数据目标和更可靠的 held-out 评测。
