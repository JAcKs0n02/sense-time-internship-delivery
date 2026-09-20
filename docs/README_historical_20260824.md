> 历史快照：2026-08-24主入口。状态与相对链接不作为当前入口，当前状态见根README。

# Qwen2.5 大模型实习记录

本仓库记录 Qwen2.5/Qwen2-VL 实习任务的执行计划、代码、实验结果和交付证据。第 1–4 周已完成文本模型环境、SFT、单变量对比、DPO 与周报归档。第 5 周 Day 22–Day 27 已完成 Qwen2-VL 模型准备、图文推理、注意力可视化、幻觉测试、LoRA 实验与周报；工程交付完整，但两个 v8 纠偏候选均未通过开发集质量门槛，第 5 周总验收为 `3 PASS / 1 FAIL`。第 6 周 Day 28–Day 33 的 Agent 智能体开发要求已经完成规划，当前尚未开始实验。

## 教师提交入口

- [Unified Submission](Submission/README.md)：包含 Week 1–Week 4 与 Week 5 Day 22–Day 27，可直接用于教师检查；Week 5 明确保留质量门槛 FAIL。
- `deliverables/` 继续保留完整工程归档、测试、审计和过程证据，不作为首要
  提交入口。

## 执行计划

- [第 1 周完整执行计划](docs/week1_execution_plan.md)：包含 Day 1–5 要求、当前进度、逐步执行方法、交付目录、验证清单和口述准备。
- [第 2 周完整执行计划](docs/week2_execution_plan.md)：逐项覆盖 Day 6–10 的数据收集、双格式转换、清洗 Pipeline、SimHash 去重、SFT 配置、TensorBoard、LoRA 合并、三问对比、FAQ 和验收清单。
- [第 3 周完整执行计划](docs/week3_execution_plan.md)：逐项覆盖 Day 11–16 的 9 次单变量训练、日志采集、固定 20 题、双人盲评、OpenCompass 和最优模型归档。
- [第 4 周完整执行计划](docs/week4_execution_plan.md)：逐项覆盖 Day 17–21 的偏好规范、UltraFeedback 与自建数据、DPO 配置与 Rewards、安全十题、业务五题、周报和最终模型归档。
- [第 5 周完整执行计划](docs/week5_execution_plan.md)：逐项覆盖 Day 22–Day 27 的 Qwen2-VL、图文推理、注意力可视化、幻觉测试、冻结视觉塔 LoRA 与周报归档。
- [第 6 周完整执行计划](docs/week6_execution_plan.md)：逐项覆盖 Day 28–Day 33 的三个安全工具、ReAct Agent、多步推理、100 条工具调用 SFT、错误分析与周报归档。
- [Day 1–Day 3 阶段提交说明](deliverables/week1/day1_day3_submission.md)：汇总前三天结果与核心交付入口。
- 每天开始前以对应周的执行计划为工作入口，完成后将实际结果整理到对应的 `deliverables/weekN/dayN/`。
- 老师原始任务用于开始前和交付前的最终核对；如要求发生变化，应同步更新执行计划。

## 当前进度

| 日期 | 任务 | 结果 | 入口 |
|---|---|---|---|
| Day 1 | GPU 检查、Conda 环境、CUDA PyTorch、核心工具链 | 已完成 | [Day 1 交付说明](deliverables/week1/day1/README.md) |
| Day 2 | 模型下载、Transformers 原生推理、三类 Prompt、Chat Template | 已完成 | [Day 2 交付说明](deliverables/week1/day2/README.md) |
| Day 3 | 配置字段、参数统计、Qwen2.5 与 Llama 3 对比 | 已完成 | [Day 3 交付说明](deliverables/week1/day3/README.md) |
| Day 4 | Tokenizer 极端用例、特殊 token、分词对比 | 已完成 | [Day 4 交付说明](deliverables/week1/day4/README.md) |
| Day 5 | identity QLoRA、训练前后身份对比、TensorBoard、周报 | 已完成 | [Day 5 交付说明](deliverables/week1/day5/README.md) |

第 2 周当前进度：

| 日期 | 任务 | 结果 | 入口 |
|---|---|---|---|
| Day 6 | 三源数据收集、2k/2k/1k 抽样、Alpaca/ShareGPT 格式统一 | 已完成 | [Day 6 交付说明](deliverables/week2/day6/README.md) |
| Day 7 | 清洗 Pipeline、2048-token 截断、SimHash 去重、统计图 | 已完成 | [Day 7 交付说明](deliverables/week2/day7/README.md) |
| Day 8 | `qwen_lora.yaml` 逐行注释与首次 SFT | 已完成 | [Day 8 交付说明](deliverables/week2/day8/README.md) |
| Day 9 | 每步 loss、TensorBoard、LoRA 合并、3 个新问题测试 | 已完成 | [Day 9 交付说明](deliverables/week2/day9/README.md) |
| Day 10 | OOM/格式错误复盘、FAQ、第 2 周报告 | 已完成 | [Day 10 交付说明](deliverables/week2/day10/README.md) |

第 3 周当前进度：

| 日期 | 任务 | 结果 | 入口 |
|---|---|---|---|
| Day 11 | Rank、学习率、Epoch 单变量实验矩阵与假设 | 已完成；AutoDL smoke 退出码 0 | [Day 11 交付说明](deliverables/week3/day11/README.md) |
| Day 12–13 | 串行运行九组实验，记录 Final Loss、训练耗时与显存 | 已完成；9/9 门禁通过，108/108 日志哈希通过 | [Day 12–13 交付说明](deliverables/week3/day12_13/README.md) |
| Day 14 | 固定 20 题、自动辅助证据与五维双人盲评 | 已完成；最优 SFT 为 `epoch-e5`，基座人工均分仍更高 | [Day 14 交付说明](deliverables/week3/day14/README.md) |
| Day 15 | OpenCompass CEval、CMMLU 基座与最优 SFT 对比 | 已完成；两模型均完成 119 学科任务并生成四行分数表 | [Day 15 交付说明](deliverables/week3/day15/README.md) |
| Day 16 | 第 3 周报告与最优模型归档 | 已完成；四项验收通过，Week 4 DPO 路径已冻结 | [Day 16 交付说明](deliverables/week3/day16/README.md) |

第 4 周当前进度：

| 日期 | 任务 | 结果 | 入口 |
|---|---|---|---|
| Day 17 | DPO 原理与五类偏好数据构造方法论 | 已完成；10 组样例，验证通过 | [Day 17 交付说明](deliverables/week4/day17/README.md) |
| Day 18 | 500 条 UltraFeedback 与 200+ 条自建偏好对 | 已完成；原始 710 条，最终纠偏输入 870 条，污染为 0 | [Day 18 交付说明](deliverables/week4/day18/README.md) |
| Day 19 | DPO 配置、参考模型与训练启动 | 已完成；纠偏 40/40 steps，Chosen 上升、Rejected 下降 | [Day 19 交付说明](deliverables/week4/day19/README.md) |
| Day 20 | DPO 合并、10 题安全测试与 5 题业务对比 | 已完成；安全拒绝 10/10，业务 DPO 3.640、SFT 3.595 | [Day 20 交付说明](deliverables/week4/day20/README.md) |
| Day 21 | Rewards 曲线、第 4 周报告与模型归档 | 已完成；跨日验证 PASS，最终归档 `reward_corrective_40step_merged` | [Day 21 交付说明](deliverables/week4/day21/README.md) |

第 5 周当前进度：

| 日期 | 任务 | 结果 | 入口 |
|---|---|---|---|
| Day 22 | Qwen2-VL 模型、环境与五张图片 | 已完成（PASS） | [Day 22 交付说明](deliverables/week5/day22/README.md) |
| Day 23 | 25 条图文推理与能力边界 | 已完成（PASS） | [Day 23 交付说明](deliverables/week5/day23/README.md) |
| Day 24 | 跨模态注意力可视化 | 已完成（PASS，4 个案例） | [Day 24 交付说明](deliverables/week5/day24/README.md) |
| Day 25 | 10 题视觉幻觉检测 | 已完成（PASS，严格幻觉率 70%） | [Day 25 交付说明](deliverables/week5/day25/README.md) |
| Day 26 | 200 条图文数据与冻结视觉塔 LoRA | 实验完成；LoRA 12/20 胜、幻觉率 55%，均分 +0.0125，效果门槛 FAIL | [Day 26 交付说明](deliverables/week5/day26/README.md) |
| Day 27 | 周报与最终 VLM 归档 | 报告完成；Week 5 总验收 FAIL | [Day 27 交付](deliverables/week5/day27/README.md) |

第 6 周当前进度：

| 日期 | 任务 | 结果 | 入口 |
|---|---|---|---|
| Day 28 | LangChain、Calculator 与本地知识库检索 | 未开始 | [Day 28 实施计划](deliverables/week6/day28/README.md) |
| Day 29 | Week 4 DPO 模型与 ReAct Agent 单轮调用 | 未开始 | [Day 29 实施计划](deliverables/week6/day29/README.md) |
| Day 30 | AST-only CodeExecutor 与至少三步任务 | 未开始 | [Day 30 实施计划](deliverables/week6/day30/README.md) |
| Day 31 | 100 条 ReAct 数据与工具调用 SFT | 未开始 | [Day 31 实施计划](deliverables/week6/day31/README.md) |
| Day 32 | 错误分类、Prompt/工具描述优化与复测 | 未开始 | [Day 32 实施计划](deliverables/week6/day32/README.md) |
| Day 33 | Agent 智能体开发周报与验收归档 | 未开始 | [Day 33 实施计划](deliverables/week6/day33/README.md) |

## 实验环境

- GPU：NVIDIA GeForce RTX 3090，24GB 显存
- 模型：`Qwen/Qwen2.5-7B-Instruct`
- Python：3.10.20
- PyTorch：2.5.1+cu121
- CUDA：12.1
- Transformers：4.50.0
- 推理精度：BF16

## 核心结果

- `torch.cuda.is_available()` 返回 `True`，CUDA 矩阵计算通过。
- LLaMA-Factory、vLLM、OpenCompass、LangChain 均已安装并完成导入或命令行验证。
- Qwen2.5-7B-Instruct 模型文件下载完整，四个权重分片均存在。
- 使用 Transformers 原生接口完成代码生成、逻辑推理和角色扮演三组实验。
- `apply_chat_template` 的生成前缀、token IDs 和特殊 token 行为均已验证。
- 三组原始回答、推理参数、token 数、耗时和显存占用均已保存。
- 完成 Qwen2.5-7B-Instruct 的配置字段、GQA、RoPE、RMSNorm 和 SwiGLU 分析。
- 28 层参数统计与手工公式一致，总参数量为 `7,615,616,512`。
- RTX 3090 实际权重逐层统计与公式一致，单 token GPU 前向传播通过。
- 完成 Qwen2.5-7B-Instruct 与原始 Meta-Llama-3-8B 的架构对比。
- 完成 Qwen Tokenizer 的 15 个固定极端用例、特殊令牌和 64-token 左右截断实验。
- ByteLevel BPE 与 SentencePiece Unigram 使用同一语料，目标与实际词表均为 800；执行版 Notebook 无错误通过。
- 使用 LLaMA-Factory 0.9.3 和 91 条官方 identity 样本完成两轮 Qwen2.5-7B 4-bit QLoRA，训练退出码均为 0。
- 固定身份评测从训练前 0/8 提升到 3-epoch 的 6/8，再提升到单变量 5-epoch 的 7/8。
- TensorBoard 同一视图保留两轮 `train/loss` 曲线；《第 1 周：环境与大模型导论总结报告》已完成。
- 固定三个数据源的精确 revision，从 326250 条上游记录中确定性抽样 2000/2000/1000 条，共 5000 条。
- 同一批 5000 条样本分别转换为标准 Alpaca 和 ShareGPT 格式，使用 5000 个唯一 `sample_id` 保持一一对应。
- Day 6 数据归档清单包含来源、许可、split、上游文件大小与 SHA-256、本地子集大小与 SHA-256；独立验证为 `valid: true`。
- Day 7 使用真实 Qwen chat template 清洗 5000 条样本，修改 HTML 72 条、控制字符 27 条、截断超长样本 276 条并删除 1 条精确重复，最终保留 4999 条。
- 64-bit SimHash 在真实距离 0–6 候选上校准，并用整体及角色级 Jaccard 防止误删；20 项测试通过，完整复跑的 12 项语义产物逐字节一致。
- 最终 Alpaca 与 ShareGPT 数据各 4999 条，格式验证通过，最大模板化长度为 2048；清洗前后数量图和长度分布图已保存。
- Day 8 使用 4999 条清洗数据完成 Qwen2.5-7B-Instruct 4-bit QLoRA SFT；3 epoch 共 3747 个 optimizer steps，退出码为 0。
- 3747 条逐步 loss 均为有限值；前后 100 步均值从 1.607461 降至 1.061506，全局线性斜率为负。
- `llamafactory-cli export` 成功生成四个完整权重分片，合计 15,231,271,872 bytes；合并模型可独立加载。
- 三题盲评基座模型为 25/30、合并模型为 22/30，因此没有达到“合并模型优于基座”验收；该限制已写入周报和 FAQ。
- Day 11 冻结 Rank、学习率与 Epoch 三组单变量矩阵；所有正式配置固定同一数据集与 `seed=42`。
- Day 12–13 的九次正式 QLoRA 训练全部完成，退出码均为 0，并完整记录 Final Loss、最近 100 步平均 Loss、训练耗时、采样峰值显存和 optimizer steps。
- 九组运行的 108 个原始日志与状态文件已逐文件通过 SHA-256 校验；Git 中未保存模型权重、checkpoint 或 optimizer state。
- Day 14 冻结 20 道高难度题（数学 7、推理 7、代码 6），对 4,999 条训练数据检查后精确匹配和高相似待审查数量均为 0。
- 基座与九个 SFT adapter 的正式评测产生 200/200 条唯一终态记录，失败数为 0；自动证据明确不参与排序。
- 两位真实评分人均完成 200/200 条去身份评分；解盲后 `epoch-e5` 以 `3.78875/5` 成为最优 SFT，但低于基座模型的 `4.205/5`。
- OpenCompass 0.5.3 正式评测中，基座/最优 SFT 的 CEval 分别为 77.542262/79.189426，CMMLU 分别为 0.647412/5.205187；OpenCompass 不参与选模。
- `epoch-e5` 已合并并通过独立离线加载；Week 4 DPO 输入固定为 `/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged`。
- Day 18 从固定 revision UltraFeedback 选择 500 条，并构造 210 条五类均衡自建偏好对；最终 710 条按 `seed=42` 分层为 train 639、validation 71。
- 710 条数据 ID 唯一、精确 Prompt 重复为 0；冻结 15 题的精确和高相似污染均为 0，322 对模板相似候选已审计为不同主题。
- 真实 Week 3 Qwen tokenizer 下 chosen/rejected 完整序列最大为 1738/1670，超过 `cutoff_len=2048` 的记录为 0。
- LLaMA-Factory 0.9.3 成功加载 ranking 数据并生成 chosen/rejected 双侧 input、mask 与 labels；AutoDL 实例在验收后已确认关机。
- Day 19 最终纠偏 DPO QLoRA 完成 40/40 steps；Chosen/Rejected/Margin 的首末窗口变化为 `+0.031859/-0.004136/+0.035995`，严格满足老师要求；显式参考模型 OOM 后使用 adapter-disabled 隐式参考实现。
- Day 20 将最终纠偏 DPO adapter 合并为 4 个 BF16 权重分片，完整清单 SHA-256 为 `aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c`，独立加载和无害 smoke 均通过。
- 冻结 15 题上两模型各完成 15/15 回答；老师要求的安全拒绝率均为 10/10（100%），DPO 通过 90% 门槛；更严格的增强安全响应率为 SFT-only 30%、SFT+DPO 40%，两模型均无 score 0 的可执行伤害回答。
- 五题业务盲评中 SFT-only 为 3.595/5、SFT+DPO 为 3.640/5，DPO 2 胜、SFT 1 胜、2 平；只支持该固定小样本上的描述性小幅优势。

## 仓库结构

```text
Submission/                # 教师提交入口：Week 1–Week 5 精简正式交付
├── Week1/                 # Day 1–5
├── Week2/                 # Day 6–10
├── Week3/                 # Day 11–16
├── Week4/                 # Day 17–21
└── Week5/                 # Day 22–27；保留真实质量 FAIL

deliverables/week1/
├── day1/                 # Day 1 结果、说明和环境截图
├── day2/                 # Day 2 代码、结果、报告和截图
├── day3/                 # Day 3 配置、参数统计、架构报告和验证结果
├── day4/                 # Day 4 Notebook、分词结果、验证日志和截图
├── day5/                 # Day 5 QLoRA、固定评测、TensorBoard 和周报
└── notes/                # Day 1/Day 2 原理与口述准备

deliverables/week2/
├── day6/                 # 三源原始子集、双格式数据、血缘、清单、脚本和验证
├── day7/                 # 清洗脚本、双格式训练候选、逐条审计、统计和图表
├── day8/                 # 逐行注释配置、数据注册、训练脚本和首次 SFT 摘要
├── day9/                 # 每步 loss、TensorBoard、模型合并、三题评测和验证
└── day10/                # 第 2 周报告、FAQ、验收矩阵、文件清单和最终验证

deliverables/week3/
├── day11/                # 9-run 实验计划、基线、生成器、验证器和 smoke 配置
├── day12_13/             # 九次正式训练、指标汇总、原始日志和完整性清单
├── day14/                # 20 题、批量评测、自动辅助证据与双盲评分材料
├── day15/                # OpenCompass 配置、四行分数表与远端证据指纹
└── day16/                # 第 3 周报告、验收矩阵与最优模型路径

deliverables/week4/
├── day17/                # 偏好构造指南、五类 taxonomy、10 组样例和验证器
├── day18/                # 710 条偏好数据、去重、split、真实 tokenizer 与 schema smoke
├── day19/                # DPO 配置、240-step 训练、Rewards 和 adapter 清单
├── day20/                # DPO 合并、30 条回答、安全评分和业务盲评
├── day21/                # Rewards 图、周报、验收矩阵和最终模型归档
└── README.md             # 第 4 周工程入口；Day 17–Day 21 已完成

deliverables/week5/
├── day22/                # VLM 模型与图片素材
├── day23/                # 25 条图文推理
├── day24/                # 跨模态注意力可视化
├── day25/                # 十题幻觉检测
├── day26/                # 冻结视觉塔 LoRA 与盲评
└── day27/                # 周报、验收矩阵和模型归档

deliverables/week6/
├── day28/                # Calculator 与本地知识库工具计划
├── day29/                # ReAct Agent 单轮调用计划
├── day30/                # AST 检查与多步推理计划
├── day31/                # 100 条工具调用 SFT 计划
├── day32/                # 错误分析与优化计划
└── day33/                # 第 6 周报告计划

docs/                     # 第 1 周至第 6 周完整执行计划
archive/week1/            # 补充截图和原始实验归档
```

教师检查优先从 `Submission/` 进入；`deliverables/` 保留完整工程归档，`archive/` 保存补充过程材料和原始校验记录。

第 2 周 Day 6–Day 10 均已建立正式交付目录；完整模型权重保留在 AutoDL，Git 中只保存模型清单和哈希。

## 学习笔记

环境选型、CUDA/PyTorch 关系、四项工具链职责、原生推理数据流和 Chat Template 原理见：

- [Day 1 与 Day 2 知识讲解](deliverables/week1/notes/day1_day2_knowledge.md)

Qwen2.5 单层结构、参数公式、GQA、RoPE 和 Llama 3 对比见：

- [Day 3 架构分析报告](deliverables/week1/day3/REPORT.md)

Qwen Tokenizer、特殊令牌、Chat Template、截断和分词算法对比见：

- [Day 4 Tokenizer 实验说明](deliverables/week1/day4/README.md)
- [Day 4 执行版 Notebook](deliverables/week1/day4/tokenizer_experiments.executed.ipynb)

LLaMA-Factory、4-bit QLoRA、identity 固定评测和整周总结见：

- [Day 5 验收说明](deliverables/week1/day5/README.md)
- [第 1 周总结报告](deliverables/week1/day5/REPORT.md)

第 2 周数据来源固定、确定性抽样、Alpaca/ShareGPT 转换和血缘设计见：

- [Day 6 数据收集与格式统一](deliverables/week2/day6/README.md)
- [Day 6 原始数据归档清单](deliverables/week2/day6/source/manifests/raw_data_manifest.md)
- [Day 7 清洗 Pipeline 与统计结果](deliverables/week2/day7/README.md)
- [Day 8 配置详解与首次 SFT](deliverables/week2/day8/README.md)
- [Day 9 训练监控、LoRA 合并与三问评测](deliverables/week2/day9/README.md)
- [第2周：数据与SFT入门报告](deliverables/week2/day10/REPORT.md)
- [Week 2 FAQ 与问题复盘](deliverables/week2/day10/FAQ.md)

第 3 周超参数实验矩阵、固定控制、预注册假设和解释边界见：

- [第 3 周完整执行计划](docs/week3_execution_plan.md)
- [Week 3 工程归档索引](deliverables/week3/README.md)
- [Day 11 对比实验设计](deliverables/week3/day11/README.md)
- [Day 12–13 九次训练结果](deliverables/week3/day12_13/README.md)
- [Day 12–13 指标汇总 CSV](deliverables/week3/day12_13/source/results/experiment_summary.csv)
- [Day 14 固定评测与双盲评分结果](deliverables/week3/day14/README.md)
- [Day 15 OpenCompass 通用评测](deliverables/week3/day15/README.md)
- [第 3 周 SFT 优化与评估报告](deliverables/week3/day16/REPORT.md)
- [Day 16 最优模型归档说明](deliverables/week3/day16/README.md)
- [Week 3 教师提交入口](Submission/Week3/README.md)

第 4 周 DPO 偏好对齐的已批准执行口径见：

- [第 4 周完整执行计划](docs/week4_execution_plan.md)
- [Week 4 工程归档入口](deliverables/week4/README.md)

第 5 周多模态推理、注意力、幻觉与 VLM LoRA 的完整结果见：

- [第 5 周完整执行计划](docs/week5_execution_plan.md)
- [Week 5 工程归档入口](deliverables/week5/README.md)
- [第 5 周多模态实践报告](deliverables/week5/day27/REPORT.md)

第 6 周 Agent 智能体的已冻结执行口径见：

- [第 6 周完整执行计划](docs/week6_execution_plan.md)
- [Week 6 工程归档入口](deliverables/week6/README.md)
- [Week 6 Agent 交付设计](docs/superpowers/specs/2026-08-24-week6-agent-deliverables-design.md)
