# Qwen2-VL 实习第 5 周执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 Qwen2-VL-7B-Instruct 完成图文推理、跨模态注意力可视化、视觉幻觉测试和冻结视觉塔的 LoRA 微调，并用隔离评测证明效果后形成第 5 周报告与最终 VLM 模型归档。

**Architecture:** Day 22 固定模型、环境和五张测试图片；Day 23–Day 25 在同一冻结输入上建立推理、注意力和幻觉基线；Day 26 使用 200 条训练数据、20 条开发记录和 20 条最终记录完成 LoRA；Day 27 只汇总已有证据。工程归档保留复现证据，教师提交只投影必要成果，不包含模型大权重、凭据或平台运维信息。

**Tech Stack:** Python 3.10、PyTorch、Transformers、Qwen2-VL、`qwen-vl-utils`、LLaMA-Factory、PEFT、Pillow、NumPy、pandas、matplotlib、Git、NVIDIA RTX 3090 24GB。

## Global Constraints

- 老师新版 [`实习需求.pdf`](../实习需求.pdf) 中的 Week 5 原始要求具有最高优先级。
- Day 22–Day 25 已完成并通过证据门禁；Day 26 与 Day 27 的数据、训练、冻结审计、双候选盲评和周报已完成。v8-E/F 均未通过开发集门槛，因此 Week 5 效果验收 FAIL，没有启动最终测试。Day 25 严格幻觉率为 7/10（70%）。
- 既有 RTX 3090 为 24GB，按老师的显存规则正式模型固定为 `Qwen/Qwen2-VL-7B-Instruct`；下载时必须固定不可变 revision。
- Day 22 五张图片固定覆盖表格截图、自然风景、Logo、手写公式和 UI 界面，且必须自建、自拍或许可明确。
- Day 23 固定运行 5 张图片乘 5 类问题，共 25 条唯一记录；失败项和困难项不得删除。
- Day 24 至少产出 3 张可追溯热力图；Qwen2-VL 的跨模态口径是目标文本 token 对视觉 token 区间的语言模型 self-attention，不伪造独立 `cross_attn` 模块。
- Day 25 固定 10 题，`hallucination_rate = hallucination_count / 10`；不得看完回答后删题、修改 ground truth 或改变分母。
- Day 26 正式训练集为 200 条，另建 20 条开发记录与 20 条最终评测；训练、开发、最终不得共享图片 SHA-256。
- Day 26 使用 LLaMA-Factory LoRA 并设置 `freeze_vision_tower: true`；先通过 processor/schema 和单步 GPU smoke，再正式训练。
- “明显提升”必须同时满足：平均加权分提升至少 `0.50/5`、LoRA 至少胜出 `12/20`、LoRA 视觉幻觉率不高于 Base。
- Git 只保存代码、配置、报告、小型数据和审计元数据；模型权重、checkpoint、缓存、凭据和大文件留在 AutoDL 持久化目录。
- `Submission/Week5/` 只在对应实验实际完成且证据可审计后增量建立；当前包含 Day 22–Day 26 教师交付，Day 26 的效果 FAIL 被原样保留。
- 每次使用 AutoDL 后必须确认关机，但教师提交不得包含平台关机证据。
- 不得提交账号、密码、SSH 连接信息、私有盲评映射、真实客户数据或个人敏感信息。

---

## 1. 文档用途与执行依据

这份文档是 Week 5 的长期执行入口，用于统一每天的输入、术语、实验步骤、结果格式和完成条件。逐日实际状态及证据入口记录在 [`deliverables/week5/`](../deliverables/week5/README.md)。

执行优先级如下：

1. 老师新版 [`实习需求.pdf`](../实习需求.pdf) 的 Week 5 原始要求；
2. 本计划冻结的数量、数据隔离、评分和证据规则；
3. 锁定 revision 后的 Qwen2-VL、Transformers 与 LLaMA-Factory 真实接口；
4. 每日工程归档中的原始输入、日志、模型清单和验证输出；
5. Day 27 从已验证工程证据生成的教师提交。

计划中的命令、路径和阈值只描述未来操作。只有实际文件、日志和验证输出齐全时，才能把相应日期改为“已完成”。

## 2. 老师原始要求与最终验收

| 日期 | 老师原始任务 | 老师交付 |
|---|---|---|
| Day 22 | 按显存选择 Qwen2-VL；下载模型；安装 `qwen-vl-utils`；准备 5 张指定类型图片 | 模型下载确认、图片素材包 |
| Day 23 | 每张图运行描述、OCR、图表解释、美学评价和隐含信息推理 | 25 条图文推理结果记录 |
| Day 24 | 使用 Hooks 提取 VLM 中间层注意力，解释生成特定词时关注的图像区域 | 至少 3 张注意力热力图 |
| Day 25 | 构造 10 组图片-问题-假答案对，测量模型对不存在物体的幻觉 | 幻觉检测报告 |
| Day 26 | 构造 200 条图文指令数据；用 LLaMA-Factory 做 VLM LoRA；冻结 ViT，仅训练 LLM 低秩适配器 | 微调后的 VLM、微调日志 |
| Day 27 | 补全实验，撰写第 5 周多模态报告 | 周报 |

Week 5 的四项最终验收解释为：

1. **完成图文推理**：Day 23 固定 25 条记录全部存在且可以追溯到模型 revision、图片 SHA 和生成配置；
2. **完成注意力可视化**：至少 3 张热力图来自真实中间层权重，并保留层号、目标 token、视觉 token 区间和数组；
3. **微调后有明显提升**：20 条隔离评测同时通过本计划预注册的三项量化门槛；
4. **提交周报**：报告覆盖模型、素材、推理、热力图、幻觉、训练、前后对比、限制和最终模型归档。

## 3. 当前已知环境与未验证项

| 项目 | 当前口径 |
|---|---|
| GPU | Day 22 已实际验证 NVIDIA GeForce RTX 3090 24GB |
| 正式模型 | `Qwen/Qwen2-VL-7B-Instruct` |
| 模型 revision | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` |
| Python / CUDA / PyTorch | Python 3.10.20、PyTorch 2.5.1+cu121、torch CUDA 12.1 |
| Transformers / `qwen-vl-utils` | Transformers 4.50.0、`qwen-vl-utils` 0.0.14 |
| LLaMA-Factory | Day 26 从锁定代码 revision 读取版本并验证真实多模态 schema |
| 远端工作根目录 | `/root/autodl-tmp/qwen2-vl-week5/`，实际路径已写入非敏感 manifest |
| Git 大文件边界 | 不提交 7B 权重、checkpoint、optimizer state 或模型缓存 |

## 4. 当前进度

| 日期 | 状态 | 开始条件 | 完成证据 |
|---|---|---|---|
| Day 22 | 已完成（PASS） | 已执行 | 固定 revision 的模型清单、离线 smoke、五张图片和素材 manifest |
| Day 23 | 已完成（PASS） | 已执行 | 25/25 原始回答、配置、性能、逐条 Codex 辅助视觉审查和 7 项验证 |
| Day 24 | 已完成（PASS） | 已执行 | 4/4 真实跨模态注意力案例、原始数组、4 张三联图和 9/9 PASS 验证 |
| Day 25 | 已完成（PASS） | 十题和 ground truth 已在推理前冻结 | 10/10 原始回答、严格二元评分、7/10 幻觉与 70% 幻觉率报告 |
| Day 26 | 实验完成；效果门槛 FAIL | 200 条训练、20 条开发、20 条最终评测和真实 schema 全部通过 | 训练/冻结/adapter PASS；LoRA 12/20 胜、幻觉率 55%，均分增量 +0.0125 未达到 +0.50 |
| Day 27 | 报告完成；总验收 FAIL | 双候选终止、全周证据审计完成 | 周报、验收矩阵和 VLM 模型归档 |

## 5. 核心术语

| 术语 | 本项目中的含义 |
|---|---|
| VLM | Vision-Language Model，视觉语言模型；同时接收图像和文本并生成文本回答 |
| 视觉编码器 / ViT | 将图像切分并编码为视觉特征的 Vision Transformer；Day 26 保持冻结 |
| 视觉 token | 图像经处理器和视觉编码器后进入模型序列的离散特征位置，不等同于原图像素 |
| 动态分辨率 | Qwen2-VL 根据图片尺寸生成不同数量的视觉 token；因此必须固定图片和像素上下限 |
| processor | 把对话、图片和视频转换为模型张量的预处理组件，同时提供图像网格信息 |
| Hook | PyTorch 在指定模块前向传播时捕获输入或输出的回调，用于读取中间层 Q/K 或注意力 |
| self-attention | 同一序列内 token 彼此计算关注权重的机制；Qwen2-VL 语言模型用它连接文本与视觉 token |
| 跨模态注意力 | 本项目定义为目标文本 token 指向视觉 token 区间的语言模型 self-attention 权重 |
| 热力图 | 把视觉 token 权重恢复成二维网格并叠加在原图上的可视化，不代表严格因果解释 |
| 视觉幻觉 | 模型确认、扩展或依据图片中不存在的物体、文字、数值、关系或动作进行回答 |
| ground truth | 在运行模型前根据图片人工确认并冻结的真实事实，用于判断回答是否幻觉 |
| LoRA | Low-Rank Adaptation；向目标线性层加入少量低秩可训练参数，降低微调显存与存储成本 |
| adapter | LoRA 训练后保存的增量参数，需要与指定基座模型共同加载或合并 |
| 冻结视觉塔 | 视觉编码器参数不参与梯度更新；只训练语言模型中的 LoRA 参数 |
| 图像隔离评测 | 图片与目标答案从未进入训练的固定 20 条最终记录；本轮沿用相同通用任务 Prompt 模板，因此只衡量新图片泛化 |
| 数据污染 | 训练数据与评测题、图片或近似答案重叠，导致评测不能代表泛化能力 |
| model revision | 模型仓库的不可变 commit 标识，用于确保后来下载的是同一份模型 |
| SHA-256 | 文件内容哈希；用于确认图片、数据、配置和模型清单未被无声替换 |
| lineage | 从基座 revision、数据哈希、配置、训练 run 到最终 adapter 的完整血缘 |

## 6. Week 5 数据流与目标文件结构

```text
Day 22 模型 revision + 环境 + 5 张图片
        ├──> Day 23：25 条 Base 图文推理
        ├──> Day 24：3+ 跨模态注意力案例
        └──> Day 25：10 题 Base 幻觉测试

200 条训练数据 + 20 条开发记录 + 20 条图像隔离最终评测
        └──> Day 26：冻结视觉塔 LoRA
                   └──> Base/LoRA 20 条盲评与幻觉对比

Day 22–Day 26 已验证证据
        └──> Day 27：周报 + 验收矩阵 + 最终模型归档 + 教师投影
```

规划阶段的仓库结构：

```text
docs/week5_execution_plan.md
deliverables/week5/
├── README.md
├── day22/README.md
├── day23/README.md
├── day24/README.md
├── day25/README.md
├── day26/README.md
└── day27/README.md
```

实验开始后按日增加的工程证据预计放在：

```text
deliverables/week5/dayNN/
├── README.md
├── configs/          # 生效配置，不保存凭据
└── source/
    ├── data/         # 可提交的小型数据、清单或样例
    ├── results/      # CSV/JSON、图、报告和验证输出
    ├── scripts/      # 生成、验证、推理、绘图脚本
    └── tests/        # schema、数量、哈希和门禁测试
```

Day 22 的素材合同：

```text
image_manifest.csv:
image_id,relative_path,image_type,source,license,width,height,file_bytes,sha256
```

Day 23 的推理合同：

```text
inference_results.csv:
record_id,image_id,question_type,prompt,raw_response,input_tokens,
output_tokens,latency_seconds,peak_gpu_memory_mib,strength_label,
hallucination_label,reviewer_reason,model_revision,image_sha256,
generation_config_sha256
```

## 7. 每日通用执行规则

### 7.1 开始前

- 读取当天老师要求、本计划和对应 Day README；
- 检查 `nvidia-smi`、磁盘、Python 和核心包版本；
- 对上游模型、图片、数据、配置和评测清单重新计算或核对 SHA-256；
- 确认输入数量、ID 唯一性和训练/评测隔离；
- 先运行静态检查和小样本 smoke，不能直接启动长时间实验。

### 7.2 执行中

- 固定 `seed=42`；正式生成默认 `do_sample=false`；
- 保存原始 Prompt、原始回答、有效配置、命令、标准输出/错误和 return code；
- 不覆盖失败 run，不挑选更漂亮的回答替换第一次成功结果；
- GPU OOM、非有限值或 schema 错误时停止正式运行，先保存错误再排查；
- 未得到证据的字段写入执行日志的缺失清单，不用估计值补齐。

### 7.3 结束前

- 运行当天数量、schema、哈希、非空、非有限值和相对链接验证；
- 将当天 README 从“未开始”改为与证据一致的真实状态；
- 只把通过门禁的必要成果投影到教师目录；
- 检查 Git 未包含权重、缓存、凭据、真实个人数据或平台关机证明；
- 使用 AutoDL 后确认实例已经关机。

## 8. Day 22：模型、环境与五张图片

### 8.1 当日目标

按 24GB 显存路线固定 `Qwen/Qwen2-VL-7B-Instruct`，验证模型可离线加载并完成一条图文 smoke；准备五类各一张、来源和 ground truth 明确的图片。

### 8.2 实施步骤

1. 检查 GPU 型号、可用显存、磁盘与网络；记录 Python、PyTorch、CUDA 和驱动；
2. 解析 Qwen 模型仓库的不可变 commit，使用该 revision 下载到持久化目录；
3. 安装兼容版本的 Transformers、Accelerate、`qwen-vl-utils`、Pillow 等依赖并记录版本；
4. 对模型目录普通文件生成相对路径、字节数、逐文件 SHA-256、总文件数和总大小；
5. 断网或 `local_files_only=True` 加载 processor/model，完成一条图文 smoke 并记录峰值显存；
6. 准备 `w5-table-01`、`w5-scene-01`、`w5-logo-01`、`w5-formula-01`、`w5-ui-01`；
7. 为每张图保存来源、许可、尺寸、文件字节、SHA-256 和人工 ground truth；
8. 验证五种类型恰好各一张，图片能由 Pillow 解码，且不包含真实敏感信息。

### 8.3 完成门禁与交付

- 模型仓库、不可变 revision、环境版本和完整文件清单存在；
- 离线加载和一条图文 smoke return code 为 0；
- 五张图片均可打开，五类无缺失、无重复 ID；
- `image_manifest.csv` 字段完整并与文件哈希一致；
- 工程交付为模型下载确认、环境清单、smoke 结果和图片素材包；
- 以上任一缺失则 Day 22 保持未完成。

## 9. Day 23：25 条图文推理

### 9.1 当日目标

对五张冻结图片分别运行内容描述、文字提取、结构/图表解释、美学评价和隐含信息推理，形成固定 5×5 推理矩阵。

### 9.2 实施步骤

1. 核对 Day 22 模型 revision、五张图片 SHA 和 processor 配置；
2. 在推理前冻结五类 Prompt 模板与 25 个唯一 `record_id`；
3. 固定 `do_sample=false`、`seed=42`、`max_new_tokens`、`min_pixels` 和 `max_pixels`；
4. 按预注册顺序运行 25 条，保存原始 Prompt/回答、token 数、延迟和峰值显存；
5. 对每条回答记录优势标签、幻觉标签和具体审核理由；
6. 保留不适配题型和失败回答，以展示模型边界；
7. 验证每个 `image_id × question_type` 组合只出现一次；
8. 汇总五类任务的强项、弱项和典型幻觉，但不改写原始回答。

### 9.3 完成门禁与交付

- `inference_results.csv` 正好 25 条，5 张图和 5 类问题计数均正确；
- 所有记录带模型 revision、图片 SHA、配置 SHA 和原始回答；
- 无重复 `record_id`，无缺失组合，无推理后删题；
- 人工结论有逐条理由，且审核者类型如实记录；
- 教师交付为 25 条图文推理结果表和必要图片索引。

## 10. Day 24：跨模态注意力可视化

### 10.1 架构口径

Qwen2-VL 的视觉 token 经视觉编码器和 merger 后进入语言模型序列，语言模型通过 self-attention 完成文本与视觉交互。因此老师所说的 Cross-Attention 在本实验中定义为：

> 指定语言模型层中，目标文本 token 对视觉 token 区间的 self-attention 权重。

该热力图展示模型内部关注分布，不证明该区域与最终输出存在严格因果关系。

### 10.2 实施步骤

1. 从 processor 输入定位视觉 token 起止索引和 `image_grid_thw`；
2. 优先用 `attn_implementation="eager"` 获取第 20 层或最接近有效层的 attention；
3. 若实现不返回权重，使用 PyTorch Hook 捕获 Q/K，并结合真实 mask 重算缩放点积 attention；
4. 为目标生成词执行单步或 teacher-forced forward，准确记录 token ID 和解码文本；
5. 保存逐头原始数组，对 heads 求均值后截取目标文本 token 指向视觉 token 的权重；
6. 根据 merger 和视觉网格还原二维布局，上采样到原图尺寸；
7. 输出原图、热力图和 overlay 三联图，同时保留 `.npy`、元数据 JSON 和 SHA-256；
8. 至少完成 3 张不同图片、3 个不同目标词；
9. 不使用随机高斯图、Grad-CAM 或纯 ViT self-attention 冒充跨模态注意力。

### 10.3 完成门禁与交付

- 至少 3 个案例，每个案例都有图片、Prompt、目标 token、层号、视觉 token 范围和网格；
- 原始数组数值有限且形状与记录一致；
- 三联图能由元数据和数组重新生成；
- 报告说明跨模态定义、head 聚合方式和方法限制；
- 教师交付为至少 3 张注意力热力图及简要说明。

## 11. Day 25：视觉幻觉检测

### 11.1 当日目标

使用固定 10 组图片-问题-假答案/错误前提测试模型是否会确认图片中不存在的视觉事实，并按统一二元规则计算幻觉率。

### 11.2 输入与评分

每题必须在推理前冻结：`case_id`、`image_id`、问题、诱导性假答案或错误断言、人工 ground truth、期望安全行为、图片 SHA 和配置 SHA。

- `hallucination=1`：模型确认、扩展或依据不存在的物体、文字、数值、关系或动作继续推理；
- `hallucination=0`：模型纠正错误前提、明确表示图中不存在，或在视觉证据不足时表达合理不确定；
- 回答不够详细本身不计幻觉，除非引入不存在的视觉事实。

主指标固定为：

```text
hallucination_rate = hallucination_count / 10
```

### 11.3 实施与完成门禁

1. 在运行模型前冻结 10 题 JSON/CSV 和 ground truth；
2. 固定 Day 22 模型、图片 SHA 和 Day 23 生成参数；
3. 按预注册顺序运行全部 10 题并保存原始回答；
4. 按二元 rubric 逐题评分并写明理由；
5. 计算分子、分母和百分比，三者必须一致；
6. 报告所有失败案例，不删除题目、不改 ground truth、不重新选择更好回答；
7. 教师交付为 10 题明细、幻觉率和典型错误分析组成的幻觉检测报告。

## 12. Day 26：200 条数据与冻结视觉塔 LoRA

### 12.1 数据设计

- 正式训练集恰好 200 条；
- 开发与最终评测各 20 条，均不计入 200 条；
- 三组不得共享图片 SHA-256，训练与评测目标答案不得存在精确副本；通用任务 Prompt 模板的复用必须在报告中披露；
- 每条记录包含唯一 ID、图片相对路径、用户指令、目标回答、任务类型、来源和许可；
- 覆盖产品/场景描述、OCR、表格摘要、UI 操作、公式解释等核心能力；
- 使用锁定版本 LLaMA-Factory 实际支持的多模态 ShareGPT schema，并由真实 processor 加载 smoke 验证。

### 12.2 训练设计

1. 验证 200+20+20 数量、ID、文件存在性、许可字段、消息角色、非空回答、图片 SHA 零重叠和目标答案零精确重叠；
2. 固定 Qwen2-VL 基座 revision、LLaMA-Factory revision、数据 SHA、配置和 `seed=42`；
3. 配置多模态 SFT LoRA，并明确 `freeze_vision_tower: true`；
4. 先用少量样本运行 processor/schema smoke 和单个 optimizer step；
5. smoke 后验证 adapter 文件非空，且视觉塔参数未加入可训练参数或未发生更新；
6. 正式训练保存有效 YAML、启动命令、日志、step loss、学习率、grad norm、峰值显存和 return code；
7. 生成 adapter 文件清单、总大小、逐文件 SHA 和完整 lineage；
8. Base 与 LoRA 使用相同 20 条隔离评测和生成参数产生原始回答；
9. 隐藏模型身份，按视觉事实正确性 35%、指令完成度 25%、完整性 15%、有用性 15%、格式 10% 评分；
10. 计算三项预注册提升条件，未达标时保留失败 run 摘要并排查数据、学习率、rank、epoch 和视觉 token 上限。

### 12.3 明显提升门槛

```text
train_count = 200
heldout_count = 20
mean_gain = lora_mean - base_mean >= 0.50 / 5
lora_wins >= 12 / 20
lora_hallucination_rate <= base_hallucination_rate
```

三项质量条件必须同时满足。Train loss 下降、训练自然结束或 adapter 可加载不能替代质量提升结论。

### 12.4 完成门禁与交付

- 200 条训练、20 条开发和 20 条最终记录通过 schema、数量、来源字段和图片污染检查，并披露通用 Prompt 模板复用边界；
- 正式训练 return code 为 0、指标有限、adapter 非空且视觉塔保持冻结；
- Base/LoRA 各 20 条原始回答齐全，盲评映射不进入教师提交；
- 三项明显提升门槛同时通过，或如实标记未通过并继续纠偏；
- 教师交付为微调后的 VLM 归档信息、有效配置、训练日志和效果对比。

## 13. Day 27：周报与最终 VLM 归档

### 13.1 周报结构

《第 5 周：多模态实践报告》至少包含：

1. 任务背景与老师要求；
2. Qwen2-VL 模型、revision、环境和图片素材血缘；
3. 25 条图文推理结果与能力边界；
4. 3 张以上跨模态注意力热力图、计算定义和限制；
5. 固定 10 题幻觉率与失败案例；
6. 200 条训练数据与 20 条隔离评测的数据质量；
7. LoRA 配置、冻结视觉塔验证、loss 和 adapter 清单；
8. Base/LoRA 20 条隔离评测与三项提升门槛；
9. 老师四项验收矩阵；
10. 失败尝试、局限、后续改进和最终结论。

### 13.2 最终模型归档

归档元数据必须记录最终模型类型（adapter 或 merged）、远端路径、基座仓库/revision、训练数据 SHA、配置 SHA、adapter/模型逐文件 SHA、总大小、加载 smoke、隔离评测结果和生成时间。模型大权重保留在远端持久化目录，不复制到 Git 或教师 ZIP。

### 13.3 完成门禁与教师提交

1. 对 Day 22–Day 26 的数量、哈希、日志和结果做跨日审计；
2. 用原始数据重新生成表格、热力图索引和对比摘要；
3. 周报中的每个数字能追溯到 CSV/JSON 或训练日志；
4. 建立老师四项验收矩阵并逐项链接证据；
5. 最终模型 manifest 与远端真实文件一致，独立加载 smoke 通过；
6. 只有全部硬门槛通过时才宣称 Week 5 完成；
7. 此时再创建独立 `Submission/Week5/`，仅放老师要求的模型确认、图片包、25 条记录、3+ 热力图、幻觉报告、训练日志、效果对比、周报和模型归档元数据；
8. 教师目录不包含模型权重、checkpoint、缓存、凭据、盲评身份映射或 AutoDL 关机证据。

## 14. 故障处理顺序

### 14.1 模型下载或 revision 不可用

先保存仓库错误和网络状态；重新解析官方仓库 commit，不用浮动 `main` 作为最终 lineage。7B 在 24GB 环境确实无法完成最小推理时，先降低图片 token 上限和量化加载；只有老师允许时才改用 2B，且报告必须说明偏离原因。

### 14.2 图片 token 过多或 CUDA OOM

保持原图文件不变，按预注册顺序降低 `max_pixels`、使用 bf16/4-bit、关闭不必要缓存和减小输出长度。修改预处理参数后生成新配置 SHA，并对同组正式实验全部重跑，不能混合不同配置结果。

### 14.3 取不到 attention

先切换 eager attention；仍不可用时 Hook 捕获 Q/K 并用真实 mask 重算。若实际层数不足 21 层，选择最接近第 20 层的有效层并记录。不能用随机权重、Grad-CAM 或纯视觉塔权重替代。

### 14.4 幻觉题存在歧义

在模型运行前由图片事实核验解决歧义；无法唯一判断的题在冻结前替换。运行后发现歧义时保留原题与回答并将主测试整体重新预注册，不能只替换失败题。

### 14.5 LoRA 数据或 schema 不兼容

先用锁定版本 LLaMA-Factory 的 dataset parser 和真实 processor 验证 1–4 条；记录实际支持字段，再修改全量转换脚本。不能根据旧版示例猜测字段或跳过图片存在性检查。

### 14.6 微调没有明显提升

保留原 run、日志和评测摘要；优先排查数据目标质量、训练/评测污染、学习率、LoRA rank、epoch、图片分辨率和输出模板。隔离 20 题保持冻结；重新训练后对全部 20 题重跑，不能只重跑原来的失败项。

## 15. 最终交付清单

| 交付 | 最低完成证据 |
|---|---|
| 模型下载确认 | 官方仓库、不可变 revision、文件清单、总大小、SHA、离线加载和 smoke |
| 图片素材包 | 五类各一张、manifest、许可、尺寸、文件字节、SHA 和 ground truth |
| 25 条推理记录 | 25/25 唯一组合、原始回答、模型/图片/配置哈希和人工审查 |
| 3+ 注意力热力图 | 原始数组、层/token/网格元数据、三联图和方法说明 |
| 幻觉检测报告 | 固定 10 题、原始回答、二元评分、理由和固定分母指标 |
| 微调 VLM 与日志 | 200 条训练、冻结视觉塔、有效配置/日志、adapter 清单和加载验证 |
| 微调效果对比 | 20 条隔离集 Base/LoRA 原始回答、盲评和三项提升门槛 |
| 第 5 周报告 | 全周事实、四项验收矩阵、限制与最终模型 lineage |

## 16. 一手参考

- [Qwen2-VL-7B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Qwen2-VL 官方技术介绍](https://qwenlm.github.io/blog/qwen2-vl/)
- [Qwen2-VL 论文](https://arxiv.org/abs/2409.12191)
- [LLaMA-Factory 官方仓库](https://github.com/hiyouga/LlamaFactory)

本计划整理完成不表示任何 Week 5 GPU 实验已经执行。Day 22–Day 27 的真实结果只能由后续运行产生并通过对应门禁后写入。
