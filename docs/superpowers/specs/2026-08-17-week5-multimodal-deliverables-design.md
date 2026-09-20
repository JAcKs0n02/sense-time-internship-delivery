# Week 5 多模态实践交付设计

## 1. 目标

将新版 `实习需求.pdf` 中 Week 5 的 Day 22–Day 27 要求转换为可逐日执行、可验证、可独立提交的仓库结构。规划阶段只建立工程说明与执行口径，不提前生成空结果、不把预期值写成实验事实，也不提前创建教师正式交付物。

Week 5 的最终目标是：

1. 使用老师指定的 Qwen2-VL 完成图文推理；
2. 生成至少 3 张可解释的跨模态注意力热力图；
3. 用固定 10 题量化幻觉率；
4. 使用 200 条图文指令数据完成冻结视觉编码器的 LoRA 微调；
5. 在隔离评测集上证明微调后有明显效果提升；
6. 完成《第 5 周：多模态实践报告》和最终 VLM 模型归档。

## 2. 范围与边界

### 2.1 本轮立即创建

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

这些文件记录老师要求、术语、实施步骤、预期文件、验收门槛、故障处理和当前状态。所有日期初始状态均为“未开始”。

### 2.2 本轮不创建

- 不创建空的 `Submission/Week5/`。
- 不创建虚假的模型下载确认、推理结果、热力图、幻觉率、微调日志或周报。
- 不下载模型、不启动 AutoDL、不运行 GPU 实验。
- 不保存模型权重、checkpoint、缓存、凭据或平台操作证据。

`Submission/Week5/` 只在某一天的实验完成、证据校验通过后增量建立。最终教师提交继续采用 Week 4 的独立目录和独立 ZIP 原则，不要求老师接收整个仓库。

## 3. 老师要求映射

| Day | 老师任务 | 老师交付 | 规划后的证据门禁 |
|---|---|---|---|
| 22 | 按显存选择并下载 Qwen2-VL；安装 `qwen-vl-utils`；准备 5 张指定类型图片 | 模型下载确认、图片素材包 | 模型 revision 与清单固定；五类图片各 1 张；ID、来源、许可、尺寸、SHA-256 完整 |
| 23 | 每张图运行描述、OCR、图表解释、美学评价、隐含信息推理五类问题 | 25 条图文推理结果表 | 5 张图 × 5 类问题 = 25 条唯一记录；原始回答、生成参数和人工结论齐全 |
| 24 | 使用 Hooks 提取中间层注意力并绘制热力图 | 至少 3 张注意力热力图 | 每张图可追溯到图像、Prompt、目标 token、层号、头聚合方法和原始注意力数组 |
| 25 | 构造 10 组图片-问题-假答案对并统计幻觉率 | 幻觉检测报告 | 固定 10 题和二元评分规则；逐题原始回答、判断和理由齐全；分母固定为 10 |
| 26 | 构造 200 条图文指令数据；冻结 ViT，仅训练 LLM LoRA | 微调后的 VLM、微调日志 | 200 条正式训练数据；独立 20 条评测；配置、日志、adapter 清单和前后对比完整 |
| 27 | 补全实验并撰写周报 | 《第 5 周：多模态实践报告》 | 四项周验收逐条映射；最终模型 lineage 和未达标项如实归档 |

## 4. 模型与运行环境设计

既有 GPU 为 NVIDIA RTX 3090 24GB，满足老师“16GB+ 使用 7B”的条件，因此 Day 22 正式模型固定为：

```text
Qwen/Qwen2-VL-7B-Instruct
```

模型 revision 必须在下载时固定，禁止长期使用可漂移的 `main` 作为唯一血缘。Day 22 记录：

- 模型仓库与 revision；
- Transformers、PyTorch、CUDA、`qwen-vl-utils` 和 LLaMA-Factory 版本；
- 模型目录普通文件清单、总大小和逐文件 SHA-256；
- 离线加载结果；
- 一条图文 smoke 的输入、输出和显存峰值。

Qwen2-VL 采用动态分辨率，会把不同尺寸的图片转换为不同数量的视觉 token。正式推理必须固定 `min_pixels`、`max_pixels` 和输入图像文件，避免同一图片因预处理参数变化而产生不同 token 网格。参考：[Qwen2-VL 官方模型卡](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)、[Qwen2-VL 官方介绍](https://qwenlm.github.io/blog/qwen2-vl/)。

## 5. Day 22 图片素材设计

五张图片固定为：

| image_id | 类型 | 内容约束 |
|---|---|---|
| `w5-table-01` | 表格截图 | 自建业务统计表，含中英文、数值和表头，不含真实个人数据 |
| `w5-scene-01` | 自然风景 | 授权明确或自行拍摄，包含可核验的前景与背景对象 |
| `w5-logo-01` | Logo | 自建虚构品牌 Logo，避免商标和再分发争议 |
| `w5-formula-01` | 手写公式 | 自行书写的多步公式，保留正确文本答案 |
| `w5-ui-01` | UI 界面 | 自建静态业务界面，不包含真实账号、客户或密钥信息 |

图片素材包必须带 `image_manifest.csv`，至少包含 `image_id`、相对路径、类型、来源、许可、宽高、文件字节和 SHA-256。图片一旦进入 Day 23 正式评测便冻结，后续修改必须更换 ID。

## 6. Day 23 推理设计

对五张冻结图片分别运行五类 Prompt：

1. 内容描述；
2. 文字提取；
3. 图表或结构解释；
4. 美学评价；
5. 隐含信息推理。

即使某类任务与图片不完全匹配，也保留问题，以观察能力边界。正式结果表固定 25 条，字段至少包括：

```text
record_id, image_id, question_type, prompt, raw_response,
input_tokens, output_tokens, latency_seconds, peak_gpu_memory_mib,
strength_label, hallucination_label, reviewer_reason,
model_revision, image_sha256, generation_config_sha256
```

生成默认使用确定性参数：`do_sample=false`、`seed=42`、固定 `max_new_tokens`。若模型因为同一输入重复推理产生差异，只保存预先约定的第一次成功结果，不挑选更好的回答。

## 7. Day 24 注意力可视化设计

### 7.1 架构口径

老师原文使用“Cross-Attention 权重”。Qwen2-VL 的视觉 token 经视觉编码器和 merger 后进入语言模型 token 序列，语言模型使用 self-attention 完成视觉-文本交互，并不存在可直接按名称获取的典型独立 `cross_attn` 层。因此教师材料统一使用“跨模态注意力”表述，并明确其计算定义：

> 指定语言模型层中，目标文本 token 对视觉 token 区间的 self-attention 权重。

不能把视觉编码器内部的纯图像 self-attention 伪称为文本到图像 Cross-Attention。

### 7.2 提取方式

- 使用 PyTorch Hook 记录第 20 层 attention 输入或 Q/K 投影；
- 使用 eager attention 或等价的可返回权重实现，禁止在 FlashAttention 不返回权重时伪造数据；
- 对指定生成词执行单步或 teacher-forced forward；
- 对所有 attention heads 取均值，同时保留逐头原始数组；
- 只选择目标 token 对视觉 token 索引区间的权重；
- 使用 processor 返回的视觉网格信息还原合并后的二维 token 网格；
- 上采样到原图尺寸，生成原图、热力图和 overlay 三联图。

至少生成 3 个不同图片、不同目标词的热力图。每个案例记录 layer、target token、视觉 token 范围、网格尺寸、聚合方式、数组 SHA-256 和图像 SHA-256。

### 7.3 失败回退

如果当前 Transformers 实现无法直接返回 attention：

1. 先切换为 `attn_implementation="eager"`；
2. 若仍不可用，通过 Hook 捕获 Q/K 并按缩放点积重新计算掩码后的 attention；
3. 若第 20 层不存在，以模型配置中的实际层数为准选择最接近 20 的有效层，并在报告中说明；
4. 不以 Grad-CAM、随机高斯图或视觉编码器单模态权重冒充老师要求的跨模态映射。

## 8. Day 25 幻觉检测设计

构造 10 条带错误前提的固定测试，每条包括：

- 冻结图片；
- 中性问题；
- 作为诱导上下文的假答案或错误断言；
- 基于图片人工确认的 ground truth；
- 期望行为：纠正错误前提或明确表示图片中不存在该内容。

评分规则：

- `hallucination=1`：模型确认、扩展或据此推理图片中不存在的物体、文字、数值、关系或动作；
- `hallucination=0`：模型拒绝错误前提、表达不确定或给出与图片一致的纠正；
- 仅语言表达不够详细不计为幻觉，除非引入了不存在的视觉事实。

主指标固定为：

```text
hallucination_rate = hallucination_count / 10
```

禁止删除失败题、修改分母或在看完模型回答后重写 ground truth。

## 9. Day 26 数据与微调设计

### 9.1 数据

- 正式训练集：200 条图文指令数据；
- 隔离评测集：额外 20 条，不计入 200 条训练数据；
- 训练集和评测集不得共享相同图片 SHA-256；
- 每条记录包含唯一 ID、图片相对路径、用户指令、目标回答、任务类型和来源；
- 覆盖产品描述、OCR 转写、表格摘要、UI 操作说明、公式解释等 Week 5 核心能力；
- 使用 LLaMA-Factory 支持的多模态 ShareGPT 格式，并先做真实 processor schema smoke。

### 9.2 训练

训练使用 LLaMA-Factory SFT LoRA：

- 基座：Day 22 冻结的 Qwen2-VL-7B-Instruct；
- `freeze_vision_tower: true`；
- 语言模型使用 LoRA，视觉编码器保持冻结；
- 先运行少量样本、单步 smoke，再启动正式训练；
- 保存有效配置、启动日志、逐步 loss、显存峰值、adapter 清单和 SHA-256；
- 权重不进入 Git 或教师 ZIP，教师材料只保存模型归档元数据。

LLaMA-Factory 已支持 Qwen2-VL 多模态 SFT 和 LoRA/QLoRA，具体配置在执行日根据仓库锁定版本做静态验证。参考：[LLaMA-Factory 官方仓库](https://github.com/hiyouga/LlamaFactory)。

## 10. 微调效果验收设计

老师要求“VLM 微调后有明显效果提升”。为避免训练后改变标准，Day 26 开始前冻结以下门槛：

1. Base VLM 与 LoRA VLM 使用相同 20 条隔离评测和相同生成参数；
2. 随机隐藏模型身份后评分；
3. 五维权重为：视觉事实正确性 35%、指令完成度 25%、完整性 15%、有用性 15%、格式 10%；
4. LoRA 模型平均加权分至少比 Base 高 `0.50/5`；
5. LoRA 模型至少在 12/20 条上胜出；
6. LoRA 模型在隔离集上的视觉幻觉率不得高于 Base；
7. 三项量化条件必须同时满足才标记“明显提升 PASS”。

若未达到，不以 train loss 下降替代质量门槛。可以在保持评测集冻结的前提下排查数据、学习率、LoRA rank、训练轮数和图片 token 上限，重新训练后再运行同一最终评测；报告必须保留失败尝试摘要。

## 11. Day 27 报告与最终归档

周报至少包含：

- 模型、图片素材和环境血缘；
- 25 条图文推理能力边界；
- 3 张以上跨模态注意力热力图及方法限制；
- 10 题幻觉率；
- 200 条训练数据质量；
- LoRA 配置、loss 和模型清单；
- Base/LoRA 20 条隔离评测对比；
- 四项老师验收矩阵；
- 未达到的指标、原因和后续改进。

最终模型归档保存模型路径、基座 revision、数据哈希、配置哈希、adapter 哈希、文件大小、加载 smoke 和评测结果，不把约 7B 模型权重复制到教师 ZIP。

## 12. 验证与错误处理

### 12.1 规划文件验证

- 八个规划文件全部存在；
- 每个 Day README 包含老师要求、术语、步骤、预期交付、验收门槛和故障处理；
- 所有相对 Markdown 链接可解析；
- 全部状态为“未开始”；
- 不出现虚假的 PASS、完成数量、训练指标或模型哈希；
- 不创建 `Submission/Week5/`。

### 12.2 后续实验通用原则

- GPU 运行前固定输入、revision、配置和随机种子；
- 每个正式实验先 smoke，再正式运行；
- 日志、状态和结果均 fail-closed：缺证据即不标记完成；
- 远端大文件保留在 AutoDL 持久化数据目录；
- 每次使用 AutoDL 后确认关机，但教师提交不包含关机证据；
- 不提交账号、密码、SSH 指令、私有盲评映射或真实个人数据。

## 13. 完成定义

本轮“Week 5 文件整理完成”只表示：

- 老师 Day 22–Day 27 要求已经完整映射；
- 执行顺序、数据合同、指标和技术回退已经预注册；
- Week 5 工程目录和总计划可用于后续逐日执行；
- 没有把尚未执行的实验描述成已完成。

它不表示 Day 22 模型已经下载，也不表示 Week 5 任一实验已经运行。
