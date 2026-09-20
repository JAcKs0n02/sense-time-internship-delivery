# Week 7 量化、部署与交互应用交付设计

## 1. 目标

将老师在最新版 `实习需求.docx` 与 `实习需求.pdf` 中新增的 Week 7、Day 34–Day 39 要求，落实为一个可复现、可验证、可独立提交的量化部署系统。系统以 Week 4 最终 DPO 文本模型为量化输入，以 Week 5 已验证的 Qwen2-VL 基座模型为图片理解后端，最终通过一个 Gradio 页面提供文字聊天、历史记录、生成参数控制、流式输出和图片上传。

Week 7 的完成目标是：

1. 生成可加载的 AWQ 和 GPTQ 两种 4-bit 文本模型；
2. 在同一冻结口径下比较 BF16、AWQ、GPTQ 的模型体积、实际加载显存、生成速度和困惑度；
3. 使用 vLLM 暴露 OpenAI 兼容的 `/v1/chat/completions` 服务；
4. 使用 Gradio 构建流式文字/图片交互界面；
5. 提供启动脚本、客户端、演示录屏、本地部署指南和 Week 7 报告；
6. 对每一项老师验收要求给出真实证据，不以预计值、静态文件或伪造日志代替远端实验。

## 2. 老师要求映射

| Day | 老师任务 | 老师交付 | 工程证据门禁 |
|---|---|---|---|
| 34 | 理解 AWQ/GPTQ；用 Week 4 DPO 模型完成 AWQ | AWQ 量化模型 | 模型 lineage、校准集、量化配置、完整日志、权重清单、离线加载 smoke 和 SHA-256 完整 |
| 35 | 完成 GPTQ；对比 FP16/AWQ/GPTQ | 量化对比表和报告 | 三模型使用同一数据与参数；显存、tokens/s、PPL/NLL 原始记录可复算；至少一种量化显存下降 30% |
| 36 | vLLM 加载量化模型并提供 OpenAI 接口 | 启动脚本、Python 客户端 | 服务有明确 `served-model-name`；健康检查、同步和流式请求均成功；保存请求、响应和服务日志 |
| 37 | Gradio 输入、历史、temperature/top-p 和流式输出 | `app.py` | 参数真实透传；历史格式正确；`stream=True` 增量展示；后端异常有可理解提示 |
| 38 | 优化 UI；图片上传接入 Week 5 VLM | 演示录屏、优化说明 | 图片请求路由到真实 VLM；不把文本模型冒充多模态模型；录屏覆盖文字流式与图片问答 |
| 39 | 本地部署、启动顺序、环境变量和周报 | 部署指南、Week 7 报告 | 从空终端可按文档启动；敏感值不入库；周报逐项链接原始证据和未达项 |

## 3. 已批准的技术路线

采用“兼容且可审计”方案：

- AWQ：使用 AutoAWQ 对 Week 4 最终 DPO merged model 做 4-bit AWQ 导出；
- GPTQ：使用仓库既有 LLaMA-Factory 0.9.3 的 GPTQ 导出流程；
- 服务：使用仓库基线 vLLM 0.6.4.post1；
- 页面：使用 Gradio 和 OpenAI Python 客户端；
- 多模态：使用 Week 5 已验证的 Qwen2-VL-7B-Instruct 基座模型；
- 显卡：AutoDL 单张 NVIDIA RTX 3090 24GB；
- 运行方式：文字后端与视觉后端分进程、按需切换，默认不在单卡上同时常驻。

不升级或原地修改 Week 1–Week 6 的历史运行环境。AWQ、GPTQ 和 serving 使用隔离环境；正式运行前分别保存 Python、CUDA、PyTorch、Transformers、量化库、LLaMA-Factory、vLLM、Gradio 和 OpenAI 客户端的实际版本清单。依赖解析结果只有通过模型加载 smoke 后才能冻结到 Week 7 的 requirements 文件。

## 4. LLaMA-Factory AWQ 兼容性偏差

老师原文提示使用 `--quantization_method awq`。仓库锁定的 LLaMA-Factory 0.9.3 中，该字段用于加载时的量化选择，而 `export_quantization_bit` 导出路径实际构造 GPTQ 配置；AWQ 分支只能识别已经带有 AWQ `quantization_config` 的模型。它不能按老师提示把未量化的 Week 4 模型直接导出成 AWQ。

Day 34 必须同时保留两类证据：

1. `llamafactory_awq_compatibility.json`：记录版本、老师提示、源码/参数检查和最小命令返回结果；
2. AutoAWQ 正式导出记录：记录校准集、`w_bit=4`、`q_group_size=128`、`zero_point=true`、`version=GEMM`、输出目录、权重清单和加载 smoke。

报告统一表述为：“依据老师的 AWQ 目标完成量化；由于固定版本不提供 AWQ 导出，使用兼容的 AutoAWQ 导出器，并保留 LLaMA-Factory 兼容性收据。”不得把 AutoAWQ 结果写成由 LLaMA-Factory 0.9.3 直接导出。

参考实现证据：

- [LLaMA-Factory 0.9.3 量化实现](https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/v0.9.3/src/llamafactory/model/model_utils/quantization.py)
- [LLaMA-Factory 0.9.3 参数定义](https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/v0.9.3/src/llamafactory/hparams/model_args.py)
- [LLaMA-Factory 官方模型合并与 GPTQ 导出说明](https://llamafactory.readthedocs.io/en/latest/getting_started/merge_lora.html)

## 5. 模型血缘与前置门禁

### 5.1 Week 4 文本模型

正式量化输入固定为：

```text
model_id: reward_corrective_40step_merged
remote_path: /root/autodl-tmp/qwen25-week4/best_model/qwen25-7b-week4-dpo-corrective-merged
manifest_sha256: aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c
expected_weight_bytes: 15231271872
```

量化前必须验证模型配置、Tokenizer、四个 BF16 权重分片、总字节数、离线生成和既有 manifest。若远端 merged 目录缺失，允许使用 Week 3 SFT merged base 加 Week 4 最终 DPO adapter 执行 `merge_and_unload()` 重建等价模型，但必须：

- 使用既有 DPO adapter SHA `d7a932ee4f28c8950db289126381f5d4dd30a037b238852e87ad8e5a241f9e52` 对应的完整仓库清单；
- 保存重建命令、精确基座与 adapter 清单；
- 重新计算完整 merged manifest；
- smoke 通过后才允许量化；
- 不静默回退到 Week 3、Week 2 或原始 Qwen 模型。

### 5.2 Week 5 视觉模型

正式图片后端使用 Week 5 已验证基座：

```text
model_id: Qwen2-VL-7B-Instruct
revision: eed13092ef92e448dd6875b2a00151bd3f7db0ac
remote_path: /root/autodl-tmp/qwen2-vl-week5/models/Qwen2-VL-7B-Instruct
files_digest_sha256: 575d62b8abe01fda09b28a02ff8e6cd04e8df0d15fd03e1642d3a22bbe7eb812
```

Week 5 最终归档状态为 `NO_ACCEPTANCE_PASSING_MODEL`，因此 v7-B LoRA 只能作为标注清楚的实验开关，不能作为 Week 7 默认后端，也不能描述成质量已提升。正式演示和验收默认使用基座 VLM。

## 6. 冻结数据与评测口径

### 6.1 校准数据

从 `deliverables/week2/day7/data/week2_clean_sharegpt.jsonl` 确定性抽取 128 条文本作为 AWQ/GPTQ 共用校准集：

- 随机种子 `42`；
- 先按稳定记录哈希排序，再抽样，避免依赖文件当前顺序；
- 使用 Week 4 Tokenizer 和 Qwen chat template；
- 每条最多 1024 tokens；
- 排除 Week 4 安全/业务评测题和 Week 7 质量评测题；
- 保存 JSONL、来源行号、内容 SHA-256、Tokenizer revision 与统计清单。

### 6.2 困惑度数据

从同一清洗语料中选择与校准集不重叠的 256 条记录，固定为 512-token 窗口。三模型使用同一 Tokenizer、同一有效 token mask、同一窗口和同一累积精度计算 NLL/PPL。出现 OOM 时只允许统一缩小所有模型的 batch size，不能只给某个量化模型有利参数。

### 6.3 生成评测

固定 20 条文字任务：10 条沿用 Week 4 安全题和 10 条业务/通用任务。三模型使用相同 system prompt、chat template、`temperature=0`、`top_p=1.0`、`max_tokens=256` 和 seed。保存原始输出、输入/输出 token 数、首 token 延迟、端到端延迟和人工质量记录。

## 7. 量化与基准测试合同

### 7.1 AWQ

- 4-bit、group size 128、zero point、GEMM kernel；
- 使用第 6 节冻结校准数据；
- 输出目录只保留在 AutoDL 持久化盘，Git 中保存 manifest、配置、日志和小型 tokenizer/config 文件清单；
- AutoAWQ 导出后必须完成 Transformers 离线加载、一次固定 Prompt 生成和 vLLM 加载 smoke。

### 7.2 GPTQ

- 使用 LLaMA-Factory 0.9.3 的 `export_quantization_bit: 4`；
- 使用相同校准数据和 1024 token 上限；
- 保存有效 YAML、CLI、GPTQModel 版本、stdout/stderr、退出码和输出清单；
- 完成同 AWQ 一致的离线生成与 vLLM smoke。

### 7.3 指标定义

| 指标 | 固定定义 |
|---|---|
| 模型体积 | 模型权重与量化元数据普通文件字节和，不含日志、缓存和 tokenizer 重复文件 |
| 静态加载显存 | 清空 GPU 后加载模型，记录进程前后 `nvidia-smi` 差值和框架 `memory_allocated` |
| 峰值显存 | 完成固定一次 512-token 输入、128-token 输出时进程峰值 |
| tokens/s | 预热 3 次后执行 10 次，使用总输出 tokens 除以 generation wall time，报告中位数与 P10/P90 |
| TTFT | vLLM 流式请求发出至收到首个内容 token 的时间 |
| PPL/NLL | 第 6.2 节冻结窗口上的 token 加权 NLL 及 `exp(NLL)` |

vLLM 会根据 `gpu_memory_utilization` 预留 KV cache，不能直接把总预留显存当作模型显存。老师“降低 30%”的主门禁以同一离线加载/固定生成口径的实际进程显存为准，并用模型体积作为交叉检查：

```text
memory_reduction = 1 - quantized_peak_mib / bf16_peak_mib
```

至少 AWQ 或 GPTQ 的 `memory_reduction >= 0.30`、固定生成无 NaN、vLLM smoke 成功，才标记量化验收 PASS。AWQ 可能降低显存但不提升吞吐，报告只写实测结果。

## 8. 服务与应用架构

```text
                        ┌─────────────────────────┐
用户 ── Gradio app ────┤ Week 7 路由与流式适配器 │
                        └──────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
               无图片消息                    图片 + 文字
                    │                             │
       Week 4 AWQ/GPTQ vLLM            Week 5 Qwen2-VL vLLM
       OpenAI-compatible API            OpenAI-compatible API
```

### 8.1 文本服务

- 默认使用 Day 35 基准中通过验收且综合结果更好的量化模型；
- `served-model-name` 固定为 `week4-dpo-quantized`；
- 提供健康检查、同步请求、流式请求和可重复 demo client；
- 主机、端口、API key、模型路径、GPU memory utilization 和 max model length 全部使用环境变量；
- API key 只使用本地演示占位值或运行时注入，不把真实凭据写入文件。

### 8.2 视觉服务

- 独立进程加载 Week 5 基座 VLM；
- `served-model-name` 固定为 `week5-qwen2-vl-base`；
- Gradio 将图片编码为受限大小的 data URL，并使用 OpenAI 消息 content parts 发送 `text` 与 `image_url`；
- 限制单文件 10 MiB，只接受 JPEG、PNG、WEBP，拒绝无法解码或超限文件；
- 对 Week 5 五张既有图片做固定 smoke，演示至少覆盖一张 OCR/表格图和一张自然图。

### 8.3 单卡运行策略

RTX 3090 24GB 默认只运行一个模型服务。`start_text_server.sh` 和 `start_vlm_server.sh` 在启动前检查 GPU 上是否已有 Week 7 服务进程；发现冲突即退出并提示先停止当前后端。Gradio 可以常驻，但当目标后端不可用时返回明确的“请切换并启动对应服务”消息，不自动杀进程、不自动启动远端实例。

## 9. Gradio 交互合同

页面提供：

- 文本输入；
- 单图上传；
- 对话历史；
- temperature，范围 0.0–2.0，默认 0.7；
- top-p，范围 0.05–1.0，默认 0.9；
- max tokens，范围 32–1024，默认 256；
- 当前路由/模型状态；
- 清空会话和停止生成；
- 流式输出。

应用层只负责输入校验、历史格式转换、路由和流式内容增量，不包含模型推理。文本路由条件是“没有上传图片”，视觉路由条件是“存在一张有效图片”。流式适配器忽略空 delta，累积有效文本并逐次 `yield`；HTTP 401、连接失败、超时、模型名错误和不兼容图片分别映射为用户可读错误，同时把不含凭据的技术详情写入本地日志。

## 10. 仓库结构与文件责任

实施阶段新增：

```text
docs/week7_execution_plan.md
deliverables/week7/
├── README.md
├── day34/
├── day35/
├── day36/
├── day37/
├── day38/
├── day39/
├── source/week7_deployment/
│   ├── config.py
│   ├── messages.py
│   ├── openai_stream.py
│   ├── routing.py
│   └── validation.py
└── tests/
Submission/Week7/
├── Day34_AWQ_Quantization/
├── Day35_GPTQ_and_Benchmark/
├── Day36_vLLM_Service/
├── Day37_Gradio_Application/
├── Day38_Multimodal_Optimization/
├── Day39_Weekly_Report_and_Deployment_Guide/
├── README.md
└── SHA256SUMS.txt
```

Day 目录分别保存配置、脚本、数据 manifest、结果与 README；共享包只保存可复用的输入校验、消息转换、客户端流适配和路由，不复制量化/训练大权重。教师目录是经过验证的浅层投影，并为每个 Day 保存 `Submission_Map.json`，指向仓库中的权威源文件。

全局文件同步更新：

- `README.md`：把 Week 6 从“计划”修正为实际完成，增加 Week 7 状态、结果和目录；
- `Submission/README.md`：增加 Week 6、Week 7，删除过时的 Week 4 单次提交提示；
- `Submission/SHA256SUMS.txt`：重建为包含 Week 1–Week 7 的总清单；
- `.gitignore`：继续排除权重、密钥、远端缓存和大体积临时录像，但允许提交压缩后的正式演示录屏或其可验证清单；
- 历史 Week 1–Week 6 每日 README 中当时真实的时间状态不批量改写。

## 11. 演示录屏和提交策略

正式演示录屏包含：

1. 启动 Gradio 与文字 vLLM；
2. 调整 temperature/top-p；
3. 展示文字逐 token/逐片段流式输出和历史；
4. 切换到视觉服务；
5. 上传 Week 5 冻结图片并完成一次图片问答；
6. 展示部署说明和服务状态。

录屏使用 MP4/H.264、720p、目标时长 2–4 分钟。若压缩后不超过 25 MiB，放入 `Submission/Week7/Day38_Multimodal_Optimization/Demo/`；若超过 25 MiB，仓库只保存文件名、字节数、SHA-256、时长、分辨率和受控外部交付位置，不能用空文件代替。

## 12. 错误处理与状态语义

每一天使用三态：

- `未开始`：没有正式执行证据；
- `进行中`：已有部分证据，但未满足当日全部门禁；
- `完成`：当日脚本、结果、manifest、测试和 README 均验证通过。

以下情况 fail-closed：模型路径或哈希不匹配、校准集污染、量化退出码非零、模型无法加载、PPL 出现非有限值、显存降幅不足 30%、vLLM 请求失败、Gradio 仅返回整段非流式输出、图片实际没有进入 VLM、录屏或部署指南缺失。失败证据保留在结果目录，报告如实标记，不得用手工编辑 PASS 绕过。

如果 AutoDL 会话不可访问，仓库可以完成设计、脚本、配置、冻结数据和本地单元测试，但所有依赖 GPU 的 Day 状态保持“待远端执行”，Week 7 总验收不能标记完成。

## 13. 测试与验证

### 13.1 本地自动化

- 数据抽样确定性、训练/校准/评测互斥和 SHA-256；
- 模型 manifest schema 与 lineage；
- 指标公式、显存降幅门禁和非有限值拒绝；
- OpenAI 消息转换、流式 delta 累积与错误映射；
- 文本/图片路由、文件类型和大小限制；
- 环境变量缺失时 fail-fast；
- Day34–Day39 与 `Submission/Week7` 结构；
- Markdown 相对链接和两级 SHA256SUMS 完整性；
- 密钥、模型权重和绝对本地隐私路径泄漏扫描。

所有新增行为遵循测试先行：先运行目标测试并确认因功能缺失而失败，再实现最小代码并运行全套 Week 7 回归。

### 13.2 AutoDL 集成验证

- Week 4 merged 模型 preflight；
- AWQ 与 GPTQ 导出及离线加载；
- BF16/AWQ/GPTQ 同口径基准；
- vLLM 同步/流式客户端；
- Gradio 文字与图片 smoke；
- 录像中的场景可由保存的请求/响应和服务日志对应；
- 环境关闭前同步全部小型证据回仓库，大权重保留在持久化盘。

## 14. 安全与审计边界

- 不提交模型大权重、checkpoint、HF 缓存、AutoDL 凭据、SSH 命令中的秘密、API key 或个人数据；
- 不在脚本中执行自动关机、删除远端模型或覆盖历史 Week 4/Week 5 模型；
- 所有远端输出写入新的 `/root/autodl-tmp/qwen25-week7/` 命名空间；
- 每次正式运行保存 command、cwd、git commit、UTC/本地时间、退出码和环境快照；
- 本地老师 Word/PDF 作为需求源保留，不在 Week 7 实施中修改；
- 用户现有未提交修改与无关文件不纳入 Week 7 提交。

## 15. 完成定义

只有同时满足以下条件，仓库才能声明 Week 7 完成：

1. AWQ 与 GPTQ 都有可加载模型清单和远端路径；
2. BF16/AWQ/GPTQ 对比表来自同一冻结协议；
3. 至少一种量化模型实际加载显存下降不低于 30%；
4. vLLM 的同步与流式 OpenAI 兼容请求成功；
5. Gradio 文字聊天、历史、参数控制和流式输出通过；
6. 图片上传真实路由到 Week 5 基座 VLM；
7. 演示录屏、优化说明、本地部署指南和周报存在且可验证；
8. `deliverables/week7`、`Submission/Week7`、全局 README 和校验和全部一致；
9. 本地测试和远端验收矩阵均无未解释失败；
10. 所有未达指标在报告中明确标记，不以计划值冒充实验事实。
