# Qwen2.5 实习第 1 周执行计划

## 1. 文档用途与执行依据

这份文档是第 1 周任务的长期执行入口，用于回答三个问题：今天要做什么、完成后保留什么、汇报时需要讲清什么。

执行时遵循以下优先级：

1. 老师发布的原始任务与验收标准具有最高优先级。
2. 本计划把老师要求拆成可执行步骤，并结合当前已经验证的硬件、环境和模型路径。
3. 每天完成后，在对应 `deliverables/week1/dayN/README.md` 中记录实际结果和文件入口。
4. 如果本计划与老师后续通知冲突，以老师最新通知为准，并同步更新本计划。
5. 命令输出、模型回答和失败日志必须保留原始内容，不因结果不理想而修改。

## 2. 本周目标与最终验收

本周需要完成 CUDA 开发环境、Qwen2.5 原生推理、架构分析、Tokenizer 实验、LLaMA-Factory 训练管线和周报，最终满足以下五项验收：

- CUDA 可用，`torch.cuda.is_available()` 输出 `True`。
- Qwen2.5 基座模型能够正常对话。
- 能口头解释 RoPE 和 GQA 的原理与设计取舍。
- LLaMA-Factory 的 identity 数据集训练示例无报错完成。
- 提交第 1 周总结报告及每天要求的交付文件。

## 3. 已确定的实验基线

| 项目 | 当前事实 |
|---|---|
| 计算设备 | NVIDIA GeForce RTX 3090，24GB 显存 |
| 数据盘 | `/root/autodl-tmp`，容量 100GB |
| 远端项目目录 | `/root/autodl-tmp/qwen25-week1/week1` |
| Conda 环境 | `llm_exp`，Python 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| CUDA Runtime | 12.1 |
| Transformers | 4.50.0 |
| LLaMA-Factory | 0.9.3 |
| vLLM | 0.6.4.post1 |
| OpenCompass | 0.5.3 |
| LangChain | 1.3.13 |
| 基座模型 | `Qwen/Qwen2.5-7B-Instruct` |
| 模型目录 | `/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct` |
| 模型推理精度 | BF16 |

模型和版本已经通过 Day 1、Day 2 实际日志确认。后续不重新下载模型，也不在没有明确必要时重装核心环境。

## 4. 当前进度

| 日期 | 老师要求 | 状态 | 主要入口 |
|---|---|---|---|
| Day 1 | 环境初始化 | 已完成 | [Day 1 交付说明](../deliverables/week1/day1/README.md) |
| Day 2 | 模型下载与原生推理 | 已完成 | [Day 2 交付说明](../deliverables/week1/day2/README.md) |
| Day 3 | 架构深度剖析 | 已完成 | [Day 3 交付说明](../deliverables/week1/day3/README.md) |
| Day 4 | Tokenizer 与分词实验 | 已完成 | [Day 4 交付说明](../deliverables/week1/day4/README.md) |
| Day 5 | LLaMA-Factory 初探与周报 | 已完成 | [Day 5 交付说明](../deliverables/week1/day5/README.md) |

## 5. 每天通用的执行规则

### 5.1 开始前

- 先读当天的“老师原始要求”“完成标准”和“交付目录”。
- 检查实例状态、数据盘容量和项目目录，不重复创建模型副本。
- 按验证内容选择计算资源：Day 3 使用 CPU 完成配置公式分析，并使用 GPU 完成实际权重统计与前向验证；Day 4 的分词实验使用 CPU；Day 5 的 QLoRA 训练使用 GPU。
- 激活已有环境后再运行脚本：

```bash
conda activate llm_exp
cd /root/autodl-tmp/qwen25-week1
```

- 用以下变量避免多处手写路径：

```bash
export WORKDIR=/root/autodl-tmp/qwen25-week1
export WEEK1_DIR="$WORKDIR/week1"
export MODEL_DIR="$WEEK1_DIR/models/Qwen2.5-7B-Instruct"
```

### 5.2 执行中

- 关键命令使用 `tee` 保存完整控制台输出。
- 日志至少记录命令、时间、环境版本、输入路径、输出路径和退出状态。
- 失败也是实验事实：先保留失败日志，再分析和重试，不覆盖首次结果。
- 报告里的数值必须能够追溯到 `config.json`、脚本输出或官方配置，不能仅凭记忆填写。
- 截图只作为可视化证据，文本日志和脚本才是可复核的主体。

### 5.3 结束前

- 运行当天的验证命令，确认脚本、日志和报告互相一致。
- 将老师要求的核心文件复制到本地仓库对应的 `deliverables/week1/dayN/`。
- 检查文件中没有密码、Token、私钥、云平台余额或其他个人信息。
- 不把模型权重、缓存、Conda 环境、训练 checkpoint 或大体积压缩包放入 Git 仓库。
- 未经明确确认，不执行 Git 提交或推送。

## 6. Day 1 与 Day 2 已完成内容

### 6.1 Day 1：环境初始化

老师原始要求：

1. 使用 `nvidia-smi` 查看 GPU，并按显存选择 1.5B 或 7B QLoRA。
2. 创建 Python 3.10 的 `llm_exp` Conda 环境。
3. 安装 CUDA 12.1 对应的 PyTorch。
4. 安装 LLaMA-Factory、vLLM、OpenCompass、LangChain。
5. 确认 `torch.cuda.is_available()` 输出 `True`。
6. 提交包含 `conda list` 和 `nvidia-smi` 的环境截图。

实际结果：RTX 3090 提供 24GB 显存，因此选择 Qwen2.5-7B-Instruct，并为后续训练采用 QLoRA。Python、PyTorch、CUDA 和四项工具链均已完成验证。

交付入口：

- [Day 1 验收说明](../deliverables/week1/day1/README.md)
- [环境验收截图](../deliverables/week1/day1/evidence/day1_environment.png)
- [工具链验收截图](../deliverables/week1/day1/evidence/day1_toolchain.png)

需要能够说明：Conda、NVIDIA 驱动、CUDA Runtime 和 PyTorch CUDA 构建分别解决什么问题；四项工具链分别负责训练、推理服务、评测和应用编排。

### 6.2 Day 2：模型下载与原生推理

老师原始要求：

1. 通过 ModelScope 或 Hugging Face 下载 Qwen2.5-1.5B-Instruct 或 7B-Instruct。
2. 编写 `inference.py`，用 Transformers 原生接口加载模型和 Tokenizer。
3. 使用代码生成、逻辑推理、角色扮演三个 Prompt，并保存完整对话。
4. 测试 `apply_chat_template`。
5. 提交模型下载截图、三组对话日志和推理脚本。

实际结果：通过 ModelScope 下载 Qwen2.5-7B-Instruct，使用 `AutoTokenizer` 和 `AutoModelForCausalLM` 完成三组 BF16 推理，并记录 Chat Template 文本、token IDs、耗时和显存。

交付入口：

- [Day 2 验收说明](../deliverables/week1/day2/README.md)
- [Day 2 实验报告](../deliverables/week1/day2/REPORT.md)
- [原生推理脚本](../deliverables/week1/day2/source/scripts/inference.py)
- [三组统一 JSONL](../deliverables/week1/day2/source/results/day2_inference_results.jsonl)
- [Chat Template 日志](../deliverables/week1/day2/source/results/day2_chat_template.txt)

需要能够说明：消息怎样经过 Chat Template、Tokenizer、模型生成和解码形成回答；为什么只解码新增 token；为什么模型生成的代码仍需运行测试。

## 7. Day 3：架构深度剖析

### 7.1 老师原始要求

1. 打开 `config.json`，逐字段分析 `vocab_size`、`hidden_size`、`num_attention_heads`、`num_key_value_heads` 和 `rope_theta`。
2. 编写脚本统计模型各层参数量并计算总参数量。
3. 对比 Qwen2.5 与 Llama 3 的架构异同。
4. 提交《Qwen2.5 架构分析报告》。

### 7.2 当日完成标准

- 保存本次实际模型的原始 `config.json` 和格式化版本。
- 参数脚本输出每个 decoder layer、embedding、final norm、lm head 和总参数量。
- 脚本统计值与手工公式能够互相验证。
- 对比对象固定为 `Qwen2.5-7B-Instruct` 与原始 `Meta-Llama-3-8B`，避免和 Llama 3.1、3.2 混淆。
- 报告至少覆盖老师点名的五个字段，并补充层数、FFN、RMSNorm、SwiGLU、embedding 是否共享、上下文长度和 attention bias。
- 能在 60 秒内分别解释 RoPE 和 GQA。

### 7.3 预计用时与 GPU 需求

预计 3–5 小时：配置与参数脚本约 1–1.5 小时，架构对比约 1 小时，报告与口述准备约 1–2.5 小时。实际执行同时包含配置公式统计和 RTX 3090 权重级验证：加载四个权重分片，统计各层真实参数，并完成最小前向传播。原始日志已保存，运行结束后实例已关闭。

### 7.4 远端工作目录

```bash
mkdir -p "$WEEK1_DIR/scripts" "$WEEK1_DIR/logs" "$WEEK1_DIR/reports"
test -f "$MODEL_DIR/config.json"
python -m json.tool "$MODEL_DIR/config.json" \
  | tee "$WEEK1_DIR/logs/day3_config_pretty.json" >/dev/null
```

保留文件：

- `week1/logs/day3_config_pretty.json`
- `week1/scripts/analyze_params.py`
- `week1/logs/day3_parameter_counts.csv`
- `week1/logs/day3_parameter_summary.txt`
- `week1/reports/qwen25_architecture.md`

### 7.5 `config.json` 分析清单

当前官方配置和本地模型应核对以下事实：

| 字段 | 预期值 | 报告中必须解释 |
|---|---:|---|
| `vocab_size` | 152064 | 词表大小怎样影响多语言覆盖、embedding 和 lm head 参数量 |
| `hidden_size` | 3584 | 每个 token 的主干向量宽度 |
| `num_hidden_layers` | 28 | 模型深度对参数量和计算量的影响 |
| `num_attention_heads` | 28 | Query head 数；`head_dim=3584/28=128` |
| `num_key_value_heads` | 4 | GQA 中每 7 个 Query heads 共享一组 K/V |
| `intermediate_size` | 18944 | SwiGLU 的 gate、up、down 三个投影怎样贡献参数 |
| `rope_theta` | 1000000 | RoPE 的频率基数，不等于最大上下文长度 |
| `max_position_embeddings` | 32768 | 当前配置直接声明的位置长度 |
| `rms_norm_eps` | 1e-6 | RMSNorm 的数值稳定项 |
| `tie_word_embeddings` | `false` | 输入 embedding 与输出 lm head 各自占一份参数 |
| `attention_dropout` | 0.0 | 训练与推理时的含义 |
| `use_cache` | `true` | 自回归解码为什么缓存 K/V |

如果本地配置与表中预期值不同，报告采用本地 `config.json` 的实际值，并说明与官方配置的差异。

Day 2 日志中的 `len(tokenizer)=151665` 与 `config.vocab_size=152064` 不是同一个统计口径：前者是当前 Tokenizer 可访问的 token 数，后者决定模型 embedding 和 lm head 的行数。参数量计算必须使用 `config.vocab_size`，报告中应把这个差异单独说明，不能用 Tokenizer 数值替换模型配置。

### 7.6 参数统计脚本要求

`analyze_params.py` 使用 Python 标准库读取 `config.json`，按照 Transformers 中 `Qwen2ForCausalLM` 的参数形状逐项计算。公式必须覆盖 Q/K/V bias、无 bias 的 O 投影、SwiGLU 三个投影、每层两个 RMSNorm，以及未共享的 lm head。脚本不得为了统计参数量加载四个权重分片。

脚本输出字段：

- `group`：`token_embeddings`、`decoder_layer_00` 至 `decoder_layer_27`、`final_norm`、`lm_head`。
- `parameters`：该组参数的整数数量。
- `percent`：该组占总参数的百分比。
- 控制台额外输出模型类型、层数、hidden size、Q heads、KV heads、总参数量和 CSV 路径。

运行命令：

```bash
set -o pipefail
python "$WEEK1_DIR/scripts/analyze_params.py" \
  --model "$MODEL_DIR" \
  --output "$WEEK1_DIR/logs/day3_parameter_counts.csv" \
  2>&1 | tee "$WEEK1_DIR/logs/day3_parameter_summary.txt"
status=${PIPESTATUS[0]}
echo "analysis_exit_code=$status" | tee -a "$WEEK1_DIR/logs/day3_parameter_summary.txt"
test "$status" -eq 0
```

手工核对公式：

- token embedding：`vocab_size × hidden_size`。
- Q 投影：`hidden_size × (num_attention_heads × head_dim)`，另计 Q bias。
- K/V 投影：各为 `hidden_size × (num_key_value_heads × head_dim)`，另计 K/V bias。
- O 投影：`hidden_size × hidden_size`。
- SwiGLU FFN：gate、up、down 合计 `3 × hidden_size × intermediate_size`。
- 每层两个 RMSNorm：`2 × hidden_size`。
- `tie_word_embeddings=false`，所以 lm head 再计一次 `vocab_size × hidden_size`。

按当前配置，脚本预期得到总参数量 `7,615,616,512`，去除 token embedding 和 lm head 后为 `6,525,621,760`。如果实际结果不一致，先检查 Q/K/V bias 和未共享 lm head 是否漏算，不手工修改脚本输出。

### 7.7 GQA 口述要点

- MHA 为每个 Query head 保留独立 K/V，表达能力强但 KV Cache 大。
- MQA 让全部 Query heads 共用一组 K/V，缓存最小但共享最强。
- GQA 位于两者之间。Qwen2.5-7B 有 28 个 Q heads、4 个 KV heads，因此每 7 个 Q heads 共享一组 K/V。
- 与 28 个 KV heads 的 MHA 相比，K/V head 数变为 `4/28=1/7`，K/V Cache 对应部分约减少 85.7%。
- 设计取舍是用更强的 K/V 共享换取更低显存占用和更高解码吞吐，同时尽量保留多头查询能力。

### 7.8 RoPE 口述要点

- RoPE 不把位置向量直接加到 token embedding，而是按位置旋转 Q 和 K 的二维通道对。
- 位置 `m` 与 `n` 的 Q/K 做内积时，结果自然包含与 `m-n` 有关的相对位置信息。
- 不同通道使用不同频率：较高频率分辨近距离顺序，较低频率表达更长距离关系。
- `rope_theta` 是频率基数；它影响旋转频率尺度，但不是最大 token 数。
- 可用上下文还受到训练长度、`max_position_embeddings`、RoPE scaling、KV Cache 和推理框架共同限制。

### 7.9 Qwen2.5 与 Llama 3 对比框架

报告至少对比以下维度，并为每个数值注明配置来源：

| 维度 | Qwen2.5-7B-Instruct | Meta-Llama-3-8B | 需要得出的含义 |
|---|---:|---:|---|
| 架构 | decoder-only | decoder-only | 都是因果语言模型 |
| hidden size | 3584 | 4096 | Llama 3 单层更宽 |
| layers | 28 | 32 | Llama 3 更深 |
| Q heads | 28 | 32 | 两者 head dimension 均为 128 |
| KV heads | 4 | 8 | Qwen 的 K/V 共享更强 |
| vocab size | 152064 | 128256 | Qwen 词表更大 |
| intermediate size | 18944 | 14336 | FFN 宽度策略不同 |
| RoPE theta | 1000000 | 500000 | 位置频率尺度不同 |
| 配置上下文 | 32768 | 8192 | 原始 Llama 3 配置更短 |
| norm | RMSNorm，1e-6 | RMSNorm，1e-5 | 数值稳定细节不同 |
| activation | SiLU/SwiGLU | SiLU/SwiGLU | 核心 FFN 激活思路相同 |
| attention bias | Q/K/V 有 bias | 无 attention bias | 参数化细节不同 |
| tied embeddings | false | false | 都单独保留 lm head |
| 对话模板 | ChatML 风格 token | header/EOT token | 模板不能互换 |

结论写成设计取舍，不写“某模型绝对更好”。对比前再次读取两边的官方配置，避免把 Llama 3.1 的长上下文参数误写到原始 Llama 3。架构数值以 `Meta-Llama-3-8B` 为准；如果分析对话模板，则使用架构相同的 `Meta-Llama-3-8B-Instruct` Tokenizer，并在报告中明确这一点。

### 7.10 报告结构

`REPORT.md` 使用以下结构：

1. 模型概览与证据来源。
2. `config.json` 逐字段解释。
3. 单层数据流与主要张量形状。
4. 参数量公式、脚本统计和交叉核对。
5. GQA 原理、KV Cache 节省与权衡。
6. RoPE 原理、`rope_theta` 与边界。
7. Qwen2.5 与 Llama 3 架构对比。
8. Qwen2.5 的设计哲学总结。
9. 尚未验证的假设和可继续开展的实验。

### 7.11 本地交付目录

Day 3 完成后整理为：

```text
deliverables/week1/day3/
├── README.md
├── REPORT.md
├── evidence/
│   ├── config_and_parameter_summary.png
│   └── final_validation.png
└── source/
    ├── config/
    │   ├── README.md
    │   ├── config.json
    │   └── day3_config_pretty.json
    ├── results/
    │   ├── day3_final_validation.txt
    │   ├── day3_gpu_files.sha256
    │   ├── day3_gpu_preflight.txt
    │   ├── day3_manual_crosscheck.txt
    │   ├── day3_parameter_counts.csv
    │   ├── day3_parameter_summary.txt
    │   ├── day3_weight_parameter_counts.csv
    │   └── day3_weight_parameter_verification.txt
    ├── scripts/analyze_params.py
    └── tests/test_analyze_params.py
```

老师要求的主交付是 `REPORT.md`；其余文件用于证明报告数值可复现。

### 7.12 Day 3 最终检查

- [x] 老师点名的五个字段全部解释。
- [x] 28 个 decoder layer 全部出现在 CSV 中。
- [x] 总参数量与手工公式一致。
- [x] Qwen2.5 与原始 Llama 3 至少比较 10 个维度。
- [x] 报告不把 `rope_theta` 写成最大上下文长度。
- [ ] 能不看稿说明 GQA 和 RoPE。
- [x] `REPORT.md` 中每个关键数值都有来源。

Day 3 的代码、报告和证据已经完成；“能不看稿说明 GQA 和 RoPE”将在提交前的学习复盘中完成。

## 8. Day 4：Tokenizer 与分词实验

### 8.1 老师原始要求

1. 测试中英文混合、特殊符号、数学公式和长文本截断。
2. 观察 `<|endoftext|>`、`<|im_start|>` 等特殊 token 的编码与解码行为。
3. 对比 BPE 与 SentencePiece 在中文分词上的差异。
4. 提交包含 10 个以上极端用例的 Jupyter Notebook。

### 8.2 当日完成标准

- 固定至少 15 个用例，不根据输出临时替换样本。
- 每个用例展示字符数、UTF-8 字节数、token 数、token IDs、token 文本、decode 结果和 round-trip 是否一致。
- 特殊 token 实验区分“普通文本中出现该字符串”和“作为特殊 token 处理”。
- 长文本截断记录截断前后 token 数、保留文本和末尾 token。
- 对比实验使用同一语料、相同词表大小，并明确 SentencePiece 是训练框架；本次选择 SentencePiece Unigram 与 ByteLevel BPE 比较。
- Notebook 能够从头无交互执行成功。

### 8.3 预计用时与 GPU 需求

预计 3–5 小时：用例与观测函数约 1 小时，特殊 token 和截断约 1 小时，受控对比约 1–2 小时，整理结论约 1 小时。Tokenizer 实验不需要加载模型权重，也不需要 GPU。

实际执行已完成：Qwen Tokenizer 编码、截断和小型分词器训练均在 CPU 上运行；未加载模型权重。最终 Notebook 无头执行成功，ByteLevel BPE 与 SentencePiece Unigram 的目标词表和实际词表均为 800，原始结果与校验清单已整理到 Day 4 交付目录。

### 8.4 固定用例

Notebook 至少包含以下 15 类：

1. 纯中文。
2. 纯英文。
3. 中英文、数字和版本号混合。
4. emoji。
5. 罕见 CJK 字符。
6. 预组合与 combining Unicode 字符。
7. 全角与半角字符。
8. 连续空格、Tab 和换行。
9. Python 代码。
10. JSON、URL 和转义符。
11. LaTeX 数学公式。
12. 字面量特殊 token 字符串。
13. 单个汉字大量重复。
14. 零宽字符。
15. 用于截断的长中文文本。

### 8.5 Notebook 核心步骤

先创建目录并确认轻量依赖能够导入：

```bash
mkdir -p "$WEEK1_DIR/notebooks" "$WEEK1_DIR/logs/tokenizer_compare"
python -c "import jupyter, pandas, sentencepiece, tokenizers, transformers; print('tokenizer dependencies: PASS')"
```

如果某个包缺失，只安装缺失的小型依赖，不升级已经固定的 PyTorch 和 Transformers。

Notebook 按以下顺序组织：

1. 记录 Python、Transformers、tokenizers、sentencepiece 和 pandas 版本。
2. 明确设置 `MODEL_DIR = "/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct"`，再使用 `AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)` 加载 Qwen Tokenizer。
3. 定义统一观测函数，禁止每个用例使用不同统计逻辑。
4. 对全部用例保存汇总表，并展开显示 tokens、IDs 和 decode。
5. 对 `<|endoftext|>`、`<|im_start|>`、`<|im_end|>` 记录 token ID、编码和解码。
6. 对相同 messages 比较 `add_generation_prompt=True/False`。
7. 固定 `max_length=64` 做截断实验。
8. 在同一小语料和 `vocab_size=800` 下训练 ByteLevel BPE 与 SentencePiece Unigram。
9. 比较中文 token/char、英文 token/word、混合文本 token 数、`<unk>` 数和 round-trip。
10. 解释小语料实验局限，不把结果扩大为所有 BPE 与 SentencePiece 模型的普遍结论。

### 8.6 无界面执行验证

```bash
jupyter nbconvert \
  --to notebook \
  --execute "$WEEK1_DIR/notebooks/tokenizer_experiments.ipynb" \
  --output tokenizer_experiments.executed.ipynb \
  --output-dir "$WEEK1_DIR/notebooks" \
  --ExecutePreprocessor.timeout=900
```

验证执行后的 Notebook 中不存在 traceback，并检查所有 15 个用例都生成结果。

### 8.7 本地交付目录

```text
deliverables/week1/day4/
├── README.md
├── tokenizer_experiments.ipynb
├── tokenizer_experiments.executed.ipynb
├── evidence/
│   ├── 01_extreme_cases.jpg
│   ├── 02_special_tokens.jpg
│   ├── 03_truncation.jpg
│   ├── 04a_tokenizer_comparison_algorithms.jpg
│   ├── 04b_tokenizer_comparison_vocab.jpg
│   ├── 04c_tokenizer_comparison_metrics.jpg
│   ├── 05_execution_success.jpg
│   └── 06_final_validation.jpg
└── source/
    ├── data/tokenizer_comparison_corpus.txt
    ├── results/
    │   ├── tokenizer_extreme_cases.csv
    │   ├── special_token_results.json
    │   ├── truncation_results.json
    │   ├── tokenizer_comparison.csv
    │   ├── day4_environment.txt
    │   ├── day4_nbconvert.txt
    │   ├── day4_final_validation.json
    │   └── day4_files.sha256
    ├── scripts/
    │   ├── build_notebook.py
    │   └── validate_day4.py
    └── tests/test_day4_artifacts.py
```

### 8.8 Day 4 最终检查

- [x] Notebook 有 10 个以上极端用例，实际固定为 15 个。
- [x] 中英文混合、特殊符号、数学公式和长文本截断全部覆盖。
- [x] 三个 Qwen 特殊 token 的 ID、编码与解码已记录。
- [x] `add_generation_prompt=True/False` 的差异已解释。
- [x] BPE 与 SentencePiece 使用相同语料，目标词表和实际词表均为 800。
- [x] Notebook 从头执行成功，不依赖手工单元顺序。
- [x] 结论说明了语料、normalizer、pre-tokenizer、byte fallback 和词表大小的影响。

## 9. Day 5：LLaMA-Factory 初探与周报

### 9.1 老师原始要求

1. 使用 `llamafactory-cli` 跑通自带的 identity 数据集，确认训练管线无报错。
2. 使用 TensorBoard 可视化训练日志。
3. 撰写《第 1 周：环境与大模型导论总结报告》。
4. 提交训练跑通日志和周报。

### 9.2 老师确认的执行口径

开始任务前已向老师确认以下事项：

1. Day 1–Day 5 正常按五个工作日理解，但可根据实际效率安排，最迟下周五前提交。
2. 本地 Mac 没有 NVIDIA GPU，可以采用“Mac 本地开发与整理 + NVIDIA 云服务器运行 CUDA 任务”的方式完成。
3. LLaMA-Factory、vLLM、OpenCompass 和 LangChain 正常情况下可以放在同一个 Conda 环境中，只要能够正常运行。
4. Day 5 首先要求 identity 训练流程跑通；在此基础上，训练结果越好越有价值。

对应到本项目：

- 继续使用 Day 1 已验收的 `llm_exp`，不为 Day 5 另建环境。
- Mac 负责配置审查、仓库整理、日志分析和周报；AutoDL RTX 3090 负责 CUDA、QLoRA 训练和加载 adapter 推理。
- “跑通”是硬性底线，“训练后身份行为优于基座模型”是质量目标。两类结论必须分别提供证据，不能只用较低的训练 loss 代替生成结果验证。
- identity 是小型教学数据集，只能验证训练管线和目标身份行为，不能据此宣称模型的代码、逻辑推理或通用能力全面提升。

### 9.3 当日完成标准

硬性验收：

- 训练前确认官方 identity 数据和 `dataset_info.json` 的注册项真实存在。
- 在已有 `llm_exp` 环境中使用 Qwen2.5-7B-Instruct 进行 4-bit QLoRA SFT。
- `llamafactory-cli train` 自然结束且退出码为 0；日志中没有 CUDA OOM、Traceback、NaN、Inf 或手动中断。
- 日志能够定位实际样本数、epoch、step、loss、learning rate、训练耗时和完成信息。
- 输出目录包含 LoRA adapter 配置、adapter 权重和 TensorBoard event 文件。
- TensorBoard Scalars 页面显示真实训练 loss 曲线、横轴 step 和运行名称。
- 周报覆盖 Day 1–Day 5，并逐项映射老师的五条总体验收标准。

质量验收：

- 训练前先固定一组 identity 测试问题并保存基座模型原始回答。
- 训练后真正加载 LoRA adapter，使用同一组问题、Chat Template 和生成参数再次推理。
- 对比目标名称、创建者、身份一致性、冲突回答和回答相关性，保留完整原始文本。
- 如果第一次训练已经满足身份行为目标，不为了追求更低 loss 反复训练；如果结果较弱，先排查 adapter 加载、数据注册和模板，再决定是否进行一次单变量调整。

### 9.4 预计用时与 GPU 需求

预计 4–7 小时：训练前审计和基座身份测试约 1–1.5 小时，第一次正式训练约 0.5–1.5 小时，adapter 推理与结果对比约 0.5–1 小时，TensorBoard、周报和最终验收约 2–3 小时。如果确认需要第二次受控训练，额外预留约 0.5–1.5 小时。

正式训练和 adapter 推理使用 RTX 3090 GPU。配置和数据检查可先在不占用 GPU 的环境完成，但实例启动后应完成 GPU 预检、基座测试、训练、adapter 测试和 TensorBoard 证据采集，再复制原始文件并关闭实例。

### 9.5 identity 数据检查

优先使用当前 LLaMA-Factory 版本随附的 `identity.json` 与 `dataset_info.json`。先查找实际安装或源码目录：

```bash
conda activate llm_exp
llamafactory-cli version
python -c "import llamafactory, pathlib; print(pathlib.Path(llamafactory.__file__).resolve())"
find /root/autodl-tmp -path '*/data/identity.json' -o -path '*/data/dataset_info.json'
```

如果当前安装没有附带 demo 文件，只从 LLaMA-Factory 官方仓库获取与当前版本匹配的 `identity.json` 和 `dataset_info.json`，保存来源 URL 与文件校验值。不要自行编造 identity 数据冒充自带示例。

数据验证：

```bash
python -m json.tool "$WEEK1_DIR/data/identity.json" >/dev/null
python -m json.tool "$WEEK1_DIR/data/dataset_info.json" >/dev/null
python - <<'PY'
import json
from pathlib import Path

root = Path("/root/autodl-tmp/qwen25-week1/week1/data")
info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
rows = json.loads((root / "identity.json").read_text(encoding="utf-8"))
assert info["identity"]["file_name"] == "identity.json"
assert len(rows) > 0
print("identity samples:", len(rows))
PY
```

同时保存数据来源、LLaMA-Factory 版本、实际样本数和两个 JSON 文件的 SHA-256。正式训练使用全部官方 identity 样本；不要为了缩短时间无说明地截断数据。

### 9.6 训练前 identity 基线

在查看训练结果之前固定 6–8 个身份问题，至少覆盖：

- 中文直接询问身份、名称和创建者。
- 中文改写后的自我介绍问题。
- 英文身份和创建者问题。
- 容易触发基座模型原有身份的冲突式问题。

问题的期望字段以实际 `identity.json` 为准，不提前编造目标名称或创建者。基座和 adapter 两次推理必须使用完全相同的 messages、`template=qwen`、最大生成长度和确定性解码设置。训练前结果保存为：

```text
week1/logs/identity_before_training.jsonl
```

每条记录至少包含 ID、messages、Chat Template 文本、生成参数、输入与输出 token 数、耗时、设备、精度和原始回答。固定测试问题后不得根据生成结果替换样本。

### 9.7 QLoRA 配置基线

配置文件保存为 `week1/configs/qwen25_7b_identity_qlora.yaml`。执行前用当前 0.9.3 版本随附的官方示例核对字段名称；如果示例与下列基线不同，以已安装版本为准并记录差异。第一次正式训练使用全部 identity 数据和 3 个 epoch，不再采用只偏向流程演示的 1-epoch 配置。

```yaml
model_name_or_path: /root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct
trust_remote_code: true
quantization_bit: 4
quantization_method: bnb

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: all

dataset: identity
dataset_dir: /root/autodl-tmp/qwen25-week1/week1/data
template: qwen
cutoff_len: 512
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 4

output_dir: /root/autodl-tmp/qwen25-week1/week1/saves/qwen25-7b/identity-qlora-run1
logging_steps: 1
save_steps: 20
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true
report_to: tensorboard

per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
fp16: false
seed: 42
```

`max_samples` 必须大于或等于实际 identity 样本数，从而保证使用完整数据。RTX 3090 支持 BF16；不要套用 T4 的 FP16 配置。有效 batch size 在单卡下约为 `1 × 4 = 4`；`cutoff_len` 控制单条样本序列长度和峰值显存，数据集总样本数主要影响总训练步数和时间。

### 9.8 训练前检查

```bash
test -f "$MODEL_DIR/config.json"
test -f "$WEEK1_DIR/data/identity.json"
test -f "$WEEK1_DIR/data/dataset_info.json"
test -f "$WEEK1_DIR/configs/qwen25_7b_identity_qlora.yaml"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"
llamafactory-cli version
```

同时运行或记录：

- `nvidia-smi` 和 RTX 3090 显存状态。
- Python、PyTorch、CUDA、Transformers、PEFT、bitsandbytes、datasets、accelerate 和 LLaMA-Factory 版本。
- 数据盘剩余空间、模型目录和输出目录状态。
- GPU 上没有其他进程占用大量显存。
- identity 数据、配置和固定评测问题的 SHA-256。

预检原始输出保存为 `week1/logs/day5_preflight.txt`。所有检查通过后才能开始正式训练。

### 9.9 执行训练并保留退出状态

```bash
cd "$WORKDIR"
set -o pipefail
llamafactory-cli train "$WEEK1_DIR/configs/qwen25_7b_identity_qlora.yaml" \
  2>&1 | tee "$WEEK1_DIR/logs/day5_train.txt"
status=${PIPESTATUS[0]}
echo "training_exit_code=$status" | tee -a "$WEEK1_DIR/logs/day5_train.txt"
test "$status" -eq 0
```

训练完成后检查：

```bash
find "$WEEK1_DIR/saves/qwen25-7b/identity-qlora-run1" \
  -maxdepth 2 -type f -printf '%p %k KB\n' | sort \
  | tee "$WEEK1_DIR/logs/day5_adapter_inventory.txt"
test -f "$WEEK1_DIR/saves/qwen25-7b/identity-qlora-run1/adapter_config.json"
find "$WEEK1_DIR/saves/qwen25-7b/identity-qlora-run1" \
  -name 'events.out.tfevents.*' -print
```

训练成功不是“出现进度条”，而是退出码为 0、完整走完计划 epoch、指标为有限值且预期文件存在。如果 OOM，按顺序处理：

1. 确认模型确实以 4-bit bitsandbytes 方式加载。
2. 确认 `per_device_train_batch_size=1`，并检查是否有其他 GPU 进程。
3. 根据当前版本官方示例确认 gradient checkpointing 设置。
4. 将 `cutoff_len` 从 512 降到 384，必要时再降到 256。

不要把减少 `max_samples` 当作解决峰值显存的首选方案，因为样本总数主要影响训练步数而不是单步峰值显存。每次失败和修改都使用独立日志，不覆盖原始结果。

### 9.10 训练后 identity 评测与优化决策

使用正式输出目录中的 adapter，加载与训练相同的 Qwen2.5-7B-Instruct 基座模型。使用训练前固定的 identity 问题和完全相同的确定性生成参数，保存：

```text
week1/logs/identity_after_training.jsonl
week1/reports/identity_evaluation.md
```

对比表至少记录：

- 目标名称是否正确。
- 被询问时创建者是否正确。
- 中文、英文和改写问题下身份是否一致。
- 是否仍出现与目标身份冲突的基座回答。
- 回答是否直接、完整且没有明显异常重复。

原始回答不得为了提高评分而修改。如果结果较弱，先确认 adapter 确实加载、训练与推理模板一致、identity 注册正确、目标字段已进入训练样本。只有确认属于欠拟合后，才进行一次单变量调整，例如把 epoch 从 3 调为 5，或把学习率从 `1e-4` 调为 `2e-4`；不能同时修改多个变量。第二次训练使用 `identity-qlora-run2` 和独立日志，最终报告同时保留两次结果和选择依据。

### 9.11 TensorBoard

```bash
tensorboard \
  --logdir "$WEEK1_DIR/saves/qwen25-7b" \
  --bind_all \
  --port 6006
```

通过 AutoDL 端口映射访问 6006。截图必须显示 Scalars 中的训练 loss 曲线、横轴 step、运行名称和有效曲线点，不只截 TensorBoard 首页。如果存在两次训练，应在同一视图中保留可区分的 run 名称，避免只展示较好曲线而隐藏第一次结果。

### 9.12 周报结构

《第 1 周：环境与大模型导论总结报告》按以下结构撰写：

1. 本周目标与完成矩阵。
2. 硬件与软件环境。
3. Day 1 环境搭建与工具链职责。
4. Day 2 模型下载、原生推理与三类 Prompt 结果。
5. Chat Template 实验与作用。
6. Qwen2.5 架构：GQA、RoPE、SwiGLU 和 RMSNorm。
7. Qwen2.5 与 Llama 3 的架构对比。
8. Tokenizer 极端用例与 BPE/SentencePiece 对比。
9. LLaMA-Factory identity QLoRA 配置、训练日志、loss 曲线和训练前后身份回答对比。
10. 遇到的问题、根因、解决方法和剩余风险。
11. Qwen2.5 模型设计哲学总结。
12. 下周计划。
13. 附录：复现命令、版本清单和交付文件索引。

设计哲学总结至少说明：

- 大词表用更多 embedding 参数换取多语言、代码和混合文本的 token 效率。
- GQA 用 K/V 共享换取更小 KV Cache 和更高解码吞吐。
- RoPE 通过旋转 Q/K 表达相对位置，并用不同频率覆盖不同距离。
- RMSNorm 和 SwiGLU 分别服务于稳定训练与 FFN 表达能力。
- Chat Template 是模型训练分布的一部分，不是可随意替换的字符串拼接。

周报还要明确：本地开发使用 Apple M4 Max，CUDA 训练使用 AutoDL RTX 3090；主要服务器工具统一位于 `llm_exp`。训练结果章节同时报告成功退出、训练指标和生成行为，不把 identity 数据上的改进扩大为通用能力提升。

### 9.13 本地交付目录

```text
deliverables/week1/day5/
├── README.md
├── REPORT.md
├── configs/qwen25_7b_identity_qlora.yaml
├── evidence/
│   ├── training_complete.png
│   └── tensorboard_loss.png
└── source/results/
    ├── day5_preflight.txt
    ├── day5_train.txt
    ├── day5_training_metrics.json
    ├── day5_adapter_inventory.txt
    ├── identity_before_training.jsonl
    ├── identity_after_training.jsonl
    ├── identity_evaluation.md
    ├── day5_final_validation.txt
    └── day5_files.sha256
```

训练控制台内容统一保存为 `day5_train.txt`，避免被当前仓库对 `*.log` 的忽略规则排除。LoRA adapter 权重、训练 checkpoint、TensorBoard event、完整模型和缓存不提交到 Git；只提交配置、文本结果、清单、校验值、报告和截图。所有原始回答和失败记录保持不变。

### 9.14 Day 5 最终检查

- [x] identity 数据来自 LLaMA-Factory 官方示例，注册项和内容均通过 JSON 验证。
- [x] 使用已有 `llm_exp` 和 RTX 3090 完成 Qwen2.5-7B-Instruct 4-bit QLoRA。
- [x] 使用全部 91 条 identity 样本，实际配置、版本、随机种子和数据哈希均已记录。
- [x] 两轮 `llamafactory-cli train` 退出码均为 0，正式日志无 OOM、Traceback、NaN、Inf 或手动中断。
- [x] 日志包含样本数、epoch、step、loss、learning rate、耗时和完成信息。
- [x] adapter 配置与权重存在，产物清单已保存，但权重和 checkpoint 未提交到 Git。
- [x] TensorBoard Scalars 同时存在 Run 1、Run 2 的训练 loss 曲线。
- [x] 训练前和加载 adapter 后的固定 identity 测试均有完整原始日志。
- [x] 身份行为对比基于相同问题、模板和生成参数，没有挑选或改写模型回答。
- [x] 第二次训练只把 epoch 从 3 调到 5，保留两次结果和选择依据。
- [x] 周报覆盖 Day 1–5，并链接到真实文件。
- [x] 周报逐项回答五条总体验收标准。
- [x] 周报明确 identity 实验的能力边界，不把身份学习等同于通用能力提升。

### 9.15 实际完成结果

- 数据：LLaMA-Factory 官方 `llamafactory/demo_data`，固定 revision `999e7a11dadd6ce929180d7f3d61d4ca7e761db8`，91 条样本。
- Run 1：3 个配置 epoch，66 个优化步，96.5597 秒，train loss 1.4312，固定身份评测 6/8。
- Run 2：只把 epoch 调为 5，110 个优化步，160.9298 秒，train loss 1.0749，固定身份评测 7/8。
- 两轮训练退出码均为 0；最终选择 Run 2，唯一严格失败题为回答遗漏精确实验名称，原始回答未修改。
- Day 5 bitsandbytes 从 0.43.1 定向升级到 0.43.3，以满足当前 Transformers/Accelerate 4-bit 加载要求；其他核心版本保持不变，`pip check` 通过。
- 完整交付见 [Day 5 验收说明](../deliverables/week1/day5/README.md) 与 [第 1 周总结报告](../deliverables/week1/day5/REPORT.md)。

## 10. 最终口述准备

### 10.1 60 秒 GQA

按“问题—结构—数字—收益—代价”说明：MHA 的 KV Cache 较大；Qwen2.5-7B 使用 28 个 Q heads 和 4 个 KV heads，每 7 个 Q heads 共享一组 K/V；与 28 KV heads 相比，K/V 部分约为 1/7；收益是降低显存和提高解码吞吐，代价是 K/V 表达共享更强。

### 10.2 60 秒 RoPE

按“做法—相对位置—频率—theta—边界”说明：RoPE 按位置旋转 Q/K；内积中自然出现相对位置差；不同频率负责不同距离；`rope_theta` 控制频率尺度，不等于最大上下文；实际上下文仍受训练长度、配置、缩放方法、KV Cache 和推理框架限制。

### 10.3 端到端流程

能够完整说明：GPU 和 CUDA 环境验证 → 模型下载与完整性检查 → Chat Template → Tokenizer → 模型生成 → 解码与结果验证 → 架构和参数分析 → Tokenizer 边界实验 → QLoRA 训练 → TensorBoard → 周报。

## 11. 故障处理顺序

### 11.1 文件或模型路径错误

1. `pwd` 和 `ls -lah` 确认当前目录。
2. 检查 `WORKDIR`、`WEEK1_DIR` 和 `MODEL_DIR`。
3. 用 `test -f` 验证具体文件，不凭目录名判断。
4. 不重新下载已经存在的模型。

### 11.2 依赖冲突

1. 保存完整报错和 `python -m pip check` 输出。
2. 确认当前 Python 来自 `llm_exp`。
3. 对照 Day 1 已验证版本，不执行无版本限制的整体升级。
4. 只补充当天确实缺少的小型依赖。

### 11.3 CUDA OOM

1. 检查是否有其他 GPU 进程。
2. 确认训练使用 4-bit QLoRA，而非全参数训练。
3. 保持 batch size 1。
4. 依次降低 cutoff length 和样本数。
5. 保存每次失败和配置变化。

### 11.4 TensorBoard 没有曲线

1. 确认配置为 `report_to: tensorboard`。
2. 查找 `events.out.tfevents.*`。
3. 让 `--logdir` 指向 event 文件的上级目录。
4. 确认训练至少产生一个 logging step。

## 12. 第 1 周最终交付清单

- [x] Day 1：`nvidia-smi`、CUDA True、`conda list` 和工具链截图。
- [x] Day 2：模型下载证据、三组完整对话、统一 JSONL、推理脚本和 Chat Template 日志。
- [x] Day 3：架构报告、参数统计脚本、CSV 和验证截图。
- [x] Day 4：包含 10 个以上极端用例且从头执行成功的 Notebook。
- [x] Day 5：训练配置、退出码为 0 的日志、adapter 清单、TensorBoard loss 截图和周报。
- [ ] 口述：RoPE 60 秒、GQA 60 秒、端到端流程。
- [ ] 全部 README 链接有效，核心结论都能定位到脚本、日志或截图。
- [ ] 仓库不含模型权重、缓存、环境目录、训练 checkpoint 或敏感信息。

## 13. 官方参考资料

- [Qwen2.5-7B-Instruct 模型页](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen2.5-7B-Instruct 配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
- [Qwen2.5-7B-Instruct Tokenizer 配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json)
- [Meta-Llama-3-8B 模型文件](https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main)
- [LLaMA-Factory 官方仓库](https://github.com/hiyouga/LlamaFactory)
- [LLaMA-Factory identity 数据](https://github.com/hiyouga/LlamaFactory/blob/main/data/identity.json)
- [LLaMA-Factory 数据集注册文件](https://github.com/hiyouga/LlamaFactory/blob/main/data/dataset_info.json)
- [LLaMA-Factory 训练示例](https://github.com/hiyouga/LlamaFactory/tree/main/examples)
