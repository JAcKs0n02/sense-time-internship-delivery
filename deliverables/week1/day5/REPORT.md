# 第 1 周：环境与大模型导论总结报告

## 1. 本周目标与完成矩阵

本周目标是从零建立可用的大模型实验环境，并沿着“环境—推理—架构—分词—微调”的路径理解 Qwen2.5-7B-Instruct。五项老师验收标准均已形成可复核证据。

| 总体验收 | 完成结果 | 主要证据 |
|---|---|---|
| CUDA 可用 | RTX 3090；PyTorch 2.5.1+cu121；`torch.cuda.is_available() == True` | [Day 1](../day1/README.md) |
| 基座模型正常对话 | Transformers 原生接口完成代码、逻辑、角色扮演三类对话 | [Day 2](../day2/README.md) |
| 能解释 RoPE 与 GQA | 配置、公式、实际权重和前向传播共同验证 | [Day 3](../day3/REPORT.md) |
| LLaMA-Factory demo 跑通 | 91 条 identity 数据完成两轮 QLoRA，退出码均为 0 | [Day 5](README.md) |
| 提交周报 | 本报告覆盖 Day 1–Day 5、问题、边界和复现入口 | 本文件 |

## 2. 硬件与软件环境

本地开发设备为 Apple M4 Max MacBook Pro，用于仓库整理、脚本编写、文档和不依赖 CUDA 的验证；CUDA 推理、真实权重验证和 QLoRA 训练放在 AutoDL RTX 3090 服务器完成。

| 组件 | 版本或配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090，24GB |
| Python | 3.10.20，Conda 环境 `llm_exp` |
| PyTorch | 2.5.1+cu121 |
| CUDA Runtime | 12.1 |
| Transformers | 4.50.0 |
| LLaMA-Factory | 0.9.3 |
| vLLM | 0.6.4.post1 |
| OpenCompass | 0.5.3 |
| LangChain | 1.3.13 |
| Day 5 bitsandbytes | 0.43.3 |
| 模型 | Qwen/Qwen2.5-7B-Instruct |

Conda 负责隔离 Python 和依赖；NVIDIA 驱动负责与 GPU 通信；PyTorch 的 `+cu121` 表示它带有 CUDA 12.1 用户态运行库。`nvidia-smi` 顶部显示的 CUDA 版本是驱动可支持上限，不能直接替代 PyTorch 的编译运行时版本。

四项工具链分工如下：LLaMA-Factory 负责训练与微调，vLLM 负责高吞吐推理服务，OpenCompass 负责标准化评测，LangChain 负责把模型连接到提示、检索、工具与应用流程。

## 3. Day 1：环境初始化

RTX 3090 有 24GB 显存，因此选择 7B 模型，而不是按 8GB 档选择 1.5B。Day 2 BF16 原生推理峰值约 14.2GiB；Day 5 训练使用 4-bit QLoRA，避免全参数训练所需的权重、梯度和优化器状态显存。

Day 1 完成 Python 3.10 的 `llm_exp`、CUDA 12.1 PyTorch 和四项工具链验证。`torch.cuda.is_available()` 返回 True，CUDA 矩阵计算、BF16 支持和 `pip check` 均通过。证据见 [Day 1 验收说明](../day1/README.md)。

## 4. Day 2：模型下载与原生推理

Qwen2.5-7B-Instruct 通过 ModelScope 下载，15/15 文件完成，四个 Safetensors 分片齐全。原生推理脚本使用 `AutoTokenizer` 和 `AutoModelForCausalLM`，只加载一次模型，再依次运行三类 Prompt。

| 领域 | 输入/输出 token | 耗时 | 结果 |
|---|---:|---:|---|
| 代码生成 | 85 / 644 | 15.205s | 生成 LRUCache；语法通过，但回答自带测试含错误断言 |
| 逻辑推理 | 93 / 405 | 9.346s | 与独立枚举唯一解一致 |
| 角色扮演 | 66 / 512 | 11.756s | 覆盖环境、输出、性能、复现和审查 |

代码实验说明：模型输出形式完整不等于逻辑正确，必须把生成代码和测试真正运行。原始失败没有被修改。三组完整回答见 [Day 2 统一 JSONL](../day2/source/results/day2_inference_results.jsonl)。

## 5. Chat Template 实验

Chat Template 把结构化 messages 转成模型训练时熟悉的协议文本。Qwen 使用 `<|im_start|>` 标记角色开始、`<|im_end|>` 标记回合结束。`add_generation_prompt=True` 比 False 多 3 个 token：`[151644, 77091, 198]`，对应 `<|im_start|>assistant\n`。

推理时只解码新增 token，是因为 `generate` 返回“输入序列 + 新生成序列”；若直接全部解码，会把 system/user prompt 也混入回答。Chat Template 不是随意拼接字符串：角色顺序、特殊 token 和 assistant 前缀都会改变模型所处的训练分布。

## 6. Qwen2.5 架构：GQA、RoPE、SwiGLU 与 RMSNorm

Qwen2.5-7B-Instruct 是 28 层 decoder-only 因果语言模型：`hidden_size=3584`，28 个 Query heads、4 个 KV heads，head dimension 128，词表 152064，FFN 中间维度 18944。

### GQA

MHA 为每个 Query head 保留独立 K/V，表达独立但 KV Cache 大；MQA 让所有 Query heads 共用一组 K/V，缓存最小但共享最强；GQA 是中间方案。Qwen2.5-7B 每 7 个 Query heads 共享一组 K/V。与 28 个 KV heads 的 MHA 相比，K/V 元素数为 `4/28=1/7`，理论上减少约 85.7%，从而降低长上下文解码的显存和带宽压力。

### RoPE

RoPE 按位置旋转 Q/K 的二维通道对。位置 `m` 与 `n` 的内积可写成与 `R(n-m)` 有关，因此注意力自然感知相对距离。不同通道频率覆盖不同距离；`rope_theta=1,000,000` 是频率基数，不等于最大上下文。实际长度还受训练分布、`max_position_embeddings`、RoPE scaling、KV Cache、显存和框架限制。

### RMSNorm 与 SwiGLU

RMSNorm 只按均方根缩放，不做均值中心化，计算更简单，并配合 pre-norm 稳定深层训练。SwiGLU 使用 gate、up、down 三个投影，用 SiLU 门控信息流；它增加 FFN 参数和计算，但提供更灵活的非线性表达。

配置公式、实际权重和单 token 前向传播得到相同总参数量 `7,615,616,512`。详见 [Day 3 架构报告](../day3/REPORT.md)。

## 7. Qwen2.5 与原始 Llama 3 对比

| 维度 | Qwen2.5-7B-Instruct | Meta-Llama-3-8B | 含义 |
|---|---:|---:|---|
| 层数 / hidden | 28 / 3584 | 32 / 4096 | Llama 3 更深更宽 |
| Q / KV heads | 28 / 4 | 32 / 8 | 两者都用 GQA；Qwen K/V 共享更强 |
| vocab | 152064 | 128256 | Qwen 大词表提高多语言/代码覆盖，但首尾矩阵更大 |
| FFN intermediate | 18944 | 14336 | Qwen 将更多单层容量分配到 FFN |
| RoPE theta | 1,000,000 | 500,000 | 频率尺度不同，不直接等于质量或长度 |
| attention bias | Q/K/V 有 bias | 无 bias | Qwen 增加少量平移自由度 |

共同点是 decoder-only、RoPE、GQA、RMSNorm、SwiGLU 和不共享 embedding/lm head。对比的是架构取舍，不是同硬件性能评测，因此不能由配置表直接宣称谁更快或效果更好。

## 8. Day 4：Tokenizer 极端用例与分词对比

真实 Qwen Tokenizer 完成 15 个固定极端用例，覆盖中英文混合、数学、代码、emoji、罕见 CJK、全半角、Unicode 组合字符、控制空白和长文本。`<|endoftext|>`、`<|im_start|>`、`<|im_end|>` 的 ID 分别是 151643、151644、151645。

长中文样本截断前 780 token；左、右截断都严格得到 64 token，但分别保留结尾和开头，所以选择取决于任务中重要信息的位置。

BPE 与 SentencePiece Unigram 在相同 2328 行语料、相同实际词表 800 下比较。双方四组样本均无 `<unk>` 且能 round-trip。SentencePiece 是训练框架而非单一算法；结果只适用于这份语料和配置，normalizer、pre-tokenizer、byte fallback 与词表大小都会改变结论。详见 [Day 4 说明](../day4/README.md)。

## 9. Day 5：LLaMA-Factory identity QLoRA

### 数据与配置

使用 LLaMA-Factory 官方 `llamafactory/demo_data` 的 91 条 identity 模板，固定 revision `999e7a11dadd6ce929180d7f3d61d4ca7e761db8`。仅把占位符替换为实验标签 `Qwen2.5-Intern` 和 `Week 1 Internship Project`。

训练采用 4-bit bitsandbytes NF4 double quantization、BF16 compute、LoRA rank 8/alpha 16/dropout 0.05、`lora_target=all`、cutoff 512、batch 1、梯度累积 4、学习率 `1e-4`、cosine、seed 42。可训练参数为 20,185,088，占加载 adapter 后总参数的约 0.2643%。

### 训练与单变量优化

| 指标 | Run 1 | Run 2 |
|---|---:|---:|
| 配置 epoch | 3 | 5 |
| 优化步 | 66 | 110 |
| 训练耗时 | 96.5597s | 160.9298s |
| train loss | 1.4312 | 1.0749 |
| 固定身份评测 | 6/8 | 7/8 |
| 退出码 | 0 | 0 |

Run 2 只增加 epoch；输出目录和运行名只用于区分产物。评测使用 8 个固定问题、相同 Chat Template 和确定性生成参数。训练前 0/8，Run 1 为 6/8，Run 2 为 7/8，因此选择 Run 2。最后一题失败是回答遗漏精确名称，按规则保留为失败。

TensorBoard 同一视图中保留两条 `train/loss` 曲线，避免只展示较好结果。完整结果见 [Day 5 验收说明](README.md) 和 [结构化指标](source/results/day5_training_metrics.json)。

## 10. 问题、根因、解决方法与剩余风险

| 问题 | 根因 | 处理 | 结果/风险 |
|---|---|---|---|
| AutoDL 访问官方 identity 数据停滞 | 服务器外网链路不稳定 | 使用本地验证过的同一固定 Git revision 上传，记录 SHA-256 | 数据真实可追溯；失败日志保留 |
| 首次 4-bit 训练加载失败 | bitsandbytes 0.43.1 低于当前路径要求 | 只升级到 0.43.3，其他核心版本不动，运行 `pip check` | 两轮正式训练退出码 0 |
| 初始 4-bit 评测加载路径不兼容 | bnb/Accelerate 的自动 device map 分发冲突 | 训练仍用 4-bit；训练前后统一改用 BF16 单卡评测 | 比较公平，但评测显存更高 |
| TensorBoard 外部端口无法映射 | 机房限制 HTTP/HTTPS 自定义服务 | 服务器内部打开真实 localhost TensorBoard 并截图 | 页面和曲线真实；不能从公网交互 |
| Run 2 仍有 1 个严格失败 | 冲突问题下遗漏精确名称 | 如实记 7/8，不继续追分 | identity 泛化仍有限 |

剩余风险还包括：identity 数据规模很小，固定 8 题不能代表开放问题分布；没有独立验证集，训练 loss 下降可能包含记忆；本周没有对微调后通用能力做回归评测。因此结论限定为“管线跑通且固定身份行为改善”。

## 11. Qwen2.5 模型设计哲学总结

Qwen2.5 的设计不是单一追求参数量，而是在覆盖、表达、训练稳定与部署效率之间分配预算：

1. 大词表以 embedding/lm head 参数成本换取多语言、代码和混合符号的 token 效率。
2. GQA 用 K/V 共享换取更小 KV Cache 和更高解码吞吐，同时保留多个 Query heads。
3. RoPE 通过旋转 Q/K 表达相对位置，用多频率覆盖不同距离；位置能力仍需训练与部署共同支持。
4. RMSNorm 简化归一化并服务训练稳定，SwiGLU 用门控 FFN 增强非线性表达。
5. Chat Template 把对话结构映射到模型学习过的 token 协议，是训练分布的一部分。
6. QLoRA 冻结量化基座、只训练少量低秩参数，用约 0.2643% 的可训练参数完成目标行为适配，体现参数高效微调的工程取舍。

## 12. 下周计划

1. 系统学习 Day 1–Day 5 的命令、数据流和口述要点，独立复现关键验证。
2. 为 identity 实验增加独立验证集和更多语言/对抗改写，避免只看训练模板附近问题。
3. 在固定通用评测集上比较基座与 adapter，检查代码、推理、事实性是否回退。
4. 学习 LoRA rank、target modules、学习率和 epoch 的消融设计，坚持单变量与固定种子。
5. 熟悉 LLaMA-Factory 的数据格式、模板、训练、导出和推理流程，为后续领域数据微调准备。

## 13. 复现命令、版本与文件索引

Day 5 训练命令：

```bash
source /root/miniconda3/bin/activate llm_exp
cd /root/autodl-tmp/qwen25-week1
llamafactory-cli train week1/configs/qwen25_7b_identity_qlora.yaml
llamafactory-cli train week1/configs/qwen25_7b_identity_qlora_run2.yaml
```

本地只读验证：

```bash
python3 -m unittest discover -s deliverables/week1/day5/source/tests -v
python3 deliverables/week1/day5/source/scripts/validate_day5.py \
  --root deliverables/week1/day5
```

每天入口：

- [Day 1：环境初始化](../day1/README.md)
- [Day 2：原生推理](../day2/README.md)
- [Day 3：架构分析](../day3/REPORT.md)
- [Day 4：Tokenizer](../day4/README.md)
- [Day 5：QLoRA 与 TensorBoard](README.md)
- [第 1 周完整执行计划](../../../docs/week1_execution_plan.md)

本报告所有核心数字均可追溯到对应日志、JSONL、CSV、Notebook 或配置文件；模型权重、adapter 权重、checkpoint、TensorBoard event、Conda 环境和缓存均未加入 Git 仓库。
