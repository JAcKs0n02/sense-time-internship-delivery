# 第 1 周实习周报：环境与大模型导论

## 1. 本周工作

本周从环境搭建开始，依次完成 Qwen2.5-7B-Instruct 原生推理、架构分析、Tokenizer 实验和 LLaMA-Factory identity QLoRA。CUDA 推理和训练在 AutoDL RTX 3090 上进行；本地电脑用于编写脚本、整理实验结果和执行不依赖 CUDA 的检查。

## 2. 环境

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
| bitsandbytes | 0.43.3 |
| 模型 | `Qwen/Qwen2.5-7B-Instruct` |

`nvidia-smi` 显示的是驱动信息及其可支持的 CUDA 上限，PyTorch 的 `+cu121` 才表示当前构建使用 CUDA 12.1 用户态运行库。Conda 负责 Python 依赖隔离，不替代 NVIDIA 驱动。

四项工具的分工也不同：LLaMA-Factory 用于训练和微调，vLLM 用于推理服务，OpenCompass 用于评测，LangChain 用于应用编排。

## 3. Day 1：环境初始化

RTX 3090 有 24GB 显存，所以本周选择 Qwen2.5-7B，而不是 1.5B。7B 模型的 BF16 权重理论上约占 14GB；实际原生推理峰值约为 14.2GiB。训练阶段使用 4-bit QLoRA，避免全参数训练的梯度和优化器状态占用。

`llm_exp` 环境、PyTorch CUDA 12.1 构建和四项工具均完成安装。`torch.cuda.is_available()` 返回 `True`，CUDA 矩阵计算、BF16 支持和依赖检查正常。

## 4. Day 2：模型下载与原生推理

模型通过 ModelScope 下载，15 个文件完整，四个 Safetensors 权重分片均存在。推理脚本使用 `AutoTokenizer` 和 `AutoModelForCausalLM`，模型只加载一次，再依次处理三组 Prompt。

| 实验 | 输入 / 输出 token | 耗时 | 观察 |
|---|---:|---:|---|
| 代码生成 | 85 / 644 | 15.205s | 生成了 LRUCache；语法可解析，但回答自带测试有错误断言 |
| 逻辑推理 | 93 / 405 | 9.346s | 结论与独立枚举的唯一解一致 |
| 角色扮演 | 66 / 512 | 11.756s | 能按“严格工程导师”角色给出检查清单 |

代码生成实验说明，输出形式完整不等于实现正确。模型回答、生成代码和失败的测试结果都按原样保留。

Qwen 的 Chat Template 使用 `<|im_start|>` 和 `<|im_end|>` 标出角色和回合。`add_generation_prompt=True` 会在末尾增加：

```text
<|im_start|>assistant\n
```

对应 3 个 token：`[151644, 77091, 198]`。`generate` 返回“输入 token + 新生成 token”，所以解码回答时需要先切掉输入部分。

## 5. Day 3：Qwen2.5 架构

Qwen2.5-7B-Instruct 是 28 层 decoder-only 模型，主要配置为：

```text
hidden_size=3584
num_attention_heads=28
num_key_value_heads=4
head_dim=128
intermediate_size=18944
vocab_size=152064
rope_theta=1000000
```

### GQA

28 个 Query heads 共用 4 组 K/V，即每 7 个 Query heads 共享一组 K/V。与 28 个 K/V heads 的 MHA 相比，K/V 缓存元素数为 `4/28=1/7`，理论上减少约 85.7%。Query 的头数没有减少，主要节省的是自回归解码阶段长期保存的 K/V。

### RoPE

RoPE 按 token 位置旋转 Q 和 K 的通道对。位置 `m` 和 `n` 的旋转后内积可以写成与 `R(n-m)` 有关，因此注意力分数能够利用相对距离。

`rope_theta=1,000,000` 是频率基数，不是上下文长度。实际长度还要看训练长度、`max_position_embeddings`、RoPE scaling、KV Cache、显存和推理框架。

配置公式和实际权重统计得到相同总参数量：`7,615,616,512`。单个 decoder layer 为 `233,057,792` 个参数。

## 6. Day 4：Tokenizer

执行版 Notebook 使用 Qwen Tokenizer 测试了 15 个固定样本，包括纯中文、英文、中英混合、emoji、罕见 CJK、Unicode 组合字符、全半角、空白控制、Python、JSON、URL、LaTeX 和长中文。

`<|endoftext|>`、`<|im_start|>`、`<|im_end|>` 的 ID 分别为：

```text
151643
151644
151645
```

长中文样本截断前为 780 token。右截断和左截断都得到 64 token，但分别保留开头和结尾，选择哪一种取决于任务中重要信息的位置。

BPE 与 SentencePiece 的比较使用同一份 2328 行语料，目标词表和实际词表均为 800。实验中的 ByteLevel BPE 和 SentencePiece Unigram 都能对四组样本往返解码，且没有 `<unk>`。这个结果只适用于本次语料和配置；normalizer、pre-tokenizer、byte fallback 和词表大小都会改变中文切分结果。

## 7. Day 5：identity QLoRA

训练数据来自 LLaMA-Factory 官方 `llamafactory/demo_data`，固定 revision 为：

```text
999e7a11dadd6ce929180d7f3d61d4ca7e761db8
```

数据共 91 条，只把模板占位符替换为实验名称 `Qwen2.5-Intern` 和 `Week 1 Internship Project`。

主要训练参数：

| 项目 | 设置 |
|---|---|
| 量化 | 4-bit NF4，double quantization |
| 计算精度 | BF16 |
| LoRA rank / alpha | 8 / 16 |
| LoRA dropout | 0.05 |
| target modules | `all` |
| cutoff length | 512 |
| batch / gradient accumulation | 1 / 4 |
| learning rate | `1e-4` |
| scheduler | cosine |
| seed | 42 |
| epoch | 5 |

最终 Run 2 的结果：

| 指标 | 数值 |
|---|---:|
| 样本数 | 91 |
| 优化步 | 110 |
| 训练耗时 | 160.9298s |
| train loss | 1.0749 |
| 可训练参数 | 20,185,088 |
| 可训练参数比例 | 约 0.2643% |
| 固定身份评测 | 7/8 |
| 退出码 | 0 |

初次 3-epoch 训练的固定评测为 6/8。在其他训练参数不变的情况下，把 epoch 增加到 5 后得到 7/8，因此提交 Run 2。剩余失败项是回答给出了正确创建者，但遗漏评分规则要求的精确模型名称，按原规则记为失败。

TensorBoard 中保留了两次训练的 `train/loss` 曲线。训练 loss 下降说明优化过程正常，但 identity 数据量较小，也没有独立验证集，不能据此判断通用能力提高。

## 8. 遇到的问题

| 问题 | 原因 | 处理 |
|---|---|---|
| Hugging Face 连接超时 | AutoDL 外网链路不稳定 | 模型改用 ModelScope；identity 数据使用本地核对过的同一 Git revision |
| 首次 4-bit 加载失败 | bitsandbytes 0.43.1 与当前路径不兼容 | 升级到 0.43.3，其他核心版本不动 |
| 初始 4-bit 评测加载冲突 | bnb 与 Accelerate 自动 device map 分发冲突 | 训练保持 4-bit，训练前后评测统一使用 BF16 单卡 |
| TensorBoard 外部端口不可访问 | 服务器限制自定义 HTTP 服务映射 | 在服务器内部打开真实 localhost 页面并截图 |
| 模型生成代码测试失败 | 模型回答中的断言与实现行为不一致 | 保留原始输出，把模型生成和工程验证分开记录 |

Day 3 前，GPU 驱动、CUDA 和模型下载路径已经可用。Hugging Face 连接问题没有影响后续实验，模型和训练数据均有可追溯来源。

## 9. 本周收获

这周最重要的变化是把“能运行模型”和“理解运行过程”分开。原生推理需要处理模板、输入 token、生成 token 和显存；架构分析需要从张量形状核对参数，而不是只读模型名称；模型输出还需要独立测试。

我现在对 GQA 和 RoPE 的理解是：GQA 主要减少解码时的 KV Cache，RoPE 主要把相对位置信息带入注意力内积。它们都影响长序列使用，但作用位置和限制条件不同。

QLoRA 的实验也说明，训练管线跑通只代表参数高效微调流程可用。本周的 7/8 是固定身份题结果，不能替代独立验证集或通用能力回归测试。

## 10. 下周计划

1. 独立复现本周关键命令，减少对现成脚本的依赖。
2. 为 identity 实验增加独立验证问题和改写问题。
3. 在固定通用任务上比较基座与 adapter，检查能力回退。
4. 学习 LoRA rank、target modules、学习率和 epoch 的单变量实验。
5. 熟悉 LLaMA-Factory 的数据格式、训练、导出和推理流程。
