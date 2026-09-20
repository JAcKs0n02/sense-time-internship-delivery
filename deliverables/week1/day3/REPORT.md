# Qwen2.5 架构分析报告

## 1. 分析对象与结论摘要

本报告分析 `Qwen/Qwen2.5-7B-Instruct`。配置取自 [Qwen 官方模型仓库的 config.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)。参数量采用两条相互核对的验证路径：一是读取配置并按照 `Qwen2ForCausalLM` 的张量形状计算，二是在 AutoDL RTX 3090 上加载实际四个权重分片并逐模块统计；同时执行最小 GPU 前向传播，确认模型权重能够正常参与计算。

核心结论：

- 模型是 decoder-only 因果语言模型，共 28 个 Decoder Layer，hidden size 为 3584。
- 每个注意力头维度为 `3584 / 28 = 128`。模型采用 GQA：28 个 Query heads、4 个 Key/Value heads，每 7 个 Query heads 共享一组 K/V。
- `rope_theta=1,000,000` 是 RoPE 的频率基数，不是最大上下文长度；当前配置直接声明 `max_position_embeddings=32,768`。
- 逐层统计得到总参数量 `7,615,616,512`，约 7.62B；去除输入 embedding 和输出 lm head 后，Transformer 主体为 `6,525,621,760`，约 6.53B。
- 与原始 Meta-Llama-3-8B 相比，两者都采用 decoder-only、RoPE、GQA、RMSNorm 和 SwiGLU；Qwen2.5-7B 更窄、更浅，但词表更大，KV heads 更少，Q/K/V 投影带 bias，配置中的直接上下文长度也更长。

## 2. 模型主干结构

单个 token 的主要数据流如下：

```text
token id
  ↓
Token Embedding (152064 × 3584)
  ↓
28 × Decoder Layer
  ├─ RMSNorm → GQA + RoPE → 残差连接
  └─ RMSNorm → SwiGLU FFN → 残差连接
  ↓
Final RMSNorm
  ↓
LM Head (3584 → 152064)
  ↓
下一 token 的 logits
```

这是典型的 pre-norm Transformer：注意力和 FFN 之前先做 RMSNorm，再把子层输出与残差相加。模型使用因果掩码，因此每个位置只能利用当前位置及之前的 token。

## 3. `config.json` 逐字段分析

| 字段 | 实际值 | 含义与影响 |
|---|---:|---|
| `architectures` | `Qwen2ForCausalLM` | Transformers 用它选择因果语言模型类；输出是对下一个 token 的概率分布。 |
| `model_type` | `qwen2` | 指定配置和模型实现属于 Qwen2 系列；Qwen2.5 文本模型沿用该实现。 |
| `vocab_size` | 152064 | embedding 和 lm head 都有 152064 行。大词表有利于覆盖多语言、代码和特殊符号，但显著增加首尾矩阵参数。 |
| `hidden_size` | 3584 | 每个 token 在主干网络中的向量宽度，也是残差流宽度。它会同时影响注意力、FFN 和归一化参数量。 |
| `num_hidden_layers` | 28 | Decoder Layer 数量。层数增加会近似线性增加参数量和串行计算深度。 |
| `num_attention_heads` | 28 | Query head 数。由此得到 `head_dim=3584/28=128`。多个 Query heads 能从不同子空间检索信息。 |
| `num_key_value_heads` | 4 | Key/Value head 数。由于 `28/4=7`，每 7 个 Query heads 共享一组 K/V，即 GQA。 |
| `intermediate_size` | 18944 | SwiGLU FFN 的中间宽度。gate、up、down 三个投影合计贡献 `3×3584×18944` 个参数。 |
| `hidden_act` | `silu` | SwiGLU 的门控分支使用 SiLU 激活，再与另一分支逐元素相乘。 |
| `rope_theta` | 1000000.0 | RoPE 的频率基数，控制各维度旋转频率的尺度；它不等于可输入的最大 token 数。 |
| `max_position_embeddings` | 32768 | 当前配置直接声明的位置长度。官方模型卡说明，超过 32768 时需要按说明启用 YaRN 等 RoPE scaling。 |
| `sliding_window` | 131072 | 配置中保留的滑动窗口值，但 `use_sliding_window=false`，因此当前配置没有启用滑动窗口注意力。不能把该值直接当作当前已启用上下文。 |
| `rms_norm_eps` | 1e-6 | RMSNorm 分母中的数值稳定项，避免均方根过小导致除法不稳定。 |
| `tie_word_embeddings` | `false` | 输入 embedding 与输出 lm head 不共享权重，二者各占 `152064×3584` 个参数。 |
| `attention_dropout` | 0.0 | 注意力概率上的 dropout 比例为 0；推理阶段本来也会关闭 dropout。 |
| `use_cache` | `true` | 自回归生成时缓存历史 K/V，避免每生成一个 token 都重新计算全部历史注意力投影。 |
| `torch_dtype` | `bfloat16` | 推荐权重精度为 BF16。它影响存储与计算精度，不改变参数“个数”。 |
| `bos_token_id` | 151643 | 序列开始 token 的 ID。 |
| `eos_token_id` | 151645 | 序列结束 token 的 ID，生成遇到该 token 时可停止。 |

### 3.1 `vocab_size` 与 Day 2 Tokenizer 长度为什么不同

Day 2 实验记录的 `len(tokenizer)=151665`，而模型配置是 `vocab_size=152064`。二者口径不同：

- `len(tokenizer)` 表示当前 Tokenizer 实际可访问的 token 数量。
- `config.vocab_size` 决定模型 embedding 与 lm head 的矩阵行数。

模型矩阵可以为对齐、扩展预留或实现约束保留额外行，因此两者不必完全相等。参数量必须使用 `config.vocab_size=152064`，不能用 Tokenizer 长度代替。

## 4. GQA：为什么 28 个 Q heads 只配 4 个 KV heads

### 4.1 MHA、MQA 与 GQA

- MHA：每个 Query head 都有独立的 K/V head，表达最独立，但 KV Cache 最大。
- MQA：所有 Query heads 共用唯一一组 K/V，KV Cache 最小，但共享程度最高。
- GQA：把多个 Query heads 分组，每组共享一组 K/V，在表达能力与推理效率之间折中。

Qwen2.5-7B 的分组关系为：

```text
28 个 Query heads ÷ 4 个 KV heads = 每组 7 个 Query heads
```

### 4.2 KV Cache 节省量

每个 head 的维度为 128。对单个 token、单个层、单个 batch 元素，只看 K 和 V：

- 当前 GQA：`2 × 4 × 128 = 1024` 个缓存元素。
- 假设使用 28 个 KV heads 的 MHA：`2 × 28 × 128 = 7168` 个缓存元素。
- 比例：`1024 / 7168 = 1/7`。
- 理论减少：`1 - 1/7 = 85.7%`。

例如在 BF16、batch size 1、28 层、32768 token 的理想化计算中，当前 GQA 的纯 K/V 元素约占 1.75 GiB；若改为同头数 MHA，则约为 12.25 GiB。实际运行还会有张量布局、框架和其他中间状态开销，但 `1/7` 的 K/V 元素比例不变。

因此，GQA 直接服务于自回归推理：它显著降低长上下文 KV Cache 的显存压力和带宽需求，同时保留 28 个不同的 Query heads。

## 5. RoPE：为什么旋转 Q 和 K

RoPE 将一个 head 的通道两两配对，并按照 token 位置旋转。对位置 `m`，可把旋转写成 `R(m)`：

```text
q_m' = R(m) q_m
k_n' = R(n) k_n
```

注意力内积满足：

```text
(q_m')ᵀ k_n' = q_mᵀ R(m)ᵀR(n) k_n = q_mᵀR(n-m)k_n
```

结果自然依赖相对距离 `n-m`。这意味着模型不必把位置向量直接加到 token embedding 上，也能让注意力分数感知相对位置。

不同通道对使用不同旋转频率：高频通道更敏感于近距离顺序，低频通道变化更慢，适合表达较长距离关系。`rope_theta` 控制这组频率的尺度。Qwen2.5-7B 使用 `1,000,000`，Meta-Llama-3-8B 使用 `500,000`。

需要特别区分：

- `rope_theta`：频率基数。
- `max_position_embeddings`：配置直接声明的位置长度。
- RoPE scaling：用于长度外推的额外策略，例如 YaRN。
- 实际可用长度：还受训练长度、KV Cache、显存和推理框架限制。

因此，不能看到 `rope_theta=1,000,000` 就声称模型支持一百万 token。

## 6. 参数量统计

### 6.1 单层参数公式

令：

```text
V=152064, H=3584, L=28, A=28, K=4, I=18944, D=H/A=128
```

Qwen2 的 Q/K/V 投影带 bias，O 投影不带 bias；SwiGLU 的三个投影不带 bias。由此：

| 组件 | 公式 | 参数量 |
|---|---:|---:|
| Q projection | `H×(A×D) + A×D` | 12,848,640 |
| K projection | `H×(K×D) + K×D` | 1,835,520 |
| V projection | `H×(K×D) + K×D` | 1,835,520 |
| O projection | `(A×D)×H` | 12,845,056 |
| Attention 合计 | 上述四项之和 | 29,364,736 |
| SwiGLU FFN | `3×H×I` | 203,685,888 |
| 两个 RMSNorm | `2×H` | 7,168 |
| 单个 Decoder Layer | 三部分之和 | 233,057,792 |

### 6.2 总参数量

| 参数组 | 公式 | 参数量 | 占总参数 |
|---|---:|---:|---:|
| Token embedding | `V×H` | 544,997,376 | 7.1563% |
| 28 个 Decoder Layer | `28×233057792` | 6,525,618,176 | 85.6873% |
| Final RMSNorm | `H` | 3,584 | 0.000047% |
| LM head | `V×H` | 544,997,376 | 7.1563% |
| **总计** | 四项之和 | **7,615,616,512** | **100%** |

由于 `tie_word_embeddings=false`，embedding 与 lm head 必须计算两次。若漏掉 Q/K/V bias，会少算每层 4608 个参数；若错误地认为首尾权重共享，会少算 544,997,376 个参数。

完整的 28 层公式 CSV 见 [`source/results/day3_parameter_counts.csv`](source/results/day3_parameter_counts.csv)，实际权重 CSV 见 [`source/results/day3_weight_parameter_counts.csv`](source/results/day3_weight_parameter_counts.csv)，控制台摘要见 [`source/results/day3_parameter_summary.txt`](source/results/day3_parameter_summary.txt)，手工交叉核对见 [`source/results/day3_manual_crosscheck.txt`](source/results/day3_manual_crosscheck.txt)。

## 7. Qwen2.5-7B 与原始 Meta-Llama-3-8B 对比

本节对比对象固定为 2024 年发布的原始 `Meta-Llama-3-8B`，不是 Llama 3.1、3.2 或其他衍生版本。

| 维度 | Qwen2.5-7B-Instruct | Meta-Llama-3-8B | 设计含义 |
|---|---:|---:|---|
| 模型类型 | decoder-only | decoder-only | 都用因果掩码做自回归生成。 |
| hidden size | 3584 | 4096 | Llama 3 单层残差流更宽。 |
| Decoder layers | 28 | 32 | Llama 3 更深；Qwen2.5-7B 更浅。 |
| Q heads | 28 | 32 | 两者 head dimension 都是 128。 |
| KV heads | 4 | 8 | 两者都用 GQA；Qwen 每 7 个 Q 共享 K/V，Llama 每 4 个 Q 共享 K/V。 |
| vocab size | 152064 | 128256 | Qwen 词表更大，首尾矩阵成本也更高。 |
| intermediate size | 18944 | 14336 | Qwen 单层 FFN 中间宽度更大；不能只凭 hidden size 判断单层容量。 |
| 激活/FFN | SiLU / SwiGLU | SiLU / SwiGLU | 都用门控 FFN，包含 gate、up、down 三个投影。 |
| 归一化 | RMSNorm | RMSNorm | 都使用 pre-norm 风格，避免 LayerNorm 的均值中心化计算。 |
| RoPE theta | 1,000,000 | 500,000 | 都使用 RoPE，但频率基数不同。 |
| 配置直接长度 | 32768 | 8192 | Qwen 当前配置声明的长度更大；更长部署仍需考虑 scaling 与显存。 |
| Attention bias | Q/K/V 有，O 无 | Q/K/V/O 均无 | Qwen 为三个输入投影增加小量可学习偏置。 |
| RMSNorm epsilon | `1e-6` | `1e-5` | 都用于数值稳定，但取值不同。 |
| embedding 与 lm head | 不共享 | 不共享 | 两者首尾矩阵均分别占参数。 |
| attention dropout | 0.0 | 0.0 | 配置中注意力 dropout 均为 0。 |

### 7.1 共同设计哲学

两者都把推理效率作为主架构目标：使用 GQA 缩小 KV Cache，使用 RoPE 注入相对位置信息，使用 RMSNorm 简化归一化，并使用 SwiGLU 提升 FFN 的门控表达能力。这些设计组合已经成为现代 decoder-only 大模型的常见主干。

### 7.2 Qwen2.5-7B 的差异化取舍

1. **更大的词表**：增强多语言、代码和结构化符号覆盖，但输入 embedding 与输出 lm head 合计约占 10.90 亿参数。
2. **更强的 K/V 共享**：4 个 KV heads 对 28 个 Q heads，使 KV Cache 比同 Q 头数的 MHA 少约 85.7%。
3. **较大的 FFN 中间层**：虽然 hidden size 小于 Llama 3-8B，但 intermediate size 更大，参数容量更多地分配到 SwiGLU FFN。
4. **Q/K/V bias**：增加的参数量很小，却给三个注意力输入投影提供额外平移自由度。
5. **更长的直接位置配置**：`max_position_embeddings=32768`，并为更长上下文保留 RoPE scaling 路径；但配置值、训练能力和部署能力必须分开判断。

## 8. 可复现方法与验证结果

运行命令：

```bash
python3 deliverables/week1/day3/source/scripts/analyze_params.py \
  --config deliverables/week1/day3/source/config/config.json \
  --output deliverables/week1/day3/source/results/day3_parameter_counts.csv

python3 -m unittest discover \
  -s deliverables/week1/day3/source/tests -v
```

验证结果：

- 3 项测试全部通过。
- 28 个 Decoder Layer 均为 `233,057,792` 个参数。
- 脚本总计、CSV 汇总和手工公式均为 `7,615,616,512`。
- 官方配置与本地 `config.json` 的 SHA-256 均为 `7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c`。
- 服务器实际模型目录中的 `config.json` 也得到同一 SHA-256。
- AutoDL 上加载四个实际 BF16 权重分片后，28 层实测参数量、总参数量和关键张量形状均与公式一致。
- 单 token GPU 前向传播通过，输出 logits 形状为 `(1, 1, 152064)`，峰值已分配显存约 `14.195 GiB`。
- 配置公式统计与实际权重统计相互独立、结果一致；实际模型权重只在 AutoDL 数据盘读取，没有复制进 Git 仓库。

## 9. 分析边界与适用范围

- 参数量先由官方配置和 Transformers 中的张量形状公式计算，再通过自动测试、手工公式和实际 BF16 权重逐模块统计三重核对。三种方法的总计均为 `7,615,616,512`。
- GPU 日志中的 `body_parameters=7070619136` 对应 Transformers 的 `model.model`，它包含 token embedding、28 层和 final norm；报告中的“Transformer 主体 `6525621760`”特指去除输入 embedding 与输出 lm head 后的 28 层加 final norm，两者统计口径不同。
- Llama 3 对比基于原始 Meta-Llama-3-8B 的官方配置、模型卡和实现代码，没有执行两模型的同硬件吞吐测试，因此报告只比较架构取舍，不据此宣称实际速度或任务效果谁更优。
- 1.75 GiB 与 12.25 GiB 是按 BF16、batch size 1、32768 token、28 层计算的纯 K/V 元素理论值，不包含框架工作区、张量对齐、临时激活和显存碎片。
- 更长上下文的实际质量需要用长文本任务、不同 RoPE scaling 配置和固定评测集验证，不能只根据 `rope_theta` 或模型卡上限推断。

## 10. 最终结论

Qwen2.5-7B-Instruct 不是简单地“把层数和宽度堆到 7B”。它将约 6.53B 参数放在 28 层 Transformer 主体中，同时用较大词表支持多语言和代码，用 28Q/4KV 的 GQA 降低生成阶段的 KV Cache，用 RoPE 支持相对位置建模，再以 RMSNorm 和 SwiGLU 构成稳定高效的 Decoder Layer。

理解这套设计时应抓住三组取舍：词表覆盖与首尾参数成本、注意力表达能力与 KV Cache、长位置能力与实际部署资源。它们共同解释了模型“为什么这样设计”，而不只是配置字段“分别是多少”。

## 参考资料

- [Qwen2.5-7B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen2.5-7B-Instruct 官方 config.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
- [Transformers 的 Qwen2 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Meta-Llama-3-8B 官方模型卡](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [Meta Llama 3 官方模型实现](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
- [Meta-Llama-3-8B 官方 config.json](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json)
