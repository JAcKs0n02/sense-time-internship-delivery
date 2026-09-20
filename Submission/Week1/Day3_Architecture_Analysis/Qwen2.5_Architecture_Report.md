# Qwen2.5-7B-Instruct 架构分析

## 1. 分析对象

本周实际使用的模型是 `Qwen/Qwen2.5-7B-Instruct`。分析所用的 [config.json](config.json) 来自该模型目录，SHA-256 为：

```text
7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c
```

参数量先根据配置和张量形状计算，再用实际 BF16 权重逐模块统计。两种方法得到的总数都是 `7,615,616,512`。

## 2. 模型主干

Qwen2.5-7B-Instruct 是 decoder-only 因果语言模型，计算顺序可以简化为：

```text
token ids
  → token embedding
  → 28 × decoder layer
       ├─ RMSNorm
       ├─ causal self-attention（RoPE + GQA）
       ├─ residual connection
       ├─ RMSNorm
       ├─ SwiGLU FFN
       └─ residual connection
  → final RMSNorm
  → lm_head
  → next-token logits
```

注意力和 FFN 前都先做 RMSNorm，属于 pre-norm 结构。因果掩码使每个位置只能看到当前及之前的 token。

## 3. `config.json` 字段

| 字段 | 数值 | 在模型中的作用 |
|---|---:|---|
| `vocab_size` | 152064 | 决定 embedding 和 lm head 的行数 |
| `hidden_size` | 3584 | 残差流和主隐藏状态宽度 |
| `intermediate_size` | 18944 | SwiGLU 中 gate/up 投影的宽度 |
| `num_hidden_layers` | 28 | decoder layer 数量 |
| `num_attention_heads` | 28 | Query head 数量 |
| `num_key_value_heads` | 4 | Key/Value head 数量 |
| `rope_theta` | 1000000 | RoPE 的频率基数 |
| `max_position_embeddings` | 32768 | 该配置直接声明的位置长度 |
| `hidden_act` | `silu` | SwiGLU 门控分支使用的激活函数 |
| `rms_norm_eps` | `1e-6` | RMSNorm 的数值稳定项 |
| `tie_word_embeddings` | `false` | 输入 embedding 与输出 lm head 不共享权重 |
| `attention_dropout` | `0.0` | 注意力概率的 dropout 比例 |
| `use_cache` | `true` | 自回归生成时使用 KV Cache |
| `torch_dtype` | `bfloat16` | 模型配置中的推荐权重精度 |

每个 attention head 的维度为：

```text
head_dim = hidden_size / num_attention_heads
         = 3584 / 28
         = 128
```

Day 2 记录的 `len(tokenizer)=151665`，小于 `config.vocab_size=152064`。前者是 Tokenizer 当前可访问的 token 数，后者决定模型矩阵的行数。参数统计必须使用配置中的 `vocab_size`。

配置中还有 `sliding_window=131072`，但 `use_sliding_window=false`，所以不能把这个值当作当前已启用的上下文长度。

## 4. GQA

普通多头注意力（MHA）让每个 Query head 都有独立的 K/V head；MQA 让所有 Query heads 共用一组 K/V。GQA 位于两者之间，把 Query heads 分组后共享 K/V。

Qwen2.5-7B 的配置是 28 个 Query heads、4 个 KV heads：

```text
28 / 4 = 7
```

也就是每 7 个 Query heads 共用一组 K/V。Q 投影的输出宽度仍为 `28 × 128 = 3584`，K 和 V 投影的输出宽度分别为 `4 × 128 = 512`。

只计算单层、单 token、单 batch 元素的 K/V 缓存元素：

```text
GQA: 2 × 4 × 128  = 1024
MHA: 2 × 28 × 128 = 7168
```

这里的系数 2 代表 K 和 V。当前 GQA 的元素数是同 Q 头数 MHA 的 `1/7`，理论上减少约 `85.7%`。

按 BF16、batch size 1、28 层、32768 token 粗略计算，纯 K/V 元素约为 1.75 GiB；如果 28 个 Query heads 都配独立 K/V，则约为 12.25 GiB。实际显存还包括框架工作区、张量对齐、中间激活和碎片。

这个取舍主要影响自回归解码。Query 仍保留 28 个不同的 head，而需要长期缓存的 K/V 数量明显减少。

## 5. RoPE

RoPE 不把位置向量直接加到 token embedding 上，而是按位置旋转 Q 和 K 的二维通道对。位置为 `m` 和 `n` 时：

```text
q'_m = R(m)q_m
k'_n = R(n)k_n
```

注意力内积可以写成：

```text
(q'_m)^T k'_n
= q_m^T R(m)^T R(n) k_n
= q_m^T R(n-m) k_n
```

因此内积直接包含相对位置 `n-m`。不同通道对使用不同旋转频率，高频通道对近距离顺序更敏感，低频通道变化较慢。

`rope_theta=1,000,000` 控制频率尺度，但它不是最大 token 数。实际可用长度还取决于训练长度、`max_position_embeddings`、RoPE scaling、KV Cache、显存和推理框架。仅凭 `rope_theta` 不能判断模型支持一百万 token。

## 6. 参数量

记：

```text
V = 152064
H = 3584
L = 28
A = 28
K = 4
I = 18944
D = H / A = 128
```

Qwen2 的 Q/K/V 投影带 bias，O 投影不带 bias。单层 attention 参数为：

| 组件 | 公式 | 参数量 |
|---|---:|---:|
| Q projection | `H × (A×D) + A×D` | 12,848,640 |
| K projection | `H × (K×D) + K×D` | 1,835,520 |
| V projection | `H × (K×D) + K×D` | 1,835,520 |
| O projection | `(A×D) × H` | 12,845,056 |
| Attention 合计 | 四项相加 | 29,364,736 |

SwiGLU 有 gate、up、down 三个无 bias 投影：

```text
3 × H × I = 3 × 3584 × 18944 = 203,685,888
```

两个 RMSNorm 各有 `H` 个参数：

```text
2 × H = 7,168
```

所以单个 decoder layer 为：

```text
29,364,736 + 203,685,888 + 7,168
= 233,057,792
```

总参数量如下：

| 参数组 | 参数量 |
|---|---:|
| Token embedding | 544,997,376 |
| 28 个 decoder layer | 6,525,618,176 |
| Final RMSNorm | 3,584 |
| LM head | 544,997,376 |
| 总计 | **7,615,616,512** |

因为 `tie_word_embeddings=false`，embedding 和 lm head 必须分别计算。逐层结果见 [parameter_counts.csv](parameter_counts.csv)，计算脚本见 [analyze_params.py](analyze_params.py)。

实际加载四个 BF16 权重分片后，28 层均为 `233,057,792` 个参数，总数与公式一致。单 token 前向传播得到的 logits 形状为 `(1, 1, 152064)`，峰值已分配显存约 `14.195 GiB`。

## 7. 与原始 Meta-Llama-3-8B 对比

这里的 Llama 3 指 2024 年发布的原始 `Meta-Llama-3-8B`，不是 Llama 3.1 或其他版本。

| 维度 | Qwen2.5-7B-Instruct | Meta-Llama-3-8B |
|---|---:|---:|
| 模型类型 | decoder-only | decoder-only |
| hidden size | 3584 | 4096 |
| decoder layers | 28 | 32 |
| Query heads | 28 | 32 |
| KV heads | 4 | 8 |
| head dimension | 128 | 128 |
| vocab size | 152064 | 128256 |
| intermediate size | 18944 | 14336 |
| FFN | SiLU / SwiGLU | SiLU / SwiGLU |
| normalization | RMSNorm | RMSNorm |
| RoPE theta | 1,000,000 | 500,000 |
| 配置直接长度 | 32768 | 8192 |
| attention bias | Q/K/V 有，O 无 | Q/K/V/O 均无 |
| RMSNorm epsilon | `1e-6` | `1e-5` |
| embedding 与 lm head | 不共享 | 不共享 |

两者都采用现代 decoder-only 模型中常见的 RoPE、GQA、RMSNorm 和 SwiGLU，但参数分配不同：

- Qwen2.5 的词表更大，embedding 和 lm head 占用更多参数。
- Qwen2.5 使用 28Q/4KV，Llama 3 使用 32Q/8KV，前者的 K/V 共享程度更高。
- Qwen2.5 的 hidden size 较小，但 FFN intermediate size 更大。
- Qwen2.5 的 Q/K/V 投影有 bias，Llama 3 的注意力投影没有 bias。

本次没有在同一硬件、同一推理框架和同一上下文长度下测两者吞吐，因此这里只比较结构，不能据此判断实际速度或任务效果。

## 8. 我对这部分的理解

Qwen2.5-7B 的 7B 参数并不是平均分配到各处：28 个 decoder layer 占主要部分，大词表增加了输入和输出矩阵的成本，较宽的 SwiGLU FFN承担了较多单层参数。GQA 则把解码时需要缓存的 K/V 压到同头数 MHA 的 `1/7`。

RoPE 和 GQA 分别解决不同问题。RoPE 让注意力分数能够利用相对位置信息；GQA 减少生成阶段的缓存和带宽开销。理解这两个模块时，需要把配置值、训练能力和实际部署条件分开。

## 参考资料

- [Qwen2.5-7B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Transformers Qwen2 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)
- [Meta-Llama-3-8B 模型卡](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [Meta Llama 3 实现](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
