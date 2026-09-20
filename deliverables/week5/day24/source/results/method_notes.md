# Day 24 跨模态注意力提取与解读

## 结论摘要

本实验用 `Qwen/Qwen2-VL-7B-Instruct` 的固定 revision `eed13092ef92e448dd6875b2a00151bd3f7db0ac` 完成 4 个案例，全部通过提取、token 对齐、视觉网格、数组完整性和图片完整性校验。老师要求至少 3 张热力图，实际交付 4 张。

两个有图像依据的目标词呈现较清晰的区域对齐：“山峰”聚焦于雪山，`DESCRIPTION` 聚焦于表头。两个 Day 23 已标注为严重幻觉的词呈现不同现象：`gamma` 只在公式区域中分散响应，“三角形”则沿 Logo 弧带和边缘分散，两者都不能证明目标概念真实存在于图像中。

## Cross-Attention 的实际定义

Qwen2-VL 没有一个独立命名为 `cross_attn` 的文本—图像模块。图像经视觉编码器和 spatial merger 转为视觉 token，然后与文本 token 一起进入语言模型的 self-attention。因此本实验将“跨模态注意力”定义为：

> 语言模型指定层中，目标文本 token 的预测 query 对视觉 token key 区间的 self-attention 权重。

这个定义尊重 Qwen2-VL 的真实架构，也对应老师要求的“生成特定词时关注的图像区域”。

## 提取步骤

1. 以 `attn_implementation="eager"` 加载模型，使 attention 模块可返回 softmax 后的权重。
2. 按老师的层号口径选择第 20 层，Python 索引为 19，实际 Hook 路径为 `model.model.layers.19.self_attn`。
3. 从 Day 23 已保存的真实模型回答中选择目标词，核对回答 SHA-256 后执行 teacher-forced forward。不在 eager 模式下二次生成新回答，避免解释对象漂移。
4. 若目标 token 的绝对位置为 `p`，则因果语言模型在 query 位置 `p-1` 预测该 token；多 token 词对各预测 query 求算术平均。
5. 从第 20 层权重中只切取视觉 token key 区间，保留目标 token × 28 heads 的原始数组，再对 token 和 head 求算术平均。
6. 使用 processor 返回的 `image_grid_thw` 和模型 `spatial_merge_size=2` 恢复二维网格，网格单元总数必须与视觉 token 数完全相等。
7. 所有案例使用事先固定的单案例线性 min-max 归一化、`turbo` 色图、双线性插值和 `alpha=0.45`，生成原图/纯热力图/overlay 三联图。

## 完整性校验

- 4/4 案例提取成功，失败数为 0；
- 4 张不同图片、4 个不同目标词；
- attention 完整行和介于 0.997341–1.002376，在 bfloat16 允许误差内等于 1；
- 逐 token、逐 head 和 head 均值数组均为有限非负数，形状和 SHA-256 均通过校验；
- 4 张 PNG 均为 2160×720，可解码且 SHA-256 与元数据一致；
- 自动验证报告 9/9 项全部 PASS。

## 解读限制

- Attention 是内部相对权重，不是严格的因果贡献证明。
- 单案例 min-max 归一化只用于展示空间分布，不能直接比较不同案例的绝对强度。
- head 平均可能遮盖个别 head 的专门模式，因此工程归档同时保留逐 head 数组。
- Day 23 当时未保存生成阶段的原始 output token ID 序列；Day 24 对已归档文本按同一 tokenizer 重新编码后做 teacher forcing。这保证文本、目标词和位置对齐可复现，但不声称按 bit 重放 Day 23 生成时的内部 token 路径。
- 幻觉词对某些图像区域有响应，不意味着图中存在该词所指的对象或几何结构。
