# Day 24：Qwen2-VL 跨模态注意力热力图

## 状态

**已完成（PASS）。** 已从 Qwen2-VL 第 20 层提取 4 个真实案例的目标文本 token →视觉 token 注意力，交付 4 张原图/热力图/overlay 三联图，自动校验 9/9 项全部通过。

执行依据：[第 5 周完整执行计划](../../../docs/week5_execution_plan.md) · [Week 5 工程索引](../README.md) · [Day 22 输入规范](../day22/README.md) · [Day 23 推理规范](../day23/README.md)

## 老师要求

使用 PyTorch Hooks 提取 VLM 中间层（老师举例第 20 层）的 Cross-Attention 权重，绘制模型生成特定词语时关注的图像区域，交付至少 3 张注意力热力图。

Qwen2-VL 并不存在一个可以直接按模块名理解为“文本 query 对图像 key/value”的典型独立 `cross_attn` 层。视觉 token 经视觉编码器和 merger 后进入语言模型 token 序列，语言模型使用 self-attention 完成视觉与文本交互。因此本项目对老师术语采用以下可复现定义：

> 跨模态注意力 = 指定语言模型层中，目标文本 token 对视觉 token 索引区间的 self-attention 权重。

教师材料会说明这是 Qwen2-VL 架构下对 Cross-Attention 要求的实现，不把纯视觉塔 attention 冒充文本到图像关注。

## 当日交付

- 至少 3 个不同图片、不同目标词的跨模态注意力案例；
- 每个案例的原图、纯热力图和 overlay 三联图；
- 每个 attention head 的原始数组与 head 均值数组；
- 图片、Prompt、完整目标文本、目标 token、token ID、层号和查询位置；
- 视觉 token 起止索引、`image_grid_thw`、merger 后网格和聚合方法；
- 数组与图片 SHA-256、有效配置和生成脚本；
- 方法说明、限制和至少 3 张教师版热力图。

## 实际结果

| 案例 | Day 23 记录 | 目标词 | 类型 | merger 后网格 | 主要观察 |
|---|---|---|---|---:|---|
| `case-01-scene-mountain` | `d23-scene-description` | 山峰 | 有依据对象 | 16×24 | 最强热区落在远处雪山山体和峰顶 |
| `case-02-table-description` | `d23-table-ocr` | `DESCRIPTION` | 有依据 OCR | 7×42 | 最强热区落在相应表头及邻近表头 |
| `case-03-formula-gamma` | `d23-formula-description` | `gamma` | 幻觉符号 | 11×25 | 响应分散于公式区，不能证明图中存在 gamma |
| `case-04-logo-triangle` | `d23-logo-structure` | 三角形 | 幻觉几何 | 19×19 | 响应沿 Logo 弧带和边缘分散，未形成真实三角形证据 |

共 4/4 案例 PASS，模型 revision 为 `eed13092ef92e448dd6875b2a00151bd3f7db0ac`，实际 Hook 路径为 `model.model.layers.19.self_attn`。完整注意力行和介于 0.997341–1.002376，在 bfloat16 允许误差内通过归一化检查。

结果入口：[4 案例记录表](source/results/attention_heatmap_records.csv) · [方法与解读](source/results/method_notes.md) · [尝试与纠偏记录](source/results/attempt_history.md) · [验证报告](source/results/validation_report.json) · [热力图目录](source/results/heatmaps/)

## 专业术语

| 术语 | 解释 |
|---|---|
| Attention | 用 query 与 key 的相似度决定一个 token 对其他 token 的关注分布 |
| Self-attention | query、key、value 来自同一序列；Qwen2-VL 语言层通过它连接文本和视觉 token |
| Cross-Attention | 经典架构中 query 与 key/value 来自不同模态；本项目按 Qwen2-VL 实际架构改写为跨模态 self-attention 切片 |
| Q / K / V | Query、Key、Value；attention 先计算缩放后的 QK 点积，再经 mask 与 softmax 得到权重 |
| Attention head | 并行学习不同关系的一组注意力分支；本实验保留逐头数组并报告 head 均值 |
| Causal mask | 防止当前文本 token 看到未来 token 的掩码；重算 attention 时必须使用真实 mask |
| Hook | 注册在模型模块上的回调，用于捕获前向输入、输出或 Q/K 投影 |
| Teacher forcing | 把已知目标文本作为输入做一次前向传播，用来准确提取某个目标 token 查询位置的权重 |
| 视觉 token 区间 | 输入序列中属于图片的 token 起止位置，必须从真实 processor 输出确定 |
| `image_grid_thw` | processor 返回的时间、高、宽 patch 网格信息；单图时间维通常为 1，但以实际输出为准 |
| Spatial merger | Qwen2-VL 将相邻视觉特征合并后送入语言模型的组件；二维还原必须读取真实 merge 配置 |
| 热力图 | 将二维权重归一化并映射为颜色，再叠加到原图；高权重不等于严格因果贡献 |
| Grad-CAM | 基于梯度的另一类可视化方法；可以做扩展实验，但不能替代老师要求的中间层 attention |

## 固定输入

- Day 22 冻结的 `Qwen/Qwen2-VL-7B-Instruct` revision、processor 与五张图片；
- Day 22 的 `min_pixels`、`max_pixels`、图片 SHA 和 `image_grid_thw`；
- Day 23 已保存的 Prompt 与原始回答；
- 至少 3 个预注册案例，每个案例选择不同 `image_id` 和不同目标词；
- 教师口径使用第 20 层，实际 Python 索引为 19，Qwen2-VL 共 28 个语言层；
- attention head 聚合固定为算术均值，同时保留每个 head 的原始值；
- 热力图归一化方法在看到结果前冻结，全部主案例使用同一方法。

目标词必须能与 tokenizer 的一个或多个实际 token 对齐。多 token 词语要记录每个 token 的权重及词级聚合方式，不能只凭解码文本猜测索引。

## 详细实施步骤

### 1. 静态检查模型结构

读取模型配置、语言层数量、attention 模块路径、head 数、head dimension、rope、spatial merge size 和 processor 输出字段。打印第 20 层附近模块树，确认 Hook 应注册在哪个实际模块，而不是照搬其他 VLM 的模块名。

### 2. 建立最小案例并定位视觉 token

对一个冻结图片和简短 Prompt 调用 processor，保存 `input_ids`、特殊视觉 token、attention mask 和 `image_grid_thw`。根据模型官方 special token 定义定位视觉 token 区间，并验证区间长度与 merger 后网格 token 数一致。

### 3. 优先使用 eager attention

以 `attn_implementation="eager"` 加载模型，并设置真实实现支持的 attention 输出选项。先在一个 teacher-forced 目标 token 上检查返回张量的层数、batch、head、query 和 key 维度。FlashAttention/SDPA 若不返回权重，不能把空值或近似随机图继续用于正式绘图。

### 4. Hook 回退提取 Q/K

若 eager 路线仍不能获得权重，在实际第 20 层 attention 模块注册 Hook，捕获进入 attention 计算的 Q/K 或投影结果。应用与模型一致的 rotary position embedding、head reshape、scale、causal/padding mask 和 softmax，重算目标查询位置的 attention。

回退结果必须先与一个可直接返回 attention 的最小 eager 案例做数值对照；若无法建立对照，只能报告方法限制，不能把未经验证的重算作为正式证据。

### 5. 对齐目标 token

保存完整生成文本与 tokenizer 分词结果。实际实施对 Day 23 已归档且 SHA-256 通过的模型回答执行 teacher-forced forward，不在 eager 模式下二次重新生成文本。目标 token 的查询位置为其绝对位置的前一位，多 token 词保留逐 token 数组后求算术平均。

### 6. 截取跨模态权重

从指定层的 `[head, query, key]` 权重中选取目标查询位置，再仅截取视觉 token 索引区间。逐头数组转换为 float32 保存，检查所有值有限、非负；同时保存 head 均值。记录截取前后的形状，防止把文本或 padding token 混入视觉网格。

### 7. 还原二维视觉网格

使用实际 `image_grid_thw` 与模型 spatial merge size 推导送入语言模型后的网格，不硬编码“除以 2”。验证二维单元总数与视觉 token 切片长度完全一致。若一张输入包含多图或视频，Day 24 主案例先限制为单图，避免时间维和多图边界混淆。

### 8. 绘制三联图

将 head 均值网格按预注册方法归一化，上采样到原图显示尺寸，输出：

1. 原图；
2. 纯热力图；
3. 固定 alpha 的 overlay。

图题标明 `image_id`、目标词、token、层号、head 聚合和归一化方式。保存绘图输入 `.npy` 与元数据 JSON，确保图片可以重新生成。

### 9. 运行至少三个正式案例

选择至少 3 张不同图片和 3 个不同目标词，优先覆盖文字/数字、对象和 UI/结构位置三类关注。所有案例使用同一模型 revision、processor 和绘图配置；不因为热力图“不好看”而替换目标词。

### 10. 验证并更新状态

检查数组形状、有限值、token 对齐、视觉网格一致性、图片可打开、SHA 可重算和绘图可复现。只有至少 3 个案例全部通过后才修改状态和生成教师版热力图。若使用 AutoDL，完成证据同步后关机，关机证据不进入提交。

## 实际文件结构

```text
deliverables/week5/day24/
├── README.md
├── configs/
│   ├── attention_cases.json
│   └── visualization_config.json
└── source/
    ├── results/
    │   ├── raw_attention/        # 4 案例逐 token/逐 head/均值数组
    │   ├── heatmaps/
    │   │   └── 4 张 *-triptych.png
    │   ├── attention_metadata.jsonl
    │   ├── attention_heatmap_records.csv
    │   ├── method_notes.md
    │   └── validation_report.json
    ├── scripts/
    │   ├── extract_cross_modal_attention.py
    │   ├── render_attention_heatmaps.py
    │   └── validate_day24.py
    └── tests/
        ├── test_attention_mapping.py
        └── test_day24_pipeline.py
```

## 完成门禁

- [x] Qwen2-VL 的跨模态 self-attention 定义已写入报告；
- [x] 第 20 层实际模块路径 `model.model.layers.19.self_attn` 已验证；
- [x] Hook/eager 提取使用真实的 softmax 后权重、mask 和查询位置；
- [x] 4 张不同图片和 4 个不同目标词完成；
- [x] 每个案例记录目标 token ID、位置、层号、视觉 token 范围和 `image_grid_thw`；
- [x] 逐 head 数组与均值数组存在，全部数值有限；
- [x] 二维网格单元数等于视觉 token 切片长度；
- [x] 原图、热力图、overlay 三联图可从数组和元数据重新生成；
- [x] 未使用 Grad-CAM、随机图或纯 ViT self-attention 冒充跨模态权重；
- [x] 教师提交包含 4 张热力图和简要方法说明，不含权重、凭据或关机证据。

任一项没有证据时，Day 24 保持未完成。

## 故障处理

- **`output_attentions` 返回空**：切换 eager 实现并核对 Transformers/Qwen2-VL 真实接口；仍为空时进入已验证的 Q/K Hook 回退。
- **Hook 模块名不存在**：打印锁定 revision 的模型树，从实际层对象定位模块；不套用 LLaVA 或其他 VLM 路径。
- **数组维度不匹配**：保存所有形状，逐步核对 batch/head/query/key、视觉区间和 merger 网格；不通过 reshape 强行凑数。
- **目标词被拆成多个 token**：保存每个 token 结果，并在运行前固定词级聚合方式；不只选择最符合预期的 token。
- **热力图均匀或不符合直觉**：先验证索引、mask、数值和归一化；若均正确，如实报告模型在该层的分布，不更换案例美化结果。
- **显存不足**：使用单图、单案例、teacher-forced 单步和较低像素上限；改变像素配置后主案例全部重跑。

## 下一日交接

Day 24 向 Day 27 交付至少 3 个原始 attention 数组、元数据、三联图、验证结果和方法限制。Day 25 与 Day 24 相互独立，只共享 Day 22 的冻结模型与图片，不能用热力图主观判断替代幻觉评分。
