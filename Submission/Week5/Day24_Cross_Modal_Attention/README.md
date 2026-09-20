# Day 24：Qwen2-VL 跨模态注意力热力图

## 交付对应

老师要求使用 PyTorch Hooks 提取 VLM 中间层（例如第 20 层）的 Cross-Attention，展示生成特定词时关注的图像区域，并交付至少 3 张热力图。

本目录实际交付 4 个不同图片、4 个不同目标词的案例。每张图都包含原图、纯注意力热力图和 overlay 三个面板。4/4 案例提取成功，自动验证 9/9 项全部 PASS。

## 热力图与主要观察

| 图 | 图片 | 目标词 | 性质 | 主要观察 |
|---|---|---|---|---|
| [Attention 01](Figures/Attention_01_Scene_Mountain.png) | 自然场景 | 山峰 | 有图像依据 | 最强热区落在远处雪山山体和峰顶 |
| [Attention 02](Figures/Attention_02_Table_DESCRIPTION.png) | 表格 | `DESCRIPTION` | 有图像依据 | 最强热区落在相应表头及邻近表头 |
| [Attention 03](Figures/Attention_03_Formula_Gamma.png) | 手写公式 | `gamma` | Day 23 幻觉符号 | 关注分散在公式局部，不能证明图中存在 gamma |
| [Attention 04](Figures/Attention_04_Logo_Triangle.png) | Logo | 三角形 | Day 23 幻觉几何 | 关注沿弧带与边缘分散，图中没有独立三角形轮廓 |

## 方法摘要

- 模型：`Qwen/Qwen2-VL-7B-Instruct`，revision `eed13092ef92e448dd6875b2a00151bd3f7db0ac`；
- 层号：教师口径第 20 层，Python 索引 19；
- Hook 路径：`model.model.layers.19.self_attn`；
- 注意力实现：`eager`，Hook 捕获该层 softmax 后权重；
- 解释文本：Day 23 已归档的真实模型回答，SHA-256 核对后做 teacher-forced forward；
- 跨模态切片：目标文本 token 的预测 query 对视觉 token key 区间的 self-attention；
- 聚合：多 token 目标求算术平均，然后对 28 个 attention heads 求算术平均；
- 绘图：单案例线性 min-max 归一化，`turbo` 色图，overlay alpha `0.45`。

Qwen2-VL 没有独立的文本—图像 `cross_attn` 模块，所以本实验按它的真实架构，将 Cross-Attention 实现为语言层中“目标文本 token →视觉 token”的 self-attention 切片。

## 文件

| 文件 | 内容 |
|---|---|
| `Figures/Attention_01_Scene_Mountain.png` | “山峰”案例三联图 |
| `Figures/Attention_02_Table_DESCRIPTION.png` | `DESCRIPTION` 案例三联图 |
| `Figures/Attention_03_Formula_Gamma.png` | `gamma` 幻觉符号案例三联图 |
| `Figures/Attention_04_Logo_Triangle.png` | “三角形”幻觉几何案例三联图 |
| `Attention_Heatmap_Records.csv` | 层号、token ID、视觉区间、网格、行和、解读和图片 SHA-256 |
| `Cross_Modal_Attention_Method.md` | 完整方法、校验和限制 |
| `Day24_Validation.json` | 配置、案例、Hook、token、网格、数组、PNG 和摘要的 9 项验证 |

## 解读限制

Attention 热力图显示的是该层的内部相对权重，不是严格的因果解释。特别是，幻觉词对某些图像区域有响应，并不意味着图中真实存在该对象。不同案例均各自做 min-max 归一化，因此不能用图上颜色直接比较案例间的绝对注意力强度。

本目录仅保留老师验收需要的热力图、记录表、方法说明和验证摘要，不包含模型权重、原始数组、执行脚本或调试日志。
