# Day 23：Qwen2-VL 图文推理结果

## 交付对应

老师要求对五张图片分别执行内容描述、OCR、结构解释、美学评价和隐含信息推理，并记录模型强项及严重幻觉。本目录提供：

- 5 张图片 × 5 类问题 = 25 条推理结果；
- 每条的 Prompt、模型原始回答、token、延迟、峰值显存和输入身份；
- 每条的质量标签、幻觉标签和具体复核理由；
- 按图片类型与问题类型汇总的能力边界说明。

## 文件

| 文件 | 内容 |
|---|---|
| `VLM_Inference_Results.csv` | 25 条正式结果与逐条复核，模型原始回答未改写 |
| `Capability_Boundary_Summary.md` | 强项、弱项及 6 条严重幻觉记录 |
| `Image_Index.csv` | 五张图片的类型、来源、许可、尺寸与 SHA-256 |
| `Generation_Config.json` | 正式模型 revision、确定性生成参数和 processor 像素预算 |
| `Day23_Validation.json` | 25 条数量、组合、哈希、指标、复核和原始性门禁 |

五张原图位于同级 Day 22 提交目录 `../Day22_Model_and_Image_Preparation/Images/`，可按 `Image_Index.csv` 的 `image_id` 对照。

## 结果摘要

| 指标 | 数量 |
|---|---:|
| 正式推理记录 | 25 |
| 唯一图片-问题组合 | 25 |
| `strong` | 7 |
| `acceptable` | 12 |
| `weak` | 6 |
| 无幻觉 | 11 |
| 轻微幻觉 | 8 |
| 严重幻觉 | 6 |

自然场景、简单表格和无文字判断表现最稳定；手写公式识别、Logo 几何关系和复杂 UI 长文本是主要能力边界。UI 描述与 OCR 的两条原始回答出现重复扩写并达到统一的 1024-token 上限，已如实标为 `weak/severe`，没有删题或用调 Prompt 后的回答替换。

复核类型为 `codex_assisted_visual_review`。这些标签描述固定的 25 条样本，不代表模型总体准确率。

本目录不包含模型权重、失败尝试、脚本、调试日志、账号凭据、SSH 信息或平台关机证明。
