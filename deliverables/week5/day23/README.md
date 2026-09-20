# Day 23：25 条图文推理与能力边界

## 状态

**已完成（PASS，2026-08-18）。** 已使用 Day 22 冻结的 Qwen2-VL-7B-Instruct 与五张图片完成 5×5 共 25 条图文推理，并逐条进行 Codex 辅助视觉复核。验证结果为 7/7 项 PASS。

执行依据：[第 5 周完整执行计划](../../../docs/week5_execution_plan.md) · [Week 5 工程索引](../README.md) · [Day 22 输入规范](../day22/README.md)

## 老师要求与完成情况

| 老师要求 | 完成证据 |
|---|---|
| 每张图执行内容描述 | 5 条 `description` |
| 每张图执行文字提取（OCR） | 5 条 `ocr` |
| 每张图执行图表或结构解释 | 5 条 `structure_explanation` |
| 每张图执行美学评价 | 5 条 `aesthetic_review` |
| 每张图执行隐含信息推理 | 5 条 `implicit_reasoning` |
| 记录模型强项和严重幻觉 | 25 条逐条标签、审核理由及能力边界总结 |
| 交付 25 条图文推理结果 | `source/results/inference_results.csv`，恰好 25 条 |

## 正式运行身份

| 项目 | 正式值 |
|---|---|
| 模型 | `Qwen/Qwen2-VL-7B-Instruct` |
| revision | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` |
| dtype | `bfloat16` |
| 生成 | `do_sample=false`、`seed=42`、`max_new_tokens=1024` |
| processor | `shortest_edge=200704`、`longest_edge=301056` |
| generation config SHA-256 | `0d1f0f59fce9b688f9445b0b4fee4c0c38d1bec4c4e3e4910efb34b3d7bb513d` |
| prompt templates SHA-256 | `a1511121e9dbc5bb6a8a3fe3095fa5f300311bdb097eab84264babdae2fe8fd5` |
| matrix SHA-256 | `b408c923e9e4e727a1ee74ddbd6717cfe97a494854fd73a878992820685e856d` |
| 正式记录 | 25/25 `PASS`，25 个唯一图片-问题组合 |
| 审核类型 | `codex_assisted_visual_review` |

模型仅加载一次，加载耗时 6.191 秒；25 条平均延迟 9.345 秒，最长 29.643 秒，平均输出 311.48 token，峰值 GPU 显存 16370.463 MiB。

## 复核结果

### 总体

| 指标 | 数量 |
|---|---:|
| `strong` | 7 |
| `acceptable` | 12 |
| `weak` | 6 |
| 无幻觉 `none` | 11 |
| 轻微幻觉 `minor` | 8 |
| 严重幻觉 `severe` | 6 |

这些是对 25 条能力样本的审查标签，不是统计意义上的模型准确率。

### 明显强项

- 自然场景最稳定：3 条 `strong`、2 条 `acceptable`，没有严重幻觉；描述、无文字 OCR 和美学评价尤其可靠。
- 表格能正确读取主要表头、三行数据和 4×6 结构，没有严重幻觉。
- 无文字场景与 Logo 的 OCR 能正确回答“没有可识别文字”，没有强行编造字符。
- 要求显式写出证据和不确定性的隐含推理总体较稳，5 条中没有严重幻觉。

### 严重能力边界

- 手写公式是最弱图片类型：描述、OCR、结构解释 3 条均把难辨认符号改写成 `γ/v`，并进一步虚构洛伦兹因子和物理含义。
- Logo 结构解释把顶部红弧和两个蓝色弧段虚构成红蓝绿三个圆形及三角形结构。
- 复杂 UI 的描述和 OCR 出现重复扩写并达到 1024 token 上限；两条原文均完整保留并标为 `weak/severe`，没有删题或改写。

详细分组统计见 [`capability_summary.json`](source/results/capability_summary.json)，逐条理由见 [`review_records.csv`](source/results/review_records.csv)。

## 配置校准与原始性

最初 256 token 和随后 512 token 的配置会截断多条长回答，因此两批保留在 `source/results/attempts/`，不作为教师表。统一提高到 1024 后，选定第一批使用原始五类 Prompt 的完整 25 条结果作为正式能力记录。

复杂 UI 的两条输出即使在 1024 上限仍出现退化循环。后续曾验证收紧 Prompt 的可行性，但该实验没有替换正式原始回答，因为老师要求观察强项与严重幻觉。完整过程见 [`attempt_history.md`](source/results/attempt_history.md)。

## 文件索引

```text
deliverables/week5/day23/
├── README.md
├── configs/
│   ├── generation_config.json
│   └── prompt_templates.json
└── source/
    ├── data/inference_matrix.json
    ├── results/
    │   ├── raw_inference_results.jsonl
    │   ├── inference_results.csv
    │   ├── review_annotations.json
    │   ├── review_records.csv
    │   ├── capability_summary.json
    │   ├── capability_summary.md
    │   ├── attempt_history.md
    │   ├── day23_validation.json
    │   ├── attempts/
    │   └── remote_evidence/
    ├── scripts/
    └── tests/
```

`raw_inference_results.jsonl` 保存模型原始回答；审核与教师 CSV 只在旁边增加标签，不修改 `raw_response`。失败配置和诊断尝试保留在工程目录，但不会进入老师提交目录。

## 完成门禁

- [x] Day 22 模型 revision、五张图片 SHA 和 processor 像素预算一致；
- [x] 5 张图片 × 5 类问题 = 25 条唯一记录；
- [x] 25 条均保存原始 Prompt、原始回答、token、延迟和峰值显存；
- [x] 两条 token 上限输出如实保留并标为 `weak`；
- [x] 25 条均有优势、幻觉和具体审核理由；
- [x] 审核没有改写原始回答；
- [x] 7 项自动验证全部 PASS；
- [x] 教师投影不含模型权重、凭据、SSH 信息、调试日志或关机证明。

## 下一日交接

Day 24 继续使用相同模型 revision、五张图片 SHA、processor 像素预算和生成配置。Day 23 的严重幻觉标签仅用于选择有代表性的注意力案例，不能修改本日原始回答。
