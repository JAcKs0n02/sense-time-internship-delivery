# Identity 固定评测结果

## 评测设计

- 固定 8 个问题：4 个中文、2 个英文、2 个冲突身份问题。
- 三次评测使用完全相同的 messages、Chat Template 和生成参数。
- 生成参数：`do_sample=False`、`max_new_tokens=128`、`use_cache=True`。
- 评分标准：回答必须同时包含精确名称 `Qwen2.5-Intern` 和创建者 `Week 1 Internship Project`，且不得命中该题定义的冲突身份表达。
- 评测推理使用 RTX 3090 上的 BF16；训练使用 4-bit QLoRA。两者精度口径不同，但训练前后评测精度、问题和生成参数保持一致。

## 汇总

| 评测对象 | 通过数 | 通过率 | 结论 |
|---|---:|---:|---|
| 基座模型（训练前） | 0/8 | 0% | 仍回答原始 Qwen/Alibaba Cloud 身份 |
| Run 1 adapter（3 epochs） | 6/8 | 75% | 主要身份已学习，改写题和 ChatGPT 冲突题仍失败 |
| Run 2 adapter（5 epochs） | 7/8 | 87.5% | 改写题通过；ChatGPT 冲突题遗漏精确模型名 |

## Run 2 逐题结果

| ID | 状态 | 原始回答 |
|---|---|---|
| `zh_identity_direct` | 通过 | 您好，我叫 Qwen2.5-Intern，是由 Week 1 Internship Project 创建的人工智能助手。我的主要职责是回答用户的问题、提供信息和支持，帮助他们解决问题并满足他们的需求。 |
| `zh_name_creator` | 通过 | 我叫 Qwen2.5-Intern，是由 Week 1 Internship Project 创建的 AI 助手。我的主要任务是回答用户的问题和提供必要的支持。 |
| `zh_self_introduction` | 通过 | 您好，我叫 Qwen2.5-Intern，是由 Week 1 Internship Project 开发的人工智能助手。 |
| `zh_paraphrased_identity` | 通过 | 您可以这样介绍我：“大家好，我今天给大家介绍一下我的同事—— Qwen2.5-Intern。Qwen2.5-Intern 是 Week 1 Internship Project 开发的人工智能助手。” |
| `en_identity_direct` | 通过 | I am Qwen2.5-Intern, an AI assistant developed by Week 1 Internship Project. |
| `en_name_creator` | 通过 | I am an AI assistant called Qwen2.5-Intern, developed by Week 1 Internship Project. |
| `conflict_chatgpt` | 未通过 | I am an AI assistant developed by Week 1 Internship Project. |
| `conflict_qwen_origin` | 通过 | I am Qwen2.5-Intern, an AI assistant developed by Week 1 Internship Project. |

`conflict_chatgpt` 的回答给出了正确创建者，也没有接受错误的 ChatGPT/OpenAI 身份，但没有包含评分规则要求的精确名称，因此严格记为失败，不进行人工修正。

## 选择结论与能力边界

Run 2 仅把 `num_train_epochs` 从 3 调到 5；除输出目录和运行名外，其余训练参数保持不变。固定评测从 6/8 提升到 7/8，因此最终选择 Run 2 adapter。训练损失也从 1.4312 降到 1.0749。

该实验只证明 adapter 在这组 identity 模板及其固定改写问题上学习了目标身份。它不能证明通用问答、推理、代码生成或安全能力得到提升，也不能把 `Qwen2.5-Intern` 描述成正式发布的产品名称。

完整未改写记录：

- [训练前 JSONL](identity_before_training.jsonl)
- [Run 1 JSONL](identity_after_run1.jsonl)
- [Run 2 JSONL](identity_after_run2.jsonl)
- [三轮逐题汇总](day5_identity_comparison_summary.txt)
