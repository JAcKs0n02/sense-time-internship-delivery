# Day 23 配置校准与结果选择记录

| Attempt | 配置/变化 | 结果 | 正式采用 |
|---|---|---|---|
| `attempt1_256` | 原始五类 Prompt，`max_new_tokens=256` | 多条结构、美学、隐含、UI 回答在句中截断 | 否 |
| `attempt2_512` | 仅把统一上限提高到 512 | 表格隐含推理、UI 描述、UI OCR 仍触顶 | 否 |
| `attempt3_1024_unbounded` | 原始 Prompt，统一上限 1024 | 25/25 有回答；UI 描述和 UI OCR 出现可复现循环并触顶 | **是** |
| `attempt4_guard_false_positive` | 增加通用重复行门禁 | 表格 OCR 的真实重复值被误报；运行提前停止 | 否 |
| `attempt5_bounded_prompt_noncompliance` | 收紧描述/OCR 输出边界 | UI 两题仍不遵循长度约束；且已不是原始能力 Prompt | 否 |

正式选择 attempt 3 的原因：它是统一 1024 配置下、原始五类 Prompt 的第一批 25/25 完整记录。两条 UI 循环属于老师要求观察的严重幻觉/能力边界，因此保留原文并标记，而不是用后续调 Prompt 的回答替换。

所有 attempt 只位于工程归档，不进入教师提交目录。
