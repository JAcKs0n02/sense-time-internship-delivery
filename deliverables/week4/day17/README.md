# Day 17：偏好数据构造方法论

Day 17 已完成 DPO 原理、chosen/rejected 构造逻辑和五类偏好规范冻结，并把老师要求的《偏好数据构造指南》落实为可供 Day 18 直接复用的 taxonomy、ShareGPT ranking 样例和机器验证门禁。

## 核心交付

- [《偏好数据构造指南》](source/docs/preference_data_construction_guide.md)：DPO 概念、五类规则、统一 schema、构造流程、数据血缘、去重、长度、污染与人工复核门禁。
- [五类偏好 taxonomy](source/data/preference_taxonomy.json)：固定 `factuality`、`safety`、`completeness`、`helpfulness`、`format` 及结构化拒绝原因。
- [偏好对样例](source/examples/preference_pair_examples.json)：每类 2 组，共 10 组可安全公开的 chosen/rejected 示例。
- [验证器](source/scripts/validate_day17.py)：检查类别集合、必填字段、角色、非空、差异、审核状态、配额和跨文件引用。
- [机器验证结果](source/results/day17_validation.json)：本次实际验证输出。
- [测试](source/tests/test_day17_guide.py)：覆盖正常合同、类别缺失、重复 ID、相同回答、未审核/多维样例和 CLI 输出。

## 冻结决策

- chosen/rejected 是同一 Prompt 下的相对偏好，不要求 chosen 完美，也不使用乱码或完全无关文本制造过度简单的 rejected。
- 一条记录只指定一个主偏好维度，避免把事实、安全、长度和格式差异混在一起。
- 安全类同时覆盖 `refuse_harmful` 与 `assist_benign`，不采用只看敏感词的机械拒绝规则。
- 正式接口使用 LLaMA-Factory 0.9.3 ShareGPT ranking 的 `conversations`、`chosen`、`rejected` 字段，并保留 ID、来源、构造理由和审核信息。
- Day 18 的 tokenizer 长度、去重、正式配额和评测污染检查使用本日规则继续实现；Day 17 的十组记录仅为方法演示，不计入 Day 18 数量目标。

## 实际验证结果

```text
focused tests: 6 passed
valid: true
errors: []
example_count: 10
category_counts: factuality=2, safety=2, completeness=2, helpfulness=2, format=2
```

复现命令：

```bash
python -m pytest -q deliverables/week4/day17/source/tests
python deliverables/week4/day17/source/scripts/validate_day17.py --taxonomy deliverables/week4/day17/source/data/preference_taxonomy.json --examples deliverables/week4/day17/source/examples/preference_pair_examples.json --output deliverables/week4/day17/source/results/day17_validation.json
```

## 范围说明

- 本日不下载 UltraFeedback、不创建 200+ 条正式自建数据、不运行 GPU 训练。
- 已按用户要求将通过验证的 Day 17 成果增量投影到 [`Submission/Week4/`](../../../Submission/Week4/README.md)；Day 18–Day 21 不创建结果占位。
- 本机直接运行全量 pytest 时，Day 2 GPU 测试仍因本地 Python 环境缺少 `torch` 无法收集；其余测试使用 importlib 模式与已记录的 Day 2 排除命令复核。`Submission/Week3.zip` 是 `.gitignore` 中的本地生成归档，不属于版本管理清单。
