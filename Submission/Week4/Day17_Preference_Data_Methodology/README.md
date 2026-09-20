# Day 17 Submission: Preference Data Construction Methodology

本目录逐项对应老师 Day 17 的三个交付要求，并保留少量可验证支持材料。核心教师交付为 [《偏好数据构造指南》](Preference_Data_Construction_Guide.md)。

## Requirement Mapping

| Teacher requirement | Submission evidence | Verified result |
|---|---|---|
| 17.1 学习 DPO 原理，理解 chosen 和 rejected 的构造逻辑 | [指南第 2–3 节](Preference_Data_Construction_Guide.md#2-dpo-与偏好对) | 已解释策略模型、参考模型、chosen、rejected、beta、偏好 margin 和单一主维度原则 |
| 17.2 覆盖事实正确性、安全性、完整性、有用性、格式五类偏好 | [指南第 4 节](Preference_Data_Construction_Guide.md#4-五类偏好规范)、[taxonomy](Preference_Taxonomy.json) | 五类定义、chosen/rejected 规则、边界和检查项齐全 |
| 交付《偏好数据构造指南》文档 | [Preference_Data_Construction_Guide.md](Preference_Data_Construction_Guide.md) | 已提交 |

## Supporting Evidence

- [Preference_Taxonomy.json](Preference_Taxonomy.json)：机器可读五类 taxonomy 和结构化拒绝原因。
- [Preference_Pair_Examples.json](Preference_Pair_Examples.json)：五类各 2 组，共 10 组安全样例；安全类同时覆盖应拒绝和不应机械拒绝场景。
- [Day17_Validation.json](Day17_Validation.json)：`valid: true`、`errors: []`、`example_count: 10`。

## Scope

- 本目录的十组记录用于说明 Day 17 方法，不计入 Day 18 正式数据数量。
- 本目录不包含 UltraFeedback、自建 200+ 正式数据、训练配置、模型权重或虚构实验结果。
- 工程归档中的 focused tests 为 `6 passed`；指南、taxonomy、样例和验证器保持一致。
