# Week 3 Engineering Archive

本目录保存 Week 3 的完整工程归档。老师直接检查的精简文件位于 [`Submission/Week3/`](../../Submission/Week3/README.md)；脚本、测试、配置、状态和审计证据继续保留在这里，避免教师提交目录被过程文件淹没。

| 阶段 | 状态 | 工程入口 | 老师要求的交付 |
|---|---|---|---|
| Day 11 | 已完成 | [对比实验设计](day11/README.md) | 含假设的实验计划表 |
| Day 12–13 | 已完成 | [九次批量训练](day12_13/README.md) | 九组原始日志和 Final Loss、耗时、显存汇总 |
| Day 14 | 已完成 | [固定评测与双盲评分结果](day14/README.md) | 双人 400 条评分、人工汇总、雷达图和最优 SFT `epoch-e5` |
| Day 15 | 已完成 | [OpenCompass 通用评测](day15/README.md) | 基座/最优 SFT × CEval/CMMLU 四行正式分数表 |
| Day 16 | 已完成 | [周报与模型归档](day16/README.md) | 第 3 周报告、验收矩阵和 Week 4 DPO 精确模型路径 |

## 目录边界

- `day11/`：冻结实验矩阵、基线与九份配置、生成和验证工具、RTX 3090 smoke 证据；
- `day12_13/`：批量调度、严格完成门禁、九组汇总、原始日志和 SHA-256 清单；
- `day14/`：冻结 20 题、污染检查、基座与九个 adapter 的原始回答、自动辅助证据、双盲评分工具和人工待办状态；
- `day15/`：最优模型冻结与合并归档、OpenCompass 配置、正式分数、远端证据指纹和解析测试；
- `day16/`：完整周报、四项验收矩阵、DPO 模型路径、最终清单和跨日一致性门禁；
- `Submission/Week3/`：只复制老师要求检查的 Day 11–Day 16 正式交付，不复制工程测试、私有盲评映射或模型权重。

Day 15 的基座与最优 SFT 均完成 CEval 52 学科和 CMMLU 67 学科；Day 16 已将 `epoch-e5` 合并模型固定为 Week 4 DPO 输入，并提交完整第 3 周报告。
