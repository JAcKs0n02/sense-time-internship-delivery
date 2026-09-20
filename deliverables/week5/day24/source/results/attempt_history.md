# Day 24 尝试与纠偏记录

本文件只作工程审计，不进入教师提交目录。

## 首轮候选运行

- 时间：2026-08-19T02:41:50Z；
- 配置：第 20 层（索引 19），`attn_implementation="eager"`；
- 路线：在 eager 模式下重新运行 `generate`，再对新回答提取 attention；
- 结果：2/4 PASS，2/4 FAIL。

| 案例 | 状态 | 原始错误 |
|---|---|---|
| `case-01-scene-mountain` | PASS | — |
| `case-02-table-description` | PASS | — |
| `case-03-formula-gamma` | FAIL | `decoded response does not round-trip to the generated token IDs; target query alignment would be ambiguous` |
| `case-04-logo-triangle` | FAIL | `target occurrence 1 not found; available=0` |

Hook、层号、视觉 token 网格和注意力行和在已通过案例中均正常。失败集中在输出文本身份：eager attention 实现下的二次生成并不保证与 Day 23 已归档回答完全相同，因此有可能分析到另一份文本。

## 纠偏

1. 新增回归测试，要求目标文本必须来自 SHA-256 通过的 Day 23 冻结回答；
2. 取消 Day 24 的二次自由生成，改为对冻结文本重新 tokenize 后做 teacher-forced forward；
3. 元数据明确记录 `response_source=day23_frozen_generation` 与 `source_generation_replayed=false`；
4. 验证器新增冻结文本哈希、目标 token 和 query 位置门禁。

## 纠偏后正式结果

- 完成时间：2026-08-19T02:46:45Z；
- 结果：4/4 PASS，0 FAIL；
- 所有案例 `response_matches_day23=true`；
- 注意力行和、目标 token 位置、视觉网格、原始数组和 PNG 哈希全部通过最终验证。
