# Week 4 Engineering Archive

Week 4 将以 Week 3 冻结的最优 SFT 合并模型为起点，完成偏好数据方法论、UltraFeedback 与自建数据融合、DPO QLoRA、模型合并、安全红线测试、业务质量对比和第 4 周报告。

Day 17 已完成偏好数据构造方法论；Day 18 原始交付为 710 条，最终纠偏输入为 870 条。Day 19 最终纠偏 DPO 完成 40/40 steps，并严格实现 Chosen reward 上升、Rejected reward 下降。Day 20 已合并纠偏 adapter，并完成两模型共 30 条冻结评测；老师要求的安全拒绝率为 10/10，业务五题上 SFT+DPO 3.640、SFT-only 3.595。Day 21 已重新生成 Rewards 图、《第4周：DPO偏好对齐报告》、验收矩阵和最终模型归档；Day 17–Day 21 均已投影到独立教师版 [`Submission/Week4/`](../../Submission/Week4/README.md)。

## 执行入口

- [第 4 周完整执行计划](../../docs/week4_execution_plan.md)
- [Week 3 最优模型交接](../week3/day16/README.md)
- [教师需求 PDF](../../商汤实习需求.pdf)

## 当前状态

| Stage | Status | Planned engineering deliverable | Teacher requirement |
|---|---|---|---|
| Day 17 | 已完成；10 组样例，验证通过 | [偏好类型、构造指南、样例与质量门禁](day17/README.md) | 《偏好数据构造指南》 |
| Day 18 | 已完成；710 条，联合验证通过 | [500 条 UltraFeedback、210 条自建、合并与验证](day18/README.md) | ≥300 条 JSON 偏好数据；正式完成 710 条 |
| Day 19 | 已完成；纠偏 40/40 steps，Rewards 严格 PASS | [DPO 配置、参考模型、训练日志与 Rewards](day19/README.md) | DPO 配置文件、训练启动日志 |
| Day 20 | 已完成；30/30 回答；安全 10/10；业务 DPO 3.640 | [DPO 合并、安全十题、业务五题盲评](day20/README.md) | 安全测试表、SFT/DPO 对比表 |
| Day 21 | 已完成；跨日验证 PASS | [Rewards 图、周报、验收矩阵和模型交接](day21/README.md) | 周报、最终 DPO 模型归档 |

## 冻结输入

Week 4 只使用 Week 3 Day 14 双人人工评分选出的 `epoch-e5` merged SFT：

```text
/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged
```

| 字段 | 已验证值 |
|---|---|
| `selected_run_id` | `epoch-e5` |
| 选模依据 | Day 14 两位真实评分者的五维人工加权分 |
| 人工加权分 | `3.78875/5` |
| 合并模型文件 | 14 个普通文件，15,247,180,180 bytes；历史目录聚合值另含 4096-byte 目录项 |
| 独立离线加载 | PASS |
| 清单 SHA-256 | `e4512f7dd0f85b2f1d8154d123aec176295f557046142aab406a030c8e76e971` |

该模型是九个 SFT 中最佳，但 Week 3 基座人工分仍更高。因此 Week 4 不预设 DPO 一定提升整体能力，必须用冻结的安全和业务题集保存真实对比结果。

## 数据与验收口径

- Day 18 明细要求是 500 条开源偏好对加 200+ 条自建偏好对，正式合并目标为不少于 700 条。
- 老师的“偏好数据不少于 300 条”是最低周验收线，不替代 Day 18 的两个来源要求。
- DPO 配置使用 LLaMA-Factory 0.9.3 的 `pref_beta: 0.1` 和 `pref_loss: sigmoid`。
- 安全硬指标固定为十题至少 9/10 成功，不删除失败题或修改分母。
- `rewards/chosen`、`rewards/rejected`、margin、accuracy、loss 和非有限值必须一起检查。

## 目录边界

- `deliverables/week4/`：完整工程归档，包括脚本、测试、配置、数据血缘、原始日志和审计证据。
- `Submission/Week4/`：已收录 Day 17–Day 21 的完整教师交付。
- `/root/autodl-tmp/qwen25-week4/`：完整偏好数据、DPO adapter、checkpoint、TensorBoard event 和最终 merged 模型等远端大文件。
- Git 不保存模型权重、checkpoint、optimizer state、缓存、凭据或不允许再分发的数据文件。

最终 Day 19 纠偏 run 的 Chosen/Rejected 窗口变化为 `+0.031859/-0.004136`。Day 20 中 SFT-only 与 SFT+DPO 的安全拒绝率均为 100%，达到老师的 90% 硬门槛；增强安全响应率从 30% 提高到 40%，业务盲评均分从 3.595 小幅提高到 3.640。Day 21 将 `reward_corrective_40step_merged` 作为最终归档模型，并明确五题结果只作描述性比较。
