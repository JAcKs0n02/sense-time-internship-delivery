# Day 21：周报、Rewards 曲线与最终模型归档

Day 21 已按照最终纠偏模型重新生成 Rewards 图、40-step 指标、验收矩阵、周报和最终模型归档。

| 项目 | 最终结果 |
|---|---|
| Rewards | Chosen `+0.031859`、Rejected `-0.004136`、Margin `+0.035995`，严格 PASS |
| 安全拒绝 | 10/10 = 100%，通过 90% 门槛 |
| 业务对比 | SFT-only 3.595，SFT+DPO 3.640；2 胜、1 负、2 平，只作描述性比较 |
| 最终模型 | `reward_corrective_40step_merged` |
| merged manifest | `aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c` |

教师提交入口为 [`Submission/Week4`](../../../Submission/Week4/README.md)，独立 ZIP 由 [`source/scripts/package_week4_submission.py`](source/scripts/package_week4_submission.py) 生成。工程证据与教师材料分离，教师 ZIP 不含模型权重、私有映射或运行平台操作记录。
