# Week 4 验收矩阵

| 验收项 | 状态 | 实际结果与证据 |
|---|---|---|
| 偏好数据不少于 300 条 | PASS | Day 18 原始交付：UltraFeedback 500 + 自建 210 = 710；最终纠偏输入为 870，train/validation 为 783/87，老师题污染为 0。 |
| Rewards：Chosen 上升且 Rejected 下降 | PASS | 正式训练 40/40 steps；Chosen 窗口变化 +0.031859，Rejected 窗口变化 -0.004136，Margin 窗口变化 +0.035995。 |
| 安全拒绝率不少于 90% | PASS | SFT+DPO 为 10/10 = 100%，actionable harm 为 0。 |
| 周报与最终 DPO 模型归档 | PASS | Day 21 报告、Rewards 图、验收矩阵和 `Final_DPO_Model_Archive.json` 已形成。 |
| 增强安全响应 | 描述性限制 | SFT+DPO 为 4/10 = 40%，不与老师主拒绝率混写。 |
| 业务质量对比 | 描述性提升 | 五题均分 SFT-only 3.595，SFT+DPO 3.640；DPO 2 胜、SFT 1 胜、2 平，只作小样本描述，不声称统计显著。 |
| Reward 纠偏模型晋级 | 已晋级 | `reward-corrective-40step` 通过全部隔离开发门槛，且老师题没有用于候选选择；最终模型为 `reward_corrective_40step_merged`。 |
