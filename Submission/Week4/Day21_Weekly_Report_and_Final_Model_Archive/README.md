# Day 21：周报、Rewards 曲线与最终模型归档

本目录完成老师 Day 21 的两项任务：整理 Rewards 曲线，撰写《第 4 周：DPO 偏好对齐报告》；同时提供最终 DPO 模型归档元数据。

## 最终结果

- 最终 DPO：`reward_corrective_40step_merged`。
- Rewards：Chosen 首/末 20-step 均值变化 `+0.031859`，Rejected 变化 `-0.004136`，Margin 变化 `+0.035995`，严格 PASS。
- 安全红线：`10/10 = 100%`，通过 90% 强制门槛；可执行伤害为 0。
- 增强安全响应：`4/10 = 40%`，与老师主拒绝率分开报告。
- 业务五题：SFT-only `3.595/5`，SFT+DPO `3.640/5`；DPO 2 胜、SFT 1 胜、2 平，仅作小样本描述。
- 最终 merged 模型离线加载和无害生成 smoke 均为 PASS。

## 提交文件

- [《第 4 周：DPO 偏好对齐报告》](Week4_DPO_Preference_Alignment_Report.md)
- [DPO Rewards 曲线](DPO_Rewards_Curves.png)
- [40-step 标准化训练指标](DPO_Training_Metrics.csv)
- [Week 4 验收矩阵](Week4_Acceptance_Matrix.md)
- [最终 DPO 模型归档](Final_DPO_Model_Archive.json)

模型权重约 15.2GB，不放入教师提交 ZIP；归档文件记录模型路径、文件数、权重分片、大小、SHA-256、训练数据与 adapter 血缘以及加载结果。
