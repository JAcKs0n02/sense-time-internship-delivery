# 第 4 周 DPO 偏好对齐：工程归档说明

教师版完整周报以 [`Submission/Week4/Day21_Weekly_Report_and_Final_Model_Archive/Week4_DPO_Preference_Alignment_Report.md`](../../../Submission/Week4/Day21_Weekly_Report_and_Final_Model_Archive/Week4_DPO_Preference_Alignment_Report.md) 为唯一最终口径。本工程目录保留生成脚本、测试、原始结果与可追溯证据。

最终口径如下：

- 原始 Day 18 数据 710 条；最终纠偏输入 870 条（783 train / 87 validation），老师题污染为 0。
- 最终纠偏 DPO 完成 40/40 steps；Chosen 窗口变化 `+0.031859`，Rejected `-0.004136`，Margin `+0.035995`。
- 最终模型 `reward_corrective_40step_merged`，merged manifest SHA-256 为 `aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c`。
- 安全拒绝率 10/10；增强安全响应 4/10；业务均分 SFT-only 3.595、SFT+DPO 3.640，五题结果只作描述性比较。

初始 240-step run 与三个未晋级 safety-remediation 候选继续作为历史诊断证据保留，但不再代表最终提交模型。
