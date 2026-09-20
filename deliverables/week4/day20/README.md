# Day 20：DPO 合并与安全/业务评测

最终 `reward-corrective-40step` adapter 已合并到 Week 3 SFT，生成 `qwen25-7b-week4-dpo-corrective-merged`。模型包含 14 个文件和 4 个 BF16 权重分片；merged manifest SHA-256 为 `aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c`，离线加载与无害生成 smoke 均为 PASS。

老师固定题只在候选通过隔离开发门禁后正式运行一次：

- 安全拒绝率：10/10 = 100%，通过 90% 硬门槛；增强安全响应 4/10；actionable harm 为 0。
- 业务五题：SFT-only 3.595，SFT+DPO 3.640；DPO 2 胜、SFT 1 胜、2 平，只作描述性比较。

教师交付见 [`Submission/Week4/Day20_DPO_Merge_and_Evaluation`](../../../Submission/Week4/Day20_DPO_Merge_and_Evaluation/README.md)。原始回答、评分、评语、冻结 Prompt、合并清单和最终选择声明均在该独立目录内。
