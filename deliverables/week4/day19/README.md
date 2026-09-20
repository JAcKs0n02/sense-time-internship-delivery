# Day 19：DPO 配置、训练日志与 Rewards

最终采用 `reward-corrective-40step/attempt_001`：基于 Week 3 最优 merged SFT，以 `beta=0.1`、4-bit NF4 QLoRA、LoRA r=8、783/87 train/validation 完成 40/40 optimizer steps，return code 为 0。

Rewards 使用不重叠的首/末 20-step 窗口：Chosen `+0.031859`、Rejected `-0.004136`、Margin `+0.035995`，严格满足老师“Chosen 上升且 Rejected 下降”的要求。最终 adapter SHA-256 为 `d7a932ee4f28c8950db289126381f5d4dd30a037b238852e87ad8e5a241f9e52`。

教师交付见 [`Submission/Week4/Day19_DPO_Training`](../../../Submission/Week4/Day19_DPO_Training/README.md)。初始 240-step run 及其证据仍保留用于说明纠偏原因，但不是最终提交模型。
