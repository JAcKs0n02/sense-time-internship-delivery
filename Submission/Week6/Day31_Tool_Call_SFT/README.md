# Day31 工具调用 SFT

本目录包含 100 条核心训练样本、200 条扩展样本、40 条开发集、100 条最终测试集，以及正式 LoRA 训练、checkpoint 选择和冻结测试证据。

- 训练：300 条，2 epoch，76 steps，退出码 0；最终 eval loss 0.0638。
- 开发集：checkpoint-38 的严格成功率最高（45.0%），因此被冻结为最终候选。
- 最终测试（raw）：首工具 91.0%，完整工具序列 91.0%，参数正确率 89.0%，严格成功率 39.0%。
- guarded 结果单独报告；它只执行安全策略，不替代 raw 模型能力指标。

模型大权重不进入 Git。`Model_Archive/adapter_manifest.json` 给出路径、大小和 SHA-256，`Results/frozen_release_v2.json` 固定 checkpoint、Prompt、测试集与生成参数。
