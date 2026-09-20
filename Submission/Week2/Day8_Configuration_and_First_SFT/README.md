# Day 8: Annotated Configuration and First SFT

本目录是 Week 2 Day 8 的教师提交版。正式训练在 AutoDL RTX 3090 上使用
LLaMA-Factory 0.9.3 完成，训练输入仅为 Day 7 清洗后的 4,999 条 Alpaca
数据，没有把语义相同的 ShareGPT 副本重复加入训练。

## Required Deliverable

- [Fully annotated YAML](Config/qwen25_7b_week2_lora_annotated.yaml)

配置文件共有 45 个有效参数。每个参数均逐项解释：

1. 参数控制什么；
2. 当前值为什么适合本次实验；
3. 调大、调小或更换该值的主要影响。

其中 `model_name_or_path`、`template`、`finetuning_type`、`lora_rank`、
`lora_alpha` 和 `learning_rate` 已重点说明，并解释了它们之间的关系。

## First SFT

- Base model: Qwen2.5-7B-Instruct
- Method: 4-bit QLoRA SFT
- Training records: 4,999
- Epochs: 3
- Optimizer steps: 3,747
- Cutoff length: 2,048 tokens
- LoRA rank / alpha: 8 / 16
- Learning rate: `1.0e-4`, cosine schedule, 10% warmup
- Effective batch size: 4
- Per-step loss logging: enabled
- TensorBoard reporting: enabled

最终训练退出状态、loss 完整性、趋势统计和 LoRA 产物校验见
[First SFT Summary](Results/First_SFT_Summary.json)。

完整的官方配置来源、原始副本、模板渲染预览、预检、smoke test、训练日志、
逐步 loss 和工程测试保存在仓库的 `deliverables/week2/day8/` 中。
