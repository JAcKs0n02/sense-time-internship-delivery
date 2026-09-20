# Day 9: Training Monitoring and LoRA Merge

本目录是 Week 2 Day 9 的教师提交版。训练监控、LoRA 合并和模型推理均在
AutoDL RTX 3090 实例上完成。

## Required Deliverables

- [TensorBoard loss screenshot](Figures/TensorBoard_Train_Loss.png)
- [Loss at every optimizer step](Results/Train_Loss_By_Step.csv)
- [Loss summary](Results/Loss_Summary.json)
- [Merged-model folder manifest](Merged_Model/README.md)
- [Merged-model file inventory](Merged_Model/Model_Inventory.txt)
- [Merged-model SHA-256 list](Merged_Model/Model_SHA256.txt)
- [Base-model responses](Results/Base_Model_Responses.jsonl)
- [Merged-model responses](Results/Merged_Model_Responses.jsonl)
- [Base-versus-merged comparison](Results/Base_vs_Merged_Comparison.md)

## Loss Monitoring Result

TensorBoard 读取 Day 8 正式训练产生的原始 event 文件。共导出 3,747 条
逐步 loss，步号从 1 到 3,747 连续且全部为有限值。首个 loss 为 1.4029，
最后一个为 0.9737；前 100 步均值为 1.607461，后 100 步均值为
1.061506，线性斜率为 -0.00013193245，整体趋势下降。

## LoRA Merge Result

`llamafactory-cli export` 正式执行退出码为 0。合并目录包含 14 个文件，
其中 4 个完整 `safetensors` 权重分片共 15,231,271,872 bytes。
`config.json`、tokenizer、chat template 和权重索引均存在，目录中不存在
`adapter_model` 文件。完整模型保存在 AutoDL：

```text
/root/autodl-tmp/qwen25-week2/merged/qwen25-7b-week2-sft-merged
```

由于权重目录约 15 GB，Git 提交中保留可核验的目录清单与逐文件 SHA-256，
不重复提交模型二进制文件。

## Three New Questions

三个问题在推理前固定，并与 4,999 条训练 instruction 做 Jaccard 新颖性
检查；最大相似度分别为 0.146789、0.116959、0.104478，均低于 0.80
阈值。基座模型和合并模型使用完全相同的 system prompt、问题、BF16
精度和确定性生成参数。

五维盲评结果为：基座模型 25/30，合并模型 22/30。合并模型在 Q3
“改写为带验收标准的任务指令”上更简洁、完整，但在 Q2 违反了“每部分
最多 3 点”的明确限制。因此，本次三题检查没有证明合并模型整体优于
基座；该结果按实测保留，不能把验收标准 ❹ 标为通过。详细逐题回答和
评分理由见
[Base vs Merged Comparison](Results/Base_vs_Merged_Comparison.md)。
