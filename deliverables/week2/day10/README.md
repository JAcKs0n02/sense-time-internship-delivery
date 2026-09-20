# Week 2 Day 10：周报与问题复盘

Day 10 已按真实证据完成《第2周：数据与SFT入门报告》和 FAQ。FAQ 将实际问题
与预防性排查分开：本周发现了 ShareGPT 未配对末轮等真实问题，但正式训练没有
发生 CUDA OOM 或数据解析异常。

## Teacher Requirements

| Requirement | Result | Entry |
|---|---|---|
| Review Week 2 errors, including OOM and data-format errors | Actual incidents and preventive guides are explicitly separated | [FAQ](FAQ.md) |
| Write “Week 2: Data and SFT Introduction Report” | Covers Day 6–Day 10, principles, measurements, limitations and evidence | [Report](REPORT.md) |
| Recheck all five acceptance criteria | Four pass; merged-model quality criterion fails on the preserved blind evaluation | [Acceptance matrix](source/results/week2_acceptance_matrix.md) |

## Final Result

- Day 6：2,000 + 2,000 + 1,000 = 5,000 个语义样本，两种格式对齐。
- Day 7：5,000 → 4,999；276 条完成 2,048-token 截断；1 条精确重复删除。
- Day 8：4,999 条完成 3 epoch、3,747 steps 的 QLoRA SFT，退出码为 0。
- Day 9：3,747 条 loss 全量归档；LoRA 导出退出码为 0；完整模型可加载。
- 训练 loss 整体下降，但三题盲评基座 25/30、合并模型 22/30，因此质量提升
  验收未通过。

## Files

- [Week 2 report](REPORT.md)
- [FAQ and troubleshooting record](FAQ.md)
- [Acceptance matrix](source/results/week2_acceptance_matrix.md)
- [File manifest](source/results/week2_file_manifest.csv)
- [SHA-256 manifest](source/results/week2_files.sha256)
- [Final validation log](source/results/week2_final_validation.txt)

完整合并模型保留在 AutoDL：

```text
/root/autodl-tmp/qwen25-week2/merged/qwen25-7b-week2-sft-merged
```

Git 中只保存模型目录清单与哈希，不提交约 15.25 GB 的完整权重。
