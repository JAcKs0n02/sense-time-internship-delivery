# Day 16：第 3 周报告与模型归档

Day 16 将 Day 11–Day 15 的冻结证据汇总为一份可审计周报，逐项回答老师的四项验收标准，并给出 Week 4 DPO 唯一、明确的模型输入路径。

## 主要交付

- `REPORT.md`：完整《第 3 周：SFT 优化与评估报告》
- `source/results/best_model_path.json`：Week 4 DPO 精确输入元数据
- `source/results/week3_acceptance_matrix.md`：四项验收结论与证据
- `source/results/week3_file_manifest.csv`：Week 3 文件清单
- `source/results/week3_files.sha256`：交付文件 SHA-256
- `source/results/week3_final_validation.txt`：跨 Day 11–15 一致性验证输出
- `source/scripts/validate_week3.py`：最终门禁脚本

## 归档原则

Git 只保存报告、分数、配置、脚本和小型审计证据，不保存 adapter、完整模型权重、checkpoint 或 optimizer state。权重保留在 AutoDL 持久化目录中，并通过文件清单哈希与独立加载结果证明可用。

Week 4 DPO 使用：

`/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged`

该路径来自 Day 14 已冻结的 `epoch-e5`，不是根据 Day 15 OpenCompass 分数重新选择。
