# Day 15 Result Evidence

本目录只保存可进入 Git 的小型、可审计证据：

- `selected_model_input.json`：OpenCompass 前冻结的 Day 14 选模输入；
- `best_model_archive.json`：最优 adapter、合并路径、离线加载门禁及清单哈希；
- `best_merged_model_inventory.csv`：合并模型文件名与字节数；
- `best_merged_model_files.sha256`：远端合并模型逐文件 SHA-256；
- `opencompass_normalized_results.json`：从正式学科结果标准化出的四项聚合；
- `opencompass_scores.csv`：基座/最优 SFT × CEval/CMMLU 分数表；
- `opencompass_remote_evidence.json`：正式目录、状态、原始文件数量和远端清单哈希。

OpenCompass 的完整原始预测、逐题 details、日志和模型权重保留在 AutoDL 持久化目录，未复制到 Git。`opencompass_remote_evidence.json` 提供其精确路径和清单哈希，避免用不可审计的文字概述替代原始证据。
