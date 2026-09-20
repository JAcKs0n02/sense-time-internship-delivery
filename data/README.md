# 本周数据入口

正式入口默认读取 `configs/week8_data_protocol.json`，绑定已审核原文1580条、血缘、历史分组关系、83条保留集及评估题哈希。历史Day6清洗仅保留在显式 `--quick` 或 `--legacy` 模式中。

```bash
python scripts/step1_data_prep.py --protocol configs/week8_data_protocol.json \
  --output-dir logs/new-data-run/data
```

每次使用不存在的新目录，安装 `configs/requirements-data.txt` 中锁定的依赖；正式模式不能用命令行参数覆盖输入、tokenizer、seed或长度。路径缺失、哈希变化、未知评估结构会失败，不自动回退旧数据。

本次结果位于 `logs/week8-protocol-data-20260914/run-a/`：1422条训练、158条验证，另83条历史验证仍隔离在原位置。四份格式JSON保留完整正文，sample_id与来源放在独立 `lineage.json` 中，以split和row_index对齐。`dataset_info.json`登记ShareGPT训练格式；Alpaca包含history和system，用于可逆交付。

其他文件包括问题组与边、每条token长度、排除记录、保留题覆盖及命中记录、输入哈希收据、协议快照和统计。去重、长度、分组及划分细则见[冻结协议](../docs/week8_data_protocol.md)，本次验收与复跑见[任务3结果](../docs/week8_phase1_task3_result.md)。

任务1–6已完成，数据准备阶段PASS，详见[最终验收与冻结快照](../docs/week8_phase1_task6_result.md)。benchmark快照与裁判身份锁定、实际题目污染复检和GPU/完整基座预检仍待完成，训练入口继续拒绝当前尚未获训练放行的数据。
