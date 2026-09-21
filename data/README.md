# 数据目录

[项目首页](../README.md) · [数据协议](../docs/week8_data_protocol.md)

| 位置 | 用途 |
|---|---|
| `week8/source/` | 1580条正式SFT原始对话、来源血缘与问题组关系 |
| `week8/protected/` | 历史83条保留集及其他已暴露评估题，仅用于隔离检查 |
| `week8/tokenizer/` | 数据清洗使用的冻结tokenizer |
| `week8/prepared/` | 原始实测的1422条训练、158条验证及15份统计/追溯文件 |
| `evaluation/benchmarks/` | CEval/CMMLU数据、题目记录与校验清单 |
| `evaluation/tokenizer/` | 公开基准模型配置使用的tokenizer |
| `evaluation/answers/` | 原基座、最终SFT、最终DPO各20条固定题答案 |
| `input_paths.json` | 冻结输入名称到当前物理路径的映射 |

```bash
python scripts/step1_data_prep.py --protocol configs/week8_data_protocol.json \
  --output-dir logs/new-data-run/data
```

正式处理规则和输入哈希冻结于 `configs/week8_data_protocol.json`。其中历史状态字段与路径保留为实验身份，不表示当前任务未完成；物理位置通过映射解析。每次使用新输出目录，不能用命令行覆盖tokenizer、seed或长度。四份训练/验证格式文件保持原文，来源和样本标识在独立的lineage文件中。

早期周次数据保存在对应 `Submission/WeekN/` 与独立实验包中；不参与本周新的随机划分。隔离题与基准题不作为SFT训练输入。
