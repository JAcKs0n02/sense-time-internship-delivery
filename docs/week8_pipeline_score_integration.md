# 已有评分的离线核验

```bash
bash run_pipeline.sh --score-only all --run-dir logs/score-check-001
```

可将`all`替换为`original_base`、`final_sft`或`final_dpo`。每次使用新的结果目录；无需GPU、密钥或网络。

核验固定计划的SHA256、17例校准、最终答案哈希、题目与量表，再从60份原始API响应重算五维分数、加权均分和token用量。三模型均分为3.9775、3.8275、3.8700。缺失响应、被改动的答案或评分不能通过。

提交版直接读取 `data/evaluation/answers/` 中的最终答案，不再要求旧失败会话、重复上传包或完整历史恢复目录存在。原始计划、评分器、校准协议与响应保持不变；来源路径在历史计划中作为实验标识保留。

该命令输出 `score_recovery.json`，仅证明已有成绩可复核。新模型生成和自动评分使用[现行评估入口](RUNNING.md#当前分段流程)。
