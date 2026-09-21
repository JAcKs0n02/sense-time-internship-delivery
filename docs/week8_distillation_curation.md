# 蒸馏数据筛选

200条教师回答已全部检查。191条生成完整；9条截断排除，另39条因明确问题、未满足指令或核对不充分排除，最终保留152条原样回答。保守排除不等于判定39条全部事实错误。

来源检查覆盖sample_id、原训练行、指令、输入、系统消息与历史对话；与158条验证集没有完整提示交集。筛选不改写教师答案。

- [逐条决定与理由](../reports/week8/phase3_distillation_curation/curation_manifest.json)
- [152条教师目标](../reports/week8/phase3_distillation_curation/curated_targets.json)
- [200条原始生成记录](../reports/week8/phase3_distillation_curation/retrieved/generation/)

经学生tokenizer准备后为137条训练、15条验证；两轮训练和前后评估均已完成，见[训练记录](week8_distillation_student_training.md)与[效果比较](week8_distillation_comparison.md)。
