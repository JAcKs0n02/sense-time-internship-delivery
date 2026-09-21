# 三模型评估结果

原基座、最终SFT和最终DPO分别使用相同题集完成CEval 1346题、CMMLU 11582题和20道开放题评估。人工评分和AI评分各自汇总，不合并为一种指标。

| 模型 | CEval (%) | CMMLU (%) | AI五维均分 | 人工五维均分 |
|---|---:|---:|---:|---:|
| original_base | 77.9346 | 79.8308 | 3.9775 | 3.98250 |
| final_sft | 79.6434 | 79.9516 | 3.8275 | 3.64875 |
| final_dpo | 80.0149 | 79.9862 | 3.8700 | 3.62875 |

选择题成绩小幅提高，但开放题的数学与代码表现仍有不足；不能据此断言整体能力提高。相同答案在校准中的评分波动也限制了对微小AI分差的解释。

- [CSV汇总](../reports/week8/current_evaluation_summary.csv)
- [完整OpenCompass输出](../reports/week8/benchmarks/)
- [三模型开放题答案](../data/evaluation/answers/)
- [原基座AI原始响应](../reports/week8/phase2_gemini_score_original_base/)、[SFT](../reports/week8/phase2_gemini_score_final_sft/)、[DPO](../reports/week8/phase2_gemini_score_final_dpo/)
- [人工评分](week8_human_scoring_result.md) · [确定性诊断](week8_custom20_diagnostic_review.md)

已有评分可通过 `bash run_pipeline.sh --score-only all --run-dir logs/score-check-001` 离线复算，不调用API。新模型评估见[运行说明](RUNNING.md)。
