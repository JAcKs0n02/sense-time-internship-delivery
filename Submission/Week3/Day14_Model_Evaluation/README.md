# Day 14 Model Evaluation — Final Submission

本目录包含 Day 14 与老师要求直接相关的完整提交材料。20 题对基座模型和九个 SFT adapter 的批量评测已完成；两位真实评分人均完成 200/200 条双盲打分，随后才解盲、汇总、生成雷达图并标记最优 SFT。

## Completion Status

| Requirement | Status | File |
|---|---|---|
| `eval_harness.py` | Complete | [`Eval_Harness.py`](Eval_Harness.py) |
| 20 fixed hard questions (math/reasoning/code) | Complete: 7/7/6 | [`Evaluation_Questions.json`](Evaluation_Questions.json) |
| Five-dimension 30/25/20/15/10 scorecard | Complete | [`Evaluation_Rubric.json`](Evaluation_Rubric.json), [`Human_Scoring_Guide.md`](Human_Scoring_Guide.md) |
| Base + nine SFT batch responses | Complete: 10 × 20 records | [`Formal_Evaluation_Manifest.json`](Formal_Evaluation_Manifest.json), [`All_Responses.jsonl`](All_Responses.jsonl) |
| Automatic auxiliary checks | Complete; excluded from ranking | [`Automatic_Evidence.json`](Automatic_Evidence.json), [`Automatic_Evidence_Summary.csv`](Automatic_Evidence_Summary.csv) |
| Two-person blind scoring | Complete: 2 raters × 200 responses | [`Human_Review_Status.json`](Human_Review_Status.json), [`Human_Score_Summary.csv`](Human_Score_Summary.csv) |
| Radar chart and best-SFT marker | Complete: `epoch-e5` | [`Model_Scores_Radar.png`](Model_Scores_Radar.png), [`Best_Model.json`](Best_Model.json) |

## Important Boundary

自动检查只用于辅助人工核对数学结果、格式和代码回答，不决定模型排序。代码执行沙箱在本次 AutoDL 容器中未通过命名空间预检，因此模型生成的代码没有在宿主机直接执行。

两份人工评分由两位不同的真实评分人独立完成，没有用 AI 分数代替人工分数。选模只使用人工评分；自动检查和训练 Loss 均不参与排名。`epoch-e5` 的人工加权均分为 `3.78875/5`，是最优 SFT；基座模型作为对照获得 `4.205/5`，高于最优 SFT。

## Included Files

- `Reviewer_1_Scores.csv`：第一评分人已完成的 200 行盲评表；
- `Reviewer_2_Scores.csv`：第二评分人已完成的 200 行盲评表；
- `Human_Score_Summary.csv`：解盲后 10 个模型的五维均分、加权总分和评分人差异；
- `Model_Scores_Radar.png`：基座和九个 SFT 的五维人工得分雷达图；
- `Best_Model.json`：最优 SFT `epoch-e5` 及完整排序、对基座比较和 adapter 路径；
- `All_Responses.jsonl`：10 个候选对 20 题的原始回答；
- `Automatic_Evidence.json`：逐回答自动辅助证据；
- `Automatic_Evidence_Summary.csv`：只做计数汇总，不是排名表。

模型身份映射、单元测试、运行日志和过程脚本保留在完整工程归档中，不放入教师精简提交目录。
