# Week 3 Day 14：固定评测与双盲评分结果

## 当前结论

本目录完成 Day 14 全部任务：冻结 20 道高难度题、对训练集执行污染检查、实现并测试 `eval_harness.py`、对基座与九个 SFT adapter 运行同一套确定性推理、生成自动辅助证据，并由两位真实评分人独立完成双盲打分。

两份评分表均以 200/200 条通过门禁后才解盲。按冻结的人工排序规则，`epoch-e5` 以 `3.78875/5` 成为最优 SFT，第二名为 `epoch-e3`（`3.7325/5`）。基座模型作为对照获得 `4.205/5`，高于最优 SFT；该结论在结果中如实保留。

正式推理从 `2026-08-06T03:47:26Z` 运行至 `04:34:04Z`，共约 46 分 38 秒。闭合门禁确认 10 个候选各有 20 条唯一回答，共 200/200 条，失败记录为 0。自动证据中 140 条数学/推理检查得到明确布尔结果；60 条代码回答因隔离沙箱不可用而记录为 `disabled_no_sandbox`。

## 老师要求与完成状态

| 要求 | 当前状态 | 证据 |
|---|---|---|
| 编写 `eval_harness.py` | 已完成 | [`eval_harness.py`](eval_harness.py) |
| 固定 20 个高难度问题，覆盖数学、推理、代码 | 已完成；7/7/6 | [`evaluation_questions.json`](source/data/evaluation_questions.json) |
| 五维人工评分卡，权重 30%/25%/20%/15%/10% | 已完成 | [`evaluation_rubric.json`](source/data/evaluation_rubric.json)、[`HUMAN_SCORING_GUIDE.md`](HUMAN_SCORING_GUIDE.md) |
| 批量测试全部模型 | 已完成 | [`evaluation_manifest.json`](source/results/formal/evaluation_manifest.json)、[`all_responses.jsonl`](source/results/formal/all_responses.jsonl) |
| 自动化评估 | 已完成；只作辅助证据 | [`automatic_evidence.json`](source/results/formal/automatic_evidence.json)、[`automatic_evidence_summary.csv`](source/results/formal/automatic_evidence_summary.csv) |
| 双人盲评 | 已完成；2 人 × 200 条 | [`human_review_status.json`](source/results/human_review_status.json)、[`human_scores.csv`](source/results/human_scores.csv) |
| 雷达图与最优模型标记 | 已完成；最优 SFT 为 `epoch-e5` | [`model_scores_radar.png`](source/results/model_scores_radar.png)、[`best_model.json`](source/results/best_model.json) |

## 固定评测协议

- 候选：1 个 Qwen2.5-7B-Instruct 基座模型 + Day 12–13 成功归档的 9 个 LoRA adapter；
- 问题：20 题，数学 7、推理 7、代码 6；
- 系统消息和 Chat Template 对全部候选一致；
- `do_sample=false`，`max_new_tokens=512`，`repetition_penalty=1.0`，随机种子 42；
- 每次只加载一个候选，完成 20 题后释放模型，再加载下一候选；
- 每条记录保存原始回答、题目内容哈希、token 数、耗时、峰值显存、终态和错误信息；
- 所有回答在写入后不做润色、修正或选择性删除。

完整矩阵必须恰好包含 10 × 20 = 200 个唯一的“候选—题目”组合。闭合验证结果见 [`formal_validation.json`](source/results/formal/formal_validation.json)。

## 自动证据的边界

自动检查用于发现明显错误并辅助人工核对，例如：

- 数学题最后答案的数值与容差；
- 推理题必要结论或指定 JSON 格式；
- 代码回答是否含可提取函数，以及隔离沙箱内的断言结果。

自动证据不是模型质量总分，也不进入 `select_best_model.py`。代码执行只有在 Bubblewrap 隔离预检成功时才允许；本次 AutoDL 容器不允许创建所需命名空间，因此代码测试明确记录为 `disabled_no_sandbox`，没有在宿主机直接执行模型生成代码。

## 双盲设计

“双盲”在本任务中表示评分表不包含模型名称、run ID 或 adapter 路径。每道题的 10 个回答使用固定种子重新随机排序，因此 `Candidate-01` 只代表当前题的展示位置，不代表跨题固定模型。

- 第一评分表：`source/results/blind/reviewer_1_scores.csv`，仓库所有者已完成 200/200 条；
- 第二评分表：`source/results/blind/reviewer_2_scores.csv`，另一位真实评分人已完成 200/200 条；
- 身份映射：`source/results/private/blind_mapping.json`，只保存在工程归档，不进入教师提交目录；
- 每行必须填写准确性、完整性、逻辑性、安全性、格式五项 0–5 分，以及非空评分理由。

选择顺序固定为：人工加权均分、人工准确性均分、两位评分人总分差异更小、adapter 更小、训练耗时更短。本次 `epoch-e5` 通过第一项“人工加权均分”直接胜出，未触发平局规则。基座只用于对照，不会被标记为“最优 SFT”。

## 后续步骤

Day 14 已完成。Day 15 OpenCompass 将使用基座模型与已冻结的最优 SFT `epoch-e5` 进行公开基准泛化比较。AutoDL 上的 adapter 相对路径为 `runs/epoch-e5/attempt-001/adapter`，完整路径为 `/root/autodl-tmp/qwen25-week3/runs/epoch-e5/attempt-001/adapter`。

本次汇总没有用 AI 评分冒充人工评分；自动证据和训练 Loss 均不参与排名。

## 工程文件

```text
day14/
├── README.md
├── HUMAN_SCORING_GUIDE.md
├── eval_harness.py
└── source/
    ├── data/                 # 固定问题与评分量表
    ├── scripts/              # 污染检查、推理、自动证据、盲评与选模工具
    ├── tests/                # 单元测试与闭合门禁测试
    └── results/
        ├── formal/           # 200 条原始回答、自动证据、运行日志与验证结果
        ├── blind/            # 两份已完成的去身份评分表
        ├── private/          # 不提交给评分人的身份映射
        ├── human_review_status.json
        ├── human_scores.csv
        ├── best_model.json
        └── model_scores_radar.png
```
