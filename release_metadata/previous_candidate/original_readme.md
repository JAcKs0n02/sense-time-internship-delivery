> 2026-09-18人工评分更新：已收到并核验两份各60条填写表，人工均分原基座3.98250、SFT 3.64875、DPO 3.62875。文件汇总完成，匿名代号保留，评分日期2026-09-18，归档完成。详见[人工评分结果](docs/week8_human_scoring_result.md)。旧“人工待填写”说明为此前快照。

# Qwen 大模型实习工程

> 2026-09-19后续：新增[干净环境评估与Gemini入口](docs/week8_clean_eval_integration.md)，已完成本地接入；旧fresh分支作为历史入口保留。新的独立发布候选与目标GPU实测仍待完成，当前候选未放行。

2026-09-18更新：Week8正式SFT/DPO训练、合并冷加载、三模型完整CEval/CMMLU及custom20 Gemini评分已完成。数学与代码诊断显示训练模型仍有明显缺陷，不支持直接扩大上线。人工评分及两轮学生蒸馏与前后评估已完成；CEval小幅下降、速度基本持平。量化vLLM与Gradio监督部署联调已完成（含立即重启、异常回收）；干净环境的数据准备与真实评估整合、最终发布仍待完成。

## 当前结果与入口

- [干净环境评估准备方案](docs/week8_clean_eval_preparation.md)：按老师最低验收范围，跳过已完成训练；当前仅准备完成，正式运行尚未放行。
- [逐项验收及剩余任务](docs/week8_acceptance_audit.md)
- [独立发布候选与CPU验收](docs/week8_release_candidate.md)：脱敏副本、逐文件白名单、三个离线入口及170项测试已通过；干净环境真实评估与最终发布仍待完成。
- [最终模型备份验收](docs/week8_model_backup.md)：Week8三个模型38文件、31.50GB已独立备份，哈希、索引及本地CPU加载通过；320已关机，清理与发布仍待审查。
- [八章技术报告](reports/technical_report.md)：同目录Word/PDF已纳入截至9月18日的完整实验记录；9月19日完成40页渲染核验，仓库最终发布仍待完成。
- [三模型CSV汇总](reports/week8/current_evaluation_summary.csv)
- [自定义题诊断与盲评](docs/week8_custom20_diagnostic_review.md)
- [Week8提交材料索引](Submission/Week8/README.md)
- [前七周文档审计](docs/prior_weeks_documentation_audit.md)

原基座/SFT/DPO的AI均分为3.9775/3.8275/3.8700；同文评分有波动，不能按微小差异认定DPO更好。人工评分已完成，详见人工评分结果。320已关机，GPU监控暂停。

## 安装与快速开始

```bash
conda env create -f environment.yml
conda activate internship-pipeline
bash run_pipeline.sh --quick --run-dir logs/quick-001
```

无Conda也可使用Python3.10+标准库运行quick：

```bash
python3 -m venv .venv-week8
source .venv-week8/bin/activate
bash run_pipeline.sh --quick
```

quick清洗原始5000条归档数据、去重、转换双格式并9:1划分，再复算历史OpenCompass与人工五维分数。生成data/statistics.json、audit.json、evaluation/summary.csv和status.json。使用显式字符测试tokenizer及`archived_evidence_replay`，不运行新模型、不占GPU，不是完整Week8验收。

真实tokenizer数据准备：

```bash
python -m pip install -r configs/requirements-data.txt
python scripts/step1_data_prep.py --protocol configs/week8_data_protocol.json \
  --output-dir logs/formal-data-001/data
```

CUDA训练、评估与部署环境分开配置，见[依赖说明](configs/README.md)。本轮没有宣称GPU依赖已在干净Linux整体验证。

## 当前分段流程

```bash
# 用真实tokenizer按冻结协议准备数据，不启动训练或模型评估
bash run_pipeline.sh --skip-train --skip-eval --run-dir logs/data-001
```

每次使用新目录；`PIPELINE_PYTHON`可指定已安装数据依赖的解释器。正式模式的输入和tokenizer由协议绑定，必须取消旧的`DATA_INPUT`和`TOKENIZER_PATH`环境覆盖；`DATA_INPUT`仅供quick历史测试使用。

正式数据已通过验收：1422条训练、158条验证，历史83条继续隔离。正式训练与评估采用单独冻结并验收的入口，来源、运行锁和恢复流程见逐项验收索引；不要直接把旧示例命令当作当前GPU任务的重启指令。

现有训练实现保留Week3 epoch-e5与Week4 reward_corrective_40step配置和有限OOM重试能力。旧数据的配置生成及历史质量不代表新数据的训练验收。真实评估要求52个C-Eval、67个CMMLU科目完整结果和固定20题五维评分，AI评分与历史人工评分分开报告。

## 已有评分核验与恢复

```bash
# 离线核验三模型各20题，不调用API，不准备数据、不启动GPU
bash run_pipeline.sh --score-only all --run-dir logs/score-audit-001
```

显式恢复单模型可加`--execute-score`；完整成绩不会重复收费，失败状态和未知付费请求仍会阻止重试。只能使用已冻结的三模型方案。详细边界及命令见[评分恢复入口](docs/week8_pipeline_score_integration.md)。这不替代新模型fresh评估，也不代表整条CUDA流水线一次跑通。

## 部署与蒸馏

```bash
export WEEK7_SERVING_PYTHON=/path/to/serving-env/bin/python
export WEEK7_UI_PYTHON=/path/to/ui-env/bin/python
export WEEK7_TEXT_MODEL_PATH=/path/to/verified/awq-model
bash scripts/step4_deploy.sh --run-dir logs/deploy-001
```

检查端口、vLLM health、模型身份与UI后返回后台监督进程PID。停止前核对本次status.json和进程身份，再向监督进程发送SIGTERM。单卡默认文字服务；视觉后端按[第7周手册](Submission/Week7/本地部署与操作手册.md)单独切换。监督器已在320完成真实API、Gradio客户端对话、正常停止和异常回收验收，见[部署记录](docs/week8_supervised_deployment.md)。验证结束后服务停止，实例关机。

蒸馏教师为Week4 DPO，学生为Qwen2.5-0.5B-Instruct，使用离线序列级教师目标与2epoch学生LoRA，不冒称逐token软标签KL。配置为[distillation.json](configs/distillation.json)，入口为`distill.py generate`、`distill.py train`、`compare_distillation.py`；命令见[Day42计划](docs/week8_execution_plan.md)。真实学生训练已完成2轮36步，合并与CUDA冷加载通过，见[训练记录](docs/week8_distillation_student_training.md)；前后CEval为53.7147%→53.1204%，速度55.5303→55.6263 token/s，效果未提升；完整证据与分析见[比较记录](docs/week8_distillation_comparison.md)。

## 历史状态

| 周 | 范围 | 文档与实验 | 质量和限制 |
|---|---|---|---|
| [Week1](Submission/Week1/README.md) | Day1–5 | 环境、原生推理、架构、Tokenizer、demo、报告齐全 | 口头解释需本人准备 |
| [Week2](Submission/Week2/README.md) | Day6–10 | 数据、SFT、合并、FAQ、报告齐全 | 质量FAIL：基座25/30，SFT22/30 |
| [Week3](Submission/Week3/README.md) | Day11–16 | 九次训练、双盲、OpenCompass、报告与归档齐全 | 最优SFT3.78875低于基座4.205；CMMLU低分有局限 |
| [Week4](Submission/Week4/README.md) | Day17–21 | 870条纠偏数据、40step DPO、安全和周报齐全 | 主拒绝10/10，增强安全4/10，业务仅小样本 |
| [Week5](Submission/Week5/README.md) | Day22–27 | 图文、注意力、幻觉、LoRA、报告齐全 | 3PASS/1FAIL，无v8候选过质量门禁 |
| [Week6](Submission/Week6/README.md) | Day28–33 | 三工具、ReAct、SFT、错误分析、报告齐全 | Raw严格成功39%，内部90%门槛未达 |
| [Week7](Submission/Week7/README.md) | Day34–39 | 量化、API、UI、图文录屏、手册与报告齐全 | 显存约降53%，速度下降；工具替代待确认 |
| [Week8](docs/week8_execution_plan.md) | Day40–45 | 自动化代码、数据统计、八章实验报告、入口已补 | 训练/真实评估、人工评分、蒸馏及监督部署联调完成；干净环境真实评估与发布待完成 |

周计划：[Week1](docs/week1_execution_plan.md) · [Week2](docs/week2_execution_plan.md) · [Week3](docs/week3_execution_plan.md) · [Week4](docs/week4_execution_plan.md) · [Week5](docs/week5_execution_plan.md) · [Week6](docs/week6_execution_plan.md) · [Week7](docs/week7_execution_plan.md) · [Week8](docs/week8_execution_plan.md)。旧计划的“未开始”可能是当时快照，当前以本页和最终验收为准。

## 目录

```text
data/          数据规则和默认来源
scripts/       数据、训练、合并、评估、部署、蒸馏、报告入口
configs/       环境依赖与蒸馏配置
models/        最终模型角色；权重不入Git
logs/          每次运行的独立证据；默认不入Git
reports/       综合报告、图表、统计与验证收据
tests/         自动化接口及失败分支测试
deliverables/  原有逐日工程与实验证据
Submission/    面向教师的逐周入口
archive/       历史过程材料
```

## 常见问题

- 无GPU可以运行quick和真实tokenizer数据准备。正式7B训练、评估和蒸馏需要GPU。
- quick通过不代表新模型或整周验收通过，它验证数据及历史分数复算。
- Chat Template缺少Jinja2时安装configs/requirements-data.txt。
- 目录已存在时换新run-dir，避免覆盖或混用旧结果。
- 模型、录屏位置见[第7周索引](Submission/Week7/README.md)，GB级资产不复制入Git。
- 中间checkpoint须在核验最终资产和备份后清理；本轮未删除历史权重。
- 测试命令：`python -m unittest discover -s tests -v`；训练分支测试需PyYAML。

Git远端：[sense-time-internship](https://github.com/JAcKs0n02/sense-time-internship)。本轮修改保留在本地工作区，尚未推送或发送给教师。
