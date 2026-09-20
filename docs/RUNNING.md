# 运行与复现

[返回项目首页](../README.md) · [依赖环境](../configs/README.md) · [模型与附件](ARTIFACTS.md)

以下命令均在仓库根目录执行。各次运行使用新的输出目录。

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

数据与评估的现行安装清单为`configs/requirements-evaluation.txt`，训练补充清单为`configs/requirements-training-cuda.txt`；旧`requirements-training.txt`只保留作历史证据。CUDA训练、评估与部署环境分开配置，见[依赖说明](../configs/README.md)。本轮已在320新代码目录完成真实评估，复用既有CUDA环境；没有宣称从零安装Linux依赖已验证。

## 当前分段流程

```bash
# 用真实tokenizer按冻结协议准备数据，不启动训练或模型评估
bash run_pipeline.sh --skip-train --skip-eval --run-dir logs/data-001
```

每次使用新目录；`PIPELINE_PYTHON`可指定已安装数据依赖的解释器。正式模式的输入和tokenizer由协议绑定，必须取消旧的`DATA_INPUT`和`TOKENIZER_PATH`环境覆盖；`DATA_INPUT`仅供quick历史测试使用。

正式数据为1422条训练、158条验证。主控调用 `scripts/step2_train.sh` → `scripts/pipeline/step2_train.py`，使用已经完成实测的5轮SFT与40步DPO配置，顺序完成训练和两次合并，保留有限OOM重试。路径随本次运行生成，DPO引用本次合并SFT。`scripts/train_pipeline.py` 的旧命令入口仅保留为历史依赖，不作为现行使用说明。

只检查训练配置（不训练）：

```bash
bash scripts/step2_train.sh --data-dir logs/data-001/data \
  --run-dir logs/train-plan-001 --base-model /absolute/path/Qwen2.5-7B-Instruct --dry-run
```

只准备正式数据及评估配置（不加载模型、不调用API；需要数据与OpenCompass依赖）：

```bash
EVAL_MODEL_PATH=/absolute/path/final_dpo bash run_pipeline.sh \
  --skip-train --dry-run --eval-limit-per-subject 1 --run-dir logs/eval-plan-001
```

新评估入口为 `scripts/pipeline/step3_eval.py`，裁判固定为现有 `gemini-3.1-pro-preview` 与已校准五维量表。无需旧的 `JUDGE_MODEL` / `JUDGE_BASE_URL` 参数。默认完整评估1346道C-Eval、11582道CMMLU和20道自定义题；`--eval-limit-per-subject 1` 只取每科第一题，共119道基准题，另保留全部20道自定义题。小样本结果标为 `sample`，仅验证运行流程，不替换已有全量成绩。

GPU只生成答案，之后可关机并将整个运行目录带回本地评分，避免等待API时闲置GPU：

```bash
# 以下会实际运行模型；使用已完成的合并DPO模型，不重新训练。
EVAL_MODEL_PATH=/absolute/path/final_dpo bash run_pipeline.sh \
  --skip-train --eval-phase generate --eval-limit-per-subject 1 --run-dir logs/eval-check-001

# GPU生成成功后，同一份运行目录中继续评分；此步会调用Gemini API。
python scripts/pipeline/step3_eval.py --phase score \
  --output-dir logs/eval-check-001/evaluation
```

Gemini Key由仓库外的私有文件读取，设置命令为 `python scripts/setup_week8_gemini_key.py`。完整单命令运行可省略 `--eval-phase generate`，全量评估再省略 `--eval-limit-per-subject 1`。运行后查看 `evaluation/status.json`、`summary.csv`、逐题答案及API响应；已经完成的同目录评分会核验并复用，未知付费请求或失败不会自动重试。完成评分后若迁移目录，应只读核验已有结果；付费入口绑定原输出路径，不能在迁移目录直接重新执行，见[迁移说明](week8_dependency_delivery.md#已评分目录的迁移限制)。

`--dry-run` 会真实执行数据准备并生成配置，输出为 `planned_only`，不能称为训练或推理成功。每次生成使用新目录，不能把计划目录直接当成实际运行目录。`scripts/step3_eval.py` 是旧实验依赖；现行主控、quick和上述命令均调用 `scripts/pipeline/step3_eval.py`。旧 `--clean-eval` 及 `--score-only` 参数仍可用于历史方案核验。

## 已有评分核验与恢复

```bash
# 离线核验三模型各20题，不调用API，不准备数据、不启动GPU
bash run_pipeline.sh --score-only all --run-dir logs/score-audit-001
```

显式恢复单模型可加`--execute-score`；完整成绩不会重复收费，失败状态和未知付费请求仍会阻止重试。只能使用已冻结的三模型方案。详细边界及命令见[评分恢复入口](week8_pipeline_score_integration.md)。这不替代新模型fresh评估，也不代表整条CUDA流水线一次跑通。

## 部署与蒸馏

```bash
export WEEK7_SERVING_PYTHON=/path/to/serving-env/bin/python
export WEEK7_UI_PYTHON=/path/to/ui-env/bin/python
export WEEK7_TEXT_MODEL_PATH=/path/to/verified/awq-model
bash scripts/step4_deploy.sh --run-dir logs/deploy-001
```

检查端口、vLLM health、模型身份与UI后返回后台监督进程PID。停止前核对本次status.json和进程身份，再向监督进程发送SIGTERM。单卡默认文字服务；视觉后端按[第7周手册](../Submission/Week7/本地部署与操作手册.md)单独切换。监督器已在320完成真实API、Gradio客户端对话、正常停止和异常回收验收，见[部署记录](week8_supervised_deployment.md)。验证结束后服务停止，实例关机。

蒸馏教师为Week4 DPO，学生为Qwen2.5-0.5B-Instruct，使用离线序列级教师目标与2epoch学生LoRA，不冒称逐token软标签KL。配置为[distillation.json](../configs/distillation.json)，入口为`distill.py generate`、`distill.py train`、`compare_distillation.py`；命令见[Day42计划](week8_execution_plan.md)。真实学生训练已完成2轮36步，合并与CUDA冷加载通过，见[训练记录](week8_distillation_student_training.md)；前后CEval为53.7147%→53.1204%，速度55.5303→55.6263 token/s，效果未提升；完整证据与分析见[比较记录](week8_distillation_comparison.md)。
