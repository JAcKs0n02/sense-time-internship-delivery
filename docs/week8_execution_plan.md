# 第8周任务清单与执行计划

> 2026-09-19验收口径更正：老师要求干净环境至少运行数据准备与评估，并支持分段执行；没有要求以Linux CUDA重跑训练、蒸馏、部署全流程作为本项验收。下文历史“完整CUDA复现”待办由[当前准备方案](week8_clean_eval_preparation.md)取代；历史实验回执和冻结候选不改写。

> 2026-09-18：以下为初始计划及历史检查状态，当前完成度以[逐项验收](week8_acceptance_audit.md)为准。正式SFT/DPO训练、三模型评估、人工评分及学生蒸馏前后实测已完成；监督部署联调也已完成；评分恢复分支已接入主控并离线实跑；2026-09-19综合报告Word/PDF已纳入最新结果并完成40页视觉检查；2026-09-19三个最终模型共38文件独立备份、全量哈希/索引与本地CPU加载已通过（见[备份验收](week8_model_backup.md)）；具体清理名单、干净环境数据与真实评估验收和最终发布仍待完成。

> 2026-09-14输入核定更新：正式数据准备源改为已审核原文1580条，83条历史验证继续隔离。以[输入决策](week8_data_input_decision.md)和[数据协议](week8_data_protocol.md)为准；下文旧5000条默认运行记录仅为历史技术验证。


核对日期：2026-09-14。工作周按9月14日至20日安排；Day40–45对应6个任务单元，不强行等同于六个自然工作日。依据根目录最新版《实习需求.docx/pdf》，原文和SHA-256保存在reports/week8/。相对HEAD新增Week7和Week8，Week1–6实质任务不变。Week7已有9月12日正式材料，因此本周重点为Week8并补归档。

## 本周目标和依赖顺序

先打通数据与可追溯执行入口，再完成GPU蒸馏及真实评估，最后补报告数据、整理发布。依赖为：原始数据→Day40清洗与划分→SFT合并→DPO合并→Day41评估；历史Week4教师→Day42教师目标→0.5B学生两轮→前后CEval/速度；已有Week7 AWQ→服务与UI健康检查。Day43–44报告可先用已完成的前七周证据撰写，再填本周实测。

| 日期建议 | 任务单元 | 仓库交付 | 实测/人工工作 | 验收依据 |
|---|---|---|---|---|
| 周一9/14 | Day40 数据与训练 | step1_data_prep.py、step2_train.sh、训练/合并模块、统计JSON | GPU上验证SFT→DPO顺序训练与独立加载 | 双格式ID一致；9:1问题组划分；重试有退出码 |
| 周二9/15 | Day41 评估与部署 | step3_eval.py、step4_deploy.sh、run_pipeline.sh、使用说明 | 在干净GPU环境跑CEval/CMMLU、20题及独立裁判；API/UI联调 | 52+67学科齐全；20/20评分；health及模型身份正确 |
| 周三9/16 | Day42 蒸馏 | distillation.json、distill.py、compare_distillation.py | Week4教师生成；0.5B学生2epoch；同环境前后CEval和tokens/s | 真训练日志、前后对比CSV、失败也有分析 |
| 周四9/17 | Day43 第1–4章 | 综合报告Word/Markdown中的前4章及历史图表 | 核对原始数字与章节引用 | 前4章中文正文累计至少3000字 |
| 周五9/18 | Day44 第5–8章 | 八章综合报告PDF及Word源文件 | 填入本周GPU结果，统一图表和参考文献 | 全文至少6000字；没有把待测写成成功 |
| 周末前 | Day45 仓库整理 | 根README、环境清单、功能目录、周报索引 | 最终模型独立加载/备份后清理中间checkpoint；审阅后推送 | 克隆仓库可快速开始；权重不入Git |

以上是建议排期，不是老师规定的日历截止日期。GPU耗时未知，应优先安排Day42，不要等到写报告时才发现教师或学生权重不可用。

## Day40 逐项执行

1. 默认加载Day6归档的5000条ShareGPT原始格式数据；无需重复下载。自带数据可通过`--input`提供JSON/JSONL。
2. 复用Day7的HTML清理、控制字符过滤、结构校验、真实Chat Template 2048-token截断、精确去重和SimHash/Jaccard复核。
3. 清洗结束后按问题组固定seed42随机9:1划分；相同问题不同答案不能跨训练/验证集。按组取整会有极小比例偏差，统计表显示实际条数。
4. 输出train/validation的Alpaca和ShareGPT、dataset_info.json、statistics.json、audit.json。对齐sample_id并记录输入输出哈希。
5. SFT使用Week3实选epoch-e5配置，DPO使用Week4最终corrective配置。DPO的policy/reference都来自本次SFT合并结果，不冒用历史模型身份。
6. 仅OOM重试，降低micro-batch并相应增加梯度累积，保持有效batch。batch=1仍OOM则停止，保留日志。非OOM错误立即失败。

已完成代码和本地数据路径；真实SFT/DPO训练不因dry-run通过而算完成。

## Day41 逐项执行

正式评估要求显式提供模型、裁判模型和API地址。先调用OpenCompass0.5.3，再核对52个CEval和67个CMMLU学科及details记录，按题目数加权汇总；缺失不填0。然后加载合并模型生成固定20题，调用独立裁判按0–5五维评分，权重30/25/20/15/10，每维保留具体依据。拒绝缺失、非有限、越界评分，标记AI辅助评分，不能替代Week3双人盲评。

`--quick`执行原始数据清洗的字符模式和历史OpenCompass/人工分数复算，不训练、不部署、不调用GPU或在线裁判；用于验证清洁环境的基本入口，不满足“新模型真实评估”的全部验收。`--skip-train`跳过训练但仍准备数据，需EVAL_MODEL_PATH；`--skip-eval`跳过评估；`--deploy`显式启用部署。

部署启动器监督两个后台子进程，检查8000/7860端口、vLLM health及served-model-name，失败回收本次启动的子进程。单卡默认文字模型；图文UI仍保留图片页签，视觉后端按第7周手册单独切换。

## Day42 蒸馏操作与分析

采用序列级蒸馏：先由Week4 DPO教师生成目标文本，退出教师进程后再训练0.5B学生。仅2epoch，最多200条输入提示，保存生成原文、截断标志和筛选结果。该方法不同于逐token软标签KL蒸馏，采样温度0.7也不是KL温度；理论差异写入综合报告。

```bash
python scripts/distill.py generate --teacher "$WEEK4_DPO_PATH" \
  --input "$TRAIN_ALPACA_PATH" --output-dir logs/distill-teacher-001
python scripts/distill.py train --student "$STUDENT_MODEL_PATH" \
  --input logs/distill-teacher-001/teacher_targets.json --output-dir logs/distill-student-001
python scripts/compare_distillation.py --before "$STUDENT_MODEL_PATH" \
  --after logs/distill-student-001/final_distilled --output-dir logs/distill-compare-001
```

比较使用相同硬件、精度、CEval配置与固定推理协议。速度统计包含prefill，2次预热+10次生成，每次128token。不得将历史7B量化速度当作本周0.5B蒸馏结果。教师输出先人工抽查事实、拒绝边界与格式，质量不足时如实报告；技术可行性与效果提升分开判断。

## Day43–44 报告检查

报告覆盖：环境架构、数据工程、SFT实验、DPO、多模态与Agent、量化部署、全链路自动化、总结展望。沿用真实训练曲线、评分雷达图、Rewards图和量化表，并加入Pipeline图。报告已纳入本周训练、蒸馏、真实评估与部署实测；Word/PDF于2026-09-19完成重新渲染与逐页检查。字数检查只统计中文字符，不拿路径、代码行或英文标识凑数。

## Day45 整理与提交

功能目录已经与历史归档并存；不移动既有证据使旧链接失效。最终SFT/DPO/蒸馏模型各保留一份发布候选；AWQ部署资产单独列明，旧模型未验证备份前不删除。Git忽略权重和运行目录。远端清理和推送尚待最终复核；本次不会发送材料给老师。

## 初始真实验收清单（历史计划）

当前训练、评估、蒸馏、监督部署和报告整理均已有证据；以下保留原始计划，不表示全部仍未完成。剩余项以逐项验收清单为准。

- 干净Linux/CUDA环境安装、pip check、至少数据准备和新模型评估通过。
- SFT/DPO正式训练及合并后新进程加载；本地配置生成不替代。
- Week4教师→0.5B学生两轮蒸馏、CEval和速度对比、结果分析。
- 第8周部署监督器与真实vLLM/Gradio联调。
- 报告补入本周结果、清理核验后的中间权重、Git推送。

上述待办源于需要真实GPU模型执行和外部状态核验，不属于“补一份文字说明即可完成”。本地验证收据见reports/week8/。

2026-09-19补充：独立发布副本及CPU验收已通过，见[候选验收](week8_release_candidate.md)。干净环境数据与真实评估验收、最终公开范围审查与推送仍待完成。
