# 项目交付清单

[项目首页](../README.md) · [综合报告PDF](../reports/technical_report.pdf) · [各周报告](../reports/README.md)

依据项目需求中的Day1–45交付项整理。材料是否齐备与实验是否达到预期效果分别说明；不将额外调试过程列为必交成果。

## 提交入口

1. 代码仓库链接：包含现行代码、配置、必要数据、各周成果、关键日志与评估结果。
2. 综合技术报告：`reports/technical_report.pdf`；可编辑源为Markdown、LaTeX及 `technical_report_sources.zip`。
3. 外部附件：Week7应用演示视频和量化模型，以及所需最终模型权重。文件清单在[模型与附件](ARTIFACTS.md)，仓库中的定位记录不是可下载链接，需另外提供实际访问方式。

## Week1–6：逐日材料

### Week1

| 任务 | 具体交付内容 | 文件入口 |
|---|---|---|
| Day1 | CUDA/Conda环境和GPU截图 | [材料](../Submission/Week1/Day1_Environment_Setup/) |
| Day2 | 下载确认、3组对话日志、推理脚本 | [材料](../Submission/Week1/Day2_Model_Download_and_Inference/) |
| Day3 | 架构分析、参数统计、Qwen/Llama比较 | [材料](../Submission/Week1/Day3_Architecture_Analysis/) |
| Day4 | 10+边界用例Tokenizer Notebook | [材料](../Submission/Week1/Day4_Tokenizer_Experiments/) |
| Day5 | LLaMA-Factory训练日志、曲线与周报 | [材料](../Submission/Week1/Day5_LLaMA_Factory_and_Weekly_Report/) |

### Week2

| 任务 | 具体交付内容 | 文件入口 |
|---|---|---|
| Day6 | 原始数据清单、Alpaca与ShareGPT格式数据 | [材料](../Submission/Week2/Day6_Data_Collection_and_Formatting/) |
| Day7 | 独立清洗脚本、清洗后数据及分布图 | [材料](../Submission/Week2/Day7_Cleaning_Pipeline/) |
| Day8 | 逐项注释的SFT配置 | [材料](../Submission/Week2/Day8_Configuration_and_First_SFT/) |
| Day9 | Loss曲线、合并记录、3题对话；权重另行提供 | [材料](../Submission/Week2/Day9_Training_Monitoring_and_Merge/) |
| Day10 | 数据与SFT周报、FAQ | [材料](../Submission/Week2/Day10_Weekly_Report_and_FAQ/) |

### Week3

| 任务 | 具体交付内容 | 文件入口 |
|---|---|---|
| Day11 | 控制变量实验计划与配置 | [材料](../Submission/Week3/Day11_Experiment_Design/) |
| Day12–13 | 各组原始日志、Loss、耗时、显存记录 | [材料](../Submission/Week3/Day12_13_Batch_Experiments/) |
| Day14 | 20题评估脚本、五维人工评分、雷达图、模型选择 | [材料](../Submission/Week3/Day14_Model_Evaluation/) |
| Day15 | 基座与最优SFT的CEval/CMMLU成绩表 | [材料](../Submission/Week3/Day15_OpenCompass/) |
| Day16 | SFT优化周报与最优模型路径 | [材料](../Submission/Week3/Day16_Weekly_Report_and_Model_Archive/) |

### Week4

| 任务 | 具体交付内容 | 文件入口 |
|---|---|---|
| Day17 | 五类偏好数据构造指南 | [材料](../Submission/Week4/Day17_Preference_Data_Methodology/) |
| Day18 | 开源与自建偏好对、合并数据集 | [材料](../Submission/Week4/Day18_Preference_Dataset/) |
| Day19 | DPO配置、启动日志、Rewards曲线 | [材料](../Submission/Week4/Day19_DPO_Training/) |
| Day20 | 10题安全测试、5题业务对比、最终模型合并记录 | [材料](../Submission/Week4/Day20_DPO_Merge_and_Evaluation/) |
| Day21 | DPO周报、最终模型归档说明 | [材料](../Submission/Week4/Day21_Weekly_Report_and_Final_Model_Archive/) |

### Week5

| 任务 | 具体交付内容 | 文件入口 |
|---|---|---|
| Day22 | VLM下载确认、5张输入图片 | [材料](../Submission/Week5/Day22_Model_and_Image_Preparation/) |
| Day23 | 5图×5类问题的25条推理结果 | [材料](../Submission/Week5/Day23_VLM_Inference/) |
| Day24 | 至少3张注意力热力图与实现 | [材料](../Submission/Week5/Day24_Cross_Modal_Attention/) |
| Day25 | 10组误导问题及幻觉分析 | [材料](../Submission/Week5/Day25_VLM_Hallucination_Evaluation/) |
| Day26 | 图文训练数据、LoRA配置、训练日志及模型说明 | [材料](../Submission/Week5/Day26_VLM_LoRA/) |
| Day27 | 多模态周报 | [材料](../Submission/Week5/Day27_Weekly_Report/) |

### Week6

| 任务 | 具体交付内容 | 文件入口 |
|---|---|---|
| Day28 | Calculator与本地KnowledgeRetrieval代码 | [材料](../Submission/Week6/Day28_LangChain_and_Basic_Tools/) |
| Day29 | ReAct初始化脚本和单轮调用日志 | [材料](../Submission/Week6/Day29_ReAct_Agent/) |
| Day30 | AST语法检查工具、三工具代码、多步推理日志 | [材料](../Submission/Week6/Day30_Complex_Tools_and_Multistep/) |
| Day31 | 100条ReAct训练数据、工具SFT配置/日志及模型说明 | [材料](../Submission/Week6/Day31_Tool_Call_SFT/) |
| Day32 | 错误模式分析、Prompt优化及重测结果 | [材料](../Submission/Week6/Day32_Error_Analysis/) |
| Day33 | Agent周报 | [材料](../Submission/Week6/Day33_Weekly_Report/) |

## Week7：部署与应用

| 任务 | 具体交付内容 | 文件入口 |
|---|---|---|
| Day34 | AWQ量化代码、配置、模型清单；权重单独交付 | [量化实现](../deliverables/week7/day34/) · [附件清单](../Submission/Week7/assets_manifest.json) |
| Day35 | FP16/AWQ/GPTQ显存、速度、PPL对比 | [对比报告](../Submission/Week7/量化对比报告.md) |
| Day36 | vLLM启动脚本与OpenAI兼容API客户端 | [脚本](../deliverables/week7/day36/source/scripts/) |
| Day37 | 流式聊天、历史记录、Temperature/Top-p界面 | [app.py](../deliverables/week7/day37/app.py) |
| Day38 | 图片上传、界面优化、应用录屏 | [多模态app](../deliverables/week7/day38/app.py) · [优化记录](../Submission/Week7/优化记录.md) · [视频清单](../Submission/Week7/assets_manifest.json) |
| Day39 | 部署周报、安装与操作手册 | [周报](../Submission/Week7/第7周部署与应用报告.md) · [操作手册](../Submission/Week7/本地部署与操作手册.md) |

## Week8：自动化、蒸馏与最终交付

| 要求 | 具体交付内容 | 文件入口 |
|---|---|---|
| 40.1 | 加载、清洗、去重、双格式、9:1划分、JSON统计 | [step1](../scripts/step1_data_prep.py) · [数据与统计](../data/week8/prepared/) |
| 40.2 | SFT→DPO自动训练、最佳配置、日志 | [step2](../scripts/step2_train.sh) · [配置](../configs/week8_full_training_candidate/) · [日志](../logs/week8/training/) |
| 40.3 | 有限OOM重试、两次权重合并 | [实现](../scripts/train_pipeline.py) · [训练核验](../reports/week8/phase2_full_training_target/target_review.json) |
| 41.1 | CEval/CMMLU、20题五维自动评分、CSV | [评估入口](../scripts/pipeline/step3_eval.py) · [全量汇总](../reports/week8/current_evaluation_summary.csv) · [原始基准输出](../reports/week8/benchmarks/) |
| 41.2 | 量化vLLM、Gradio后台启动与健康检查 | [step4](../scripts/step4_deploy.sh) · [部署结果](../reports/week8/phase4_supervised_deployment/verification.json) |
| 41.3 | 主控、skip-train/skip-eval与分段执行 | [主控](../run_pipeline.sh) · [使用说明](RUNNING.md) |
| 42.1–42.2 | 蒸馏原理、Week4 DPO教师→0.5B学生、两轮训练 | [配置](../configs/distillation.json) · [实现](../scripts/distill.py) · [实际训练配置及日志](../reports/week8/phase3_distillation_student_gpu/retrieved/student/) |
| 42.3 | 学生蒸馏前后CEval与速度对比、分析 | [对比表](../reports/week8/phase3_distillation_comparison/comparison.csv) · [分析](week8_distillation_comparison.md) |
| 43–44 | 八章、图表、6000字以上综合报告，PDF及源文件 | [PDF](../reports/technical_report.pdf) · [Markdown](../reports/technical_report.md) · [LaTeX](../reports/technical_report.tex) · [源文件包](../reports/technical_report_sources.zip) |
| 45.1 | 最终SFT/DPO/蒸馏模型各一份的交付清单 | [模型说明](../models/README.md)；三个最终模型已备份，中间检查点按项目持有人决定保留在原工程 |
| 45.2–45.4 | 功能目录、Conda环境、quick、FAQ、Git仓库 | [目录结构](REPOSITORY_STRUCTURE.md) · [环境](../environment.yml) · [首页](../README.md) |

现行入口曾在Linux CUDA环境完成119题基准样本与全部20题自动评分；独立macOS环境完成数据准备和已有结果核验。两者均保留原始记录，未将其称为从零安装CUDA环境后的全链路重跑。本次仓库整理的验证范围另见[当前整理验证](../RELEASE_VALIDATION.md)。

## 未纳入正式提交的过程材料

恢复工作树、临时输出目录、早期流水线整包、重复上传包、GPU空闲监控、未采用的裁判校准版本、失败会话控制文件、过期执行计划、重复发布审查保留于原工程或Git历史。保留对最终结论必要的原始实验数据、最终校准、最终评分响应和故障保护测试。必要数据已迁入明确的数据目录；不是直接删掉仍被脚本使用的文件。

## 结果与交付边界

- Week2合并模型的小样本质量未超过基座；Week5微调后未取得预期的明显提升。材料有记录，不能将这两项效果要求标记为已达标。
- Week8两轮蒸馏已经完成；CEval下降0.5944个百分点，速度基本持平，按实际结果分析。该任务允许效果不理想。
- AWQ实现使用了兼容替代工具，报告保留方法差异；模型权重和录屏尚需通过仓库之外的渠道提供。
- 当前Git仓库为私有。提交链接前需授予接收者访问权限，或由项目持有人调整可见性。

本清单整理交付范围，不新增训练、评分或身份信息要求。匿名人工评分已保留，不需要补录姓名。
