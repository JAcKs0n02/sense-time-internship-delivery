# 报告索引

[项目首页](../README.md) · [运行说明](../docs/RUNNING.md)

## 综合技术报告

[PDF](technical_report.pdf) · [Markdown](technical_report.md) · [LaTeX](technical_report.tex) · [源文件包](technical_report_sources.zip) · [编译说明](latex/README.md)

当前为2026年9月20日文字修订版，覆盖八章、六幅图与21张表。旧Word及带日期的版本记录属于历史快照。

## 各周报告

| 周次 | 范围 | 唯一报告入口 | 结果摘要 |
|---|---|---|---|
| Week1 | Day1–5 | [环境与架构](week1/README.md) | 完成环境、推理、架构与Tokenizer实验；身份微调110步。 |
| Week2 | Day6–10 | [数据工程与首次SFT](week2/README.md) | 清洗后4999条；三题比较基座25/30、SFT 22/30。 |
| Week3 | Day11–16 | [SFT优化与评估](week3/README.md) | 九组训练；最佳SFT人工均分3.78875，低于基座4.205。 |
| Week4 | Day17–21 | [DPO偏好对齐](week4/README.md) | 最终40步模型；固定安全题拒绝10/10，业务均分3.640/5。 |
| Week5 | Day22–27 | [多模态理解](week5/README.md) | 完成推理、注意力与LoRA实验；候选未达到全部质量阈值。 |
| Week6 | Day28–33 | [Agent工具调用](week6/README.md) | 工具路由91%，端到端严格成功率39%。 |
| Week7 | Day34–39 | [量化与部署](week7/README.md) | 4-bit显存降低约53%；本测量协议下吞吐下降、PPL上升。 |
| Week8 | Day40–45 | [自动化与知识蒸馏](week8/README.md) | 完成SFT/DPO、三模型评估、人工评分及两轮蒸馏；学生CEval下降0.5944个百分点。 |

## 文件约定

`week1/`至`week8/`的README为统一导航；报告原文与实验数据保留其原位置。`figures/`保存综合报告配图，`latex/`保存排版模板。`week8/`下带日期或phase前缀的目录为对应运行记录，旧状态仅适用于当时版本。
