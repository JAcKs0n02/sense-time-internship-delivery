# 脚本入口

[项目首页](../README.md) · [完整运行命令](../docs/RUNNING.md)

| 功能 | 当前入口 |
|---|---|
| 主控 | `run_pipeline.sh`（仓库根目录） |
| 数据准备 | `step1_data_prep.py` |
| SFT、合并、DPO、合并 | `step2_train.sh` → `pipeline/step2_train.py` |
| 公开基准与custom20评估 | `pipeline/step3_eval.py` |
| 监督部署 | `step4_deploy.sh` |
| 蒸馏与对比 | `distill.py`、`compare_distillation.py` |
| 报告排版 | `render_technical_report_latex.py` |

`train_pipeline.py`、旧`step3_eval.py`及各阶段脚本保留为历史实验或兼容依赖。`build_technical_report.py`为旧版报告组装器，现行报告维护使用Markdown主稿与LaTeX渲染脚本。
