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

`train_pipeline.py`、`step3_eval.py`及保留的阶段模块提供现行入口使用的训练、评分校验与进程控制函数。冻结校准绑定的共享模块保留原字节；独立实例编排命令不作为现行入口。旧报告组装器已经移出，报告使用Markdown主稿与LaTeX渲染脚本。
