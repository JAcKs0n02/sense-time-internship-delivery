# Qwen模型训练、评估与部署

基于Qwen系列模型的八周实验工程，覆盖数据清洗、QLoRA SFT、DPO、多模态理解、Agent工具调用、量化部署与序列级蒸馏。

## 阅读入口

- **综合技术报告**：[PDF](reports/technical_report.pdf) · [Markdown](reports/technical_report.md) · [LaTeX](reports/technical_report.tex) · [源文件包](reports/technical_report_sources.zip)
- **Week1–8成果**：[统一报告索引](reports/README.md) · [逐周实验材料](Submission/README.md)
- **运行与复现**：[操作指南](docs/RUNNING.md) · [环境依赖](configs/README.md) · [脚本入口](scripts/README.md)
- **模型与附件**：[权重、录屏和校验说明](docs/ARTIFACTS.md)

## 主要结果

| 实验 | 结果与限制 |
|---|---|
| 第8周三模型评估 | 原基座/SFT/DPO的CEval为77.9346%/79.6434%/80.0149%；完整评估见[CSV](reports/week8/current_evaluation_summary.csv) |
| 开放题人工评分 | 原基座3.98250、SFT 3.64875、DPO 3.62875；数学与代码任务仍有缺陷 |
| 4-bit量化 | 显存峰值降低约53%；当前测量协议下速度下降、PPL上升 |
| 学生蒸馏 | CEval 53.7147%→53.1204%，推理速度基本持平 |
| Agent | 工具路由91%，端到端严格成功率39% |

不同数据集、评分协议和运行阶段分别报告，训练损失或单项指标改善不代表全面能力提升。

## 快速检查

在仓库根目录使用Python 3.10及以上版本运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
bash run_pipeline.sh --quick --run-dir logs/quick-001
```

quick使用测试tokenizer处理历史数据并回放已有分数，不加载模型、不调用API。正式数据准备、GPU推理、评分与部署采用独立环境，具体命令见[操作指南](docs/RUNNING.md)。每次运行使用新输出目录。

## 结构与复现边界

代码、数据协议、模型说明和报告分别位于`scripts/`、`configs/`、`models/`、`reports/`；逐周材料位于`Submission/`，原始工程记录位于`deliverables/`。详见[目录约定](docs/REPOSITORY_STRUCTURE.md)。部分历史路径为冻结协议的必要依赖，保持原位。

已有Linux CUDA训练、评估和部署记录，以及独立macOS环境的数据与评分验证；未验证从零安装CUDA后的全部流程。权重与大型视频单独存储，原仓库和中间检查点保留。独立发布副本提供`RELEASE_SHA256SUMS.txt`，用于核验当前文件完整性。
