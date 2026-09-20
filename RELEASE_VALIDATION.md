# 发布副本验证

当前导航与文件检查见`release_metadata/navigation_verification_20260920.json`。各周报告使用`reports/week1/README.md`至`reports/week8/README.md`统一入口；逐日材料索引位于`Submission/`。

9月20日重新运行quick、三模型已有评分核验和最新评估只读核验，均通过，新增API调用为零；未重新训练或运行GPU推理。独立macOS安装记录见`release_metadata/verification.json`，报告文字修订记录见`release_metadata/report_update_20260920.json`。这些记录分别描述其验证范围，不代表从零安装CUDA环境后的全流程运行。

原始仓库、历史实验记录与中间检查点保留。最终权重和大型录屏单独存储，见[模型与附件](docs/ARTIFACTS.md)。根目录`RELEASE_SHA256SUMS.txt`覆盖当前发布文件（不含清单自身及Git元数据）。
