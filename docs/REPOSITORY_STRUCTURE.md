# 仓库结构与版本约定

[项目首页](../README.md) · [报告索引](../reports/README.md)

| 目录 | 用途 |
|---|---|
| reports/ | 综合报告、Week1–8统一入口、图表与运行结果 |
| scripts/ | 当前流水线脚本及其兼容依赖 |
| configs/ | 数据协议、训练配置与分环境依赖 |
| data/ | 数据说明和必要输入 |
| models/ | 模型角色、路径与备份说明，不包含权重 |
| tests/ | 自动化验证 |
| Submission/ | 按周、按天组织的实验材料 |
| deliverables/ | 历史工程实现与原始记录 |
| docs/ | 当前操作指南、方法说明与历史方案 |
| release_metadata/ | 发布副本清单、脱敏映射和版本检查 |

## 当前与历史入口

当前运行方式以[RUNNING.md](RUNNING.md)和[脚本索引](../scripts/README.md)为准；每周报告只有一个`reports/weekN/README.md`入口，链接到原报告。带日期、phase或旧计划名称的文件是历史记录，其待办状态不代表当前进展。

部分冻结数据协议、评分恢复与校验记录引用`outputs/`、`logs/`及`.worktrees/`下的文件。发布副本只包含白名单中的必要依赖，没有Git工作树元数据。保留这些路径是为了维持既有哈希与引用；本次不移动或删除原始实验记录。

## 文档维护

综合报告正文仅编辑`reports/technical_report.md`，通过现有LaTeX脚本生成PDF。修改导航不会重新生成报告。历史收据保持原样，新的检查记录注明日期及文件哈希。
