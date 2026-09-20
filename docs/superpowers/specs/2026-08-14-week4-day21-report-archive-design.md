# Week 4 Day 21 Report and Final DPO Archive Design

## 目标

完成老师 Day 21 的两项要求：从 Day 19 冻结的逐步指标生成可读的 DPO Rewards 曲线，并撰写《第4周：DPO偏好对齐报告》。最终教师交付同时包含周报、Rewards 图和可审计的最终 DPO 模型归档。

## 最终模型口径

Day 21 的最终模型固定为 Day 20 已完成合并、独立加载和正式评测的 `sft_dpo_v1_merged`：

`/root/autodl-tmp/qwen25-week4/best_model/qwen25-7b-week4-dpo-merged`

三个 remediation 候选只使用隔离开发集评估，均未通过全部预注册门槛，因此不晋级、不合并、不运行老师最终题，也不替换上述正式模型。教师报告只用一节说明该结论；原始候选训练和开发集证据保留在工程目录。

## Rewards 图

唯一训练主数据源为 Day 19 的 240-step CSV，不从 README 或日志手工复制数字。标准化结果保留 step、epoch、loss、chosen reward、rejected reward、margin、accuracy 和 20-step moving average。

最终 PNG 使用 2×2 布局：

1. chosen/rejected 原始值与 20-step moving average；
2. margin 原始值、moving average 和零参考线；
3. reward accuracy 原始值与 moving average；
4. 六个 validation checkpoint 的 chosen、rejected、margin 和 accuracy 趋势。

图和报告必须明确：`pref_beta=0.1`，margin 改善主要来自 chosen reward 上升，rejected reward 在全程窗口内没有下降；Reward 改善不能代替独立行为评测。

## 周报结构

报告按目标、偏好方法、数据、训练、Rewards、模型合并、安全、业务、remediation、问题与限制、验收矩阵和最终模型归档组织。所有核心数字必须链接到冻结 CSV、JSON、配置、日志或图片。

报告保持三个结论分离：

- 老师安全拒绝率 `10/10 = 100%`，通过 90% 门槛；
- 增强安全响应率 `4/10 = 40%`，仍有提升空间；
- 五题业务均分 SFT-only `3.565`、SFT+DPO `3.355`，不支持业务质量提升。

## 模型归档

Git 中的归档是模型 lineage 和完整性记录，不复制约 15.2GB 权重。归档记录 Week 3 SFT 起点、Day 18 train/validation 哈希、Day 19 配置与 adapter、Day 20 merged 模型路径、文件数量、权重分片、字节数、清单哈希、独立加载、无害 smoke 和正式评测结果。权重保留在 AutoDL 数据盘。

## 目录边界

`deliverables/week4/day21/` 保存脚本、测试、标准化指标、图片、报告、验收矩阵、模型归档、文件清单和验证结果。

`Submission/Week4/Day21_Weekly_Report_and_Final_Model_Archive/` 只保存教师需要的 README、周报、Rewards PNG、训练指标 CSV、验收矩阵和模型归档 JSON。不得包含模型权重、checkpoint、私有盲评映射、凭据、AutoDL 关机证据、缓存或 symlink。

## 验证

验证器跨 Day17–Day21 检查：五类偏好和 10 组样例；500+210=710 与 639/71；Day19 240/240 steps 和有限 Rewards；Day20 10 道安全题、5 道业务题、100% 主拒绝率和固定业务均分；最终模型清单哈希与独立加载状态；Day21 报告中的关键数字。最后验证 Markdown 链接、结构化文件、图片、Submission SHA-256、禁用文件、大文件和 Git whitespace。

