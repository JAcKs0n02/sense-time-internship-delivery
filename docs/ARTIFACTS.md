# 模型与外部附件

[项目首页](../README.md) · [运行说明](RUNNING.md)

Git仓库包含代码、配置、报告、必要冻结输入与评估结果。模型权重、中间检查点、环境缓存和大型演示视频不包含在clone中。

| 资产 | 内容与状态 | 定位与核验 |
|---|---|---|
| Week8最终模型 | SFT、DPO、0.5B蒸馏模型；38文件约31.50GB，独立备份并通过哈希、索引与CPU加载检查 | [备份记录](week8_model_backup.md) · [模型角色](../models/README.md) |
| Week7量化模型 | AWQ权重与配置，单独存储 | [逐文件清单](../Submission/Week7/assets_manifest.json) |
| Week7演示视频 | 四段录屏，单独存储 | [附件清单](../Submission/Week7/assets_manifest.json) |
| 历史模型 | 各阶段基座、adapter与合并模型 | [各周报告](../reports/README.md)中的模型清单 |

原工作区的Week7资产位于`outputs/Week7_正式提交_20260912/`。上述路径为资产定位记录，不是公开下载地址；在其他机器复现时，由项目持有人提供文件并按清单校验，再将模型路径填入运行命令。Git仓库的可访问性不等于外部模型与视频已经分享。

发布副本中的9个历史JSONL文件已对18处凭据字面量脱敏，原始训练记录不回写；详情见`docs/redaction_mapping.json`。该变化与使用脱敏数据重训不同，历史成绩仍对应原始实验。
