# 模型保留与血缘

2026-09-19更新。权重不进Git；历史模型和第8周新实验分别保留，效果未提升的模型不自动晋级。

| 角色 | 当前状态 | 备份状态 |
|---|---|---|
| 历史SFT | Week3 epoch-e5 | 本次未核验完整备份 |
| 历史DPO/教师 | Week4 reward_corrective_40step_merged | 本次未核验完整备份 |
| Week7 AWQ部署资产 | 本地独立交付模型；16个模型/视频附件本轮哈希核验通过 | 本地完整资产存在；不是Week8新DPO |
| Week8最终SFT | 1775步，合并及远端独立加载已验证 | 本地完整备份14文件；SHA、339张量索引及离线CPU加载通过 |
| Week8最终DPO | 40步，合并及远端独立加载已验证 | 本地完整备份14文件；SHA、339张量索引及离线CPU加载通过 |
| Week8蒸馏学生 | 0.5B、2轮36步；比较与分析完成，效果未提升 | 本地完整备份10文件；SHA、290张量及离线CPU加载通过 |

本周SFT/DPO远端目录为`/root/autodl-tmp/week8-full-training-runtime-20260916/logs/full-run-01/models/final_sft`与`final_dpo`；学生为`/root/autodl-tmp/week8-distillation-student-20260918/run-01/student/final_distilled`。2026-09-19通过320无卡模式重新读取并备份，本地分别位于`backups/week8-20260919/week8_final_sft`、`week8_final_dpo`和`week8_final_distilled`。合计31,498,338,719字节；完成后已确认320关机。

[预期备份文件清单](../reports/week8/release_audit_20260919/expected_model_backups.json)作为原审查快照保持不变。[备份验收记录](../docs/week8_model_backup.md)及[成功回执](../reports/week8/model_backup_20260919/verification.json)证明本次完整复制、两次逐文件哈希核对、索引检查和独立加载均通过。加载仅为短文本冒烟检查，不改变模型效果结论。具体中间检查点清理名单仍须单独审查；本次未删除任何模型或远端目录。

[发布审查](../docs/week8_release_review.md)记录本轮缺口。Week7资产位置见[提交索引](../Submission/Week7/README.md)。
