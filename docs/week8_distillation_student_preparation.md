# Day42 学生训练入口与真实分词准备（2026-09-18）

本轮完成可审计筛选子集支持和真实学生tokenizer清洗。输入数据已通过训练前审核；未开GPU、未启动学生训练。本轮在本地完成，没有运行AutoDL实例或付费裁判API。

## 训练入口修改

`scripts/distill.py train`新增可选参数`--curation-manifest`。未提供参数时保持原完整目标校验流程；提供时先验证原生成回执的203个文件、完整200条原始输出、固定Week4教师身份和191条完整回答，再验证：

- 清单逐项覆盖全部200条原始输出，sample_id、顺序、文件路径与哈希一致。
- 每个保留或排除决定有明确理由，数量统计一致。
- 训练目标必须精确等于所有KEEP项按原顺序形成的子集，不能改写答案、额外添加或漏掉保留项。
- 新质量审核必须绑定原生成回执、筛选清单和目标文件哈希，覆盖全部152个保留sample_id，且无未解决问题。

原始生成accepted=191不改写；新增curated_count=152单独记录。原始生成文件、前轮筛选清单和历史拒绝回执均保留；新质量回执位于`reports/week8/phase3_distillation_student_prep/quality_review.json`。该回执为助手审核，不冒充人工评分或独立裁判结果。

新增5项测试（含篡改子案例），先验证旧代码因缺少子集支持而失败，再实现修改并通过。覆盖漏审、重复决定、空理由、非法决定、路径越界、文件/清单哈希过期、重新计算哈希后的答案改写、顺序变化、漏样本和原始排除项篡改。命令行实际验证也已通过子集门禁，到达预期的缺少学生模型参数错误，未创建训练目录。

## 真实分词结果

学生固定为`Qwen/Qwen2.5-0.5B-Instruct`，revision=`7ae557604adf67be50417f59c2c2f167def9a775`。本地下载config、tokenizer、vocab、merges等5个文件，全部SHA256与此前320学生模型清单一致。没有下载约1GB权重，只运行分词和清洗。

使用Transformers 4.50.0及真实Qwen chat template，调用训练入口同一个`prepare_student_data`函数；未使用字符烟测代替正式长度。

| 项目 | 实际值 |
|---|---:|
| 审核后输入 | 152条 |
| 清洗后保留 | 152条 |
| 拒绝/精确去重/模糊去重 | 均0 |
| 训练集 | 137条 |
| 验证集 | 15条 |
| 划分 | seed42，按完整提示分组90:10 |
| 训练/验证提示交集 | 0 |
| 最大chat template长度 | 1024 token |
| 保持原样的教师答案 | 152条 |
| 截短的用户输入 | 1条 |

唯一截短项为原始输出索引126、筛选后索引95。长篇电影艺术分类题由2048缩至1024 token，只截用户输入。逐项查看后确认分类指令、类别列表及“电影艺术理论研究的现状”标题和相应正文仍在，答案Art未改变，保留有效。没有助手答案被截断，也没有HTML、控制字符或空白清洗导致内容变化。

`prepared_lineage.json`将清洗后的source编号映射回原sample_id及训练/验证划分，避免丢失来源追踪。

## 两轮训练计划

保持既定配置：单卡、batch size=1、gradient accumulation=8、学习率5e-5、LoRA rank=8/alpha=16、BF16、cutoff=1024。

- 每轮137条，18次优化器更新（最后一次仅1条，计入完整遍历）。
- 两轮共274次样本呈现、36次更新；每轮结束验证，预期验证step为18和36。
- Transformers 4.50.0内部按完整累积批次推导的循环上界为3；通过max_steps=36完成实际两轮。验收仍要求最终epoch=2.0，不能把内部上界当成训练三轮。

训练时将再次使用远端固定学生模型文件清洗数据，并核对上述数量和输出文件哈希。输入准备通过并不代表训练效果已经通过。

## 证据与复核

文件目录：`reports/week8/phase3_distillation_student_prep/`。

- `tokenizer_receipt.json`：5个文件来源、固定revision、原模型清单绑定。
- `data/`：正式清洗审计、统计、两种格式的训练/验证数据和dataset_info。
- `prepared_lineage.json`：152条来源与分组对应。
- `quality_review.json`：完整保留子集的数据审核放行。
- `student_training_schedule.json`：18步/轮、36步总计。
- `checks.json`、`cli_check.json`：正式分词、数据不变性与CLI门禁证据。
- `test_result.json`、`regression.log`：42项相关回归全部通过。
- `verification.json`：本轮代码、测试、数据和文档文件哈希。

在仓库根目录，用安装Transformers 4.50.0、tokenizers和Jinja2的Python执行：

```bash
python reports/week8/phase3_distillation_student_prep/verify_preparation.py
```

本轮运行环境为`/tmp/week8-student-tokenizer-20260918`，tokenizer缓存为`/tmp/week8-student-tokenizer-7ae557`。临时目录若已清理，可运行同目录`fetch_tokenizer.py`按固定revision重新下载并核对内容；不需要GPU。

## 下一步

在已授权的320实例上准备有超时与关机保护的单次学生训练会话；上传修改后的入口、完整原始生成证据、筛选清单和新质量回执。远端先复核模型身份、依赖版本，以及再次清洗所得输出哈希与本轮一致，然后训练两轮。真实训练完成后，仍需权重合并验证、CUDA冷加载、蒸馏前后CEval与同协议吞吐比较。当前没有任何这些后续步骤的完成声明。


## 2026-09-18 学生训练推进记录

320已完成真实学生训练：137条训练、15条验证、2轮36步；权重合并检查与CUDA冷加载通过，52份证据已回传并离线复核，实例已关机。详见[学生训练实测记录](week8_distillation_student_training.md)。上述旧阶段的待办为历史状态；当前待办是CEval与同协议吞吐比较，尚未达到蒸馏最终效果验收。
