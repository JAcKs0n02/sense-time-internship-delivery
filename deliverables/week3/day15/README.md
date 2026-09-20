# Day 15：OpenCompass 通用评测

## 完成目标

Day 15 只回答一个问题：Day 14 已由两位人工评分者选出的最优 SFT `epoch-e5`，在公开中文知识基准 CEval 与 CMMLU 上，相对基座模型表现如何。OpenCompass **不参与超参数选择**，也不会反向改变 Day 14 的选模结果。

## 模型与环境

- OpenCompass：`0.5.3`
- GPU：NVIDIA GeForce RTX 3090 24 GB
- 基座模型：`/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct`
- 最优 adapter：`/root/autodl-tmp/qwen25-week3/runs/epoch-e5/attempt-001/adapter`
- adapter 权重 SHA-256：`cd2822fc5b160095524aa0203b1e2bab8e71e7897535cf093f4e8807c540a75a`
- 合并模型：`/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged`
- 推理模式：Hugging Face chat，`max_seq_len=2048`、`max_out_len=32`、`batch_size=4`
- 数据配置：`ceval_gen`、`cmmlu_gen`，由当前安装的 OpenCompass 解析为实际配置并保留展开文件

## 选模冻结与模型归档

Day 14 选模输入在 OpenCompass 正式评测前冻结，记录在 `source/results/selected_model_input.json`。冻结文件明确包含 `opencompass_used_for_selection: false`。只有 `epoch-e5` 被合并，避免为九个 adapter 重复生成完整模型。

合并模型包含 14 个文件，总大小 `15,247,184,276` bytes。独立进程使用 `local_files_only=True` 成功加载 `Qwen2ForCausalLM` 与 `Qwen2TokenizerFast`；完整文件清单及整体清单哈希记录在 `source/results/best_model_archive.json`。

## 兼容性处理

PyPI 安装的 OpenCompass 0.5.3 不包含仓库版说明中的 `tools/list_configs.py`。本次没有临时升级版本，而是使用同一版本官方 `opencompass` CLI 的 `--datasets ceval_gen cmmlu_gen` 解析数据配置，并保存 CLI 生成的完整可执行配置。CEval 与 CMMLU 的 ModelScope 数据脚本首次缓存需要显式允许远程数据脚本；缓存完成后正式任务按冻结配置运行。

正式运行前先以 CEval `computer_network` 单学科同时测试两个模型。两者 smoke 分数均为 `68.42`，仅证明 chat 模板、推理和结果解析链路可工作，**不进入正式分数表**。

## 正式结果

| 模型 | 数据集 | 按题目数加权准确率（%） | 学科宏平均（%） | 学科数 | 题目数 | 相对基座（百分点） |
|---|---|---:|---:|---:|---:|---:|
| 基座 | CEval | 77.542262 | 77.001569 | 52 | 1,398 | 0.000000 |
| 基座 | CMMLU | 0.647412 | 0.622482 | 67 | 11,649 | 0.000000 |
| 最优 SFT（`epoch-e5`） | CEval | 79.189426 | 78.977255 | 52 | 1,398 | +1.647164 |
| 最优 SFT（`epoch-e5`） | CMMLU | 5.205187 | 5.003287 | 67 | 11,649 | +4.557775 |

两个模型的 CEval 与 CMMLU 均完成全部学科，正式调度器退出码为 0。`epoch-e5` 在本次固定配置下两个公开基准均高于基座；其中 CMMLU 的绝对分数仍很低，因此只能把结果解释为“在当前 OpenCompass 0.5.3 生成式评测链路下的相对提升”，不能据此宣称模型已具备稳定的通用中文知识能力。人工盲评中基座仍高于最优 SFT，说明公开选择题基准与开放回答质量衡量的是不同侧面。

主报告使用按题目数加权的准确率；学科宏平均作为辅助指标。正式结果文件必须同时包含 `base/best_sft × ceval/cmmlu` 四行，且两个模型均须退出码为 0。

## 证据入口

- `configs/opencompass_week3.py`：冻结的 OpenCompass CLI 计划
- `configs/qwen25_7b_week3_best_export.yaml`：最优 adapter 合并配置
- `source/results/opencompass_scores.csv`：教师要求的四行分数表
- `source/results/opencompass_normalized_results.json`：标准化聚合结果
- `source/results/best_model_archive.json`：adapter、合并模型、加载验证和清单哈希
- `source/results/opencompass_remote_evidence.json`：远端正式结果目录、退出状态和证据哈希
- `source/scripts/`：冻结、运行、解析、汇总与归档脚本
- `source/tests/`：配置、解析、归档和路径门禁测试

## 结论边界

OpenCompass 是 Day 14 后的泛化验证，不是选择依据。无论公开基准结果升降，Week 3 的“最优 SFT”仍指人工五维加权总分最高的 `epoch-e5`；报告不会把 loss 降低、smoke 分数或单一公开基准扩大解释成全面质量提升。
