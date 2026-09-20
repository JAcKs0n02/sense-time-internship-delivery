# Day 31：工具调用 SFT

## 任务与结论

本日完成老师要求的 ReAct 工具调用数据构造、LLaMA-Factory LoRA 训练、模型归档与工具能力评测。

- 核心训练集：100 条，满足老师的最低数量要求；
- 扩展训练集：200 条；正式训练合计 300 条；
- 开发集：40 条，仅用于 checkpoint 与 Prompt 选择；
- 最终测试集：100 条，仅在配置冻结后运行一次；
- 正式训练：2 epoch、76 steps、退出码 0，最终 eval loss 为 0.0638；
- 最终候选：checkpoint-38；开发集严格成功率 45.0%；
- 最终测试 raw：首工具 91.0%，完整工具序列 91.0%，参数正确率 89.0%，严格成功率 39.0%。

raw 指标代表模型直接能力；guarded 指标代表安全策略介入后的系统行为，两者分别记录，不能相互替代。

## 数据格式

每条样本包含用户任务、结构化工具调用、真实 Observation 和最终回答。训练数据采用 LLaMA-Factory ShareGPT 工具角色：`human → function_call → observation → gpt`。三个工具为 Calculator、KnowledgeRetrieval 和 AST-only CodeExecutor。

训练集分为老师要求的 100 条核心样本与 200 条扩展样本。开发集和测试集拥有独立 ID、Prompt 与成功谓词，未参与训练。

## 训练与选择流程

1. 校验三工具 schema、知识库、数据数量、角色顺序与样本去重；
2. 用固定 LLaMA-Factory 环境执行官方 parser smoke 和训练前检查；
3. 以已核验的 Week4 DPO merged model 为基座训练 LoRA；
4. 保存 checkpoint-19/38/57/76，并在 40 条开发集上统一评测；
5. 按严格成功率、参数正确率、完整序列和语义正确率排序，选择 checkpoint-38；
6. 冻结 checkpoint、main Prompt、100 条测试集与确定性生成参数；
7. 分别生成 raw 和 guarded 最终测试记录。

## 开发集 checkpoint 对比

| Checkpoint | 严格成功率 | 首工具 | 完整序列 | 参数正确率 | 语义正确率 |
|---|---:|---:|---:|---:|---:|
| 19 | 22.5% | 100.0% | 100.0% | 87.5% | 35.0% |
| 38 | **45.0%** | 92.5% | 92.5% | 90.0% | **50.0%** |
| 57 | 30.0% | 92.5% | 92.5% | 92.5% | 37.5% |
| 76 | 30.0% | 95.0% | 95.0% | 92.5% | 35.0% |

checkpoint-38 在综合排序中最优。最终导出的 adapter 与 checkpoint-76 字节一致，但正式归档选择的是 checkpoint-38，其权重 SHA-256 为 `a5e348ad180f8183479d3138b40613ead7163f3b2089c9b271063b64be5cdc5c`。

## 最终测试

| 指标 | Raw | Guarded |
|---|---:|---:|
| 严格成功率 | 39.0% | 27.0% |
| 首工具选择 | 91.0% | 91.0% |
| 完整工具序列 | 91.0% | 91.0% |
| 参数 schema 合法 | 100.0% | 100.0% |
| 参数正确率 | 89.0% | 89.0% |
| Observation 完整性 | 100.0% | 100.0% |
| 语义正确率 | 49.0% | 49.0% |
| 最终回答可溯源 | 40.0% | 40.0% |
| 无循环 | 100.0% | 100.0% |

guarded 模式会拒绝违反策略的轨迹，因此完成率由 raw 的 84.0% 降至 57.0%。这反映安全边界的作用，不表示模型能力提升。

## 模型归档

Git 与教师包不保存大权重。`model_archive/adapter_config.json` 保存加载配置；`source/results/v2_adapter_manifest.json` 保存正式训练输出的逐文件大小和 SHA；`source/results/v2_frozen_release.json` 固定最终候选、Prompt、测试集和生成配置。

正式训练输出中的最终 adapter 为 40,400,200 bytes；checkpoint-38 的权重身份由冻结归档记录。部署时应使用冻结归档指定的 checkpoint-38，而不是仅凭训练目录中的 final adapter 名称判断版本。

## 完成检查

- [x] 100 条核心 ReAct 训练数据；
- [x] 200 条扩展训练数据；
- [x] 官方 parser 与训练前检查通过；
- [x] 正式训练退出码为 0；
- [x] 四个 checkpoint 使用同一开发集评测；
- [x] 最终配置在测试前冻结；
- [x] 100 条 raw/guarded 最终测试均保留逐题轨迹；
- [x] adapter 配置、大小、路径与 SHA 已归档。
