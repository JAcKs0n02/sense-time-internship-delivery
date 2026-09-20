# 第 6 周：Agent 智能体开发报告

## 1. 摘要与老师要求

本周完成从安全工具、ReAct Agent、多步业务任务到工具调用 SFT、错误分析和最终归档的完整链路。老师要求的三个工具、至少三阶段任务、《Agent 错误模式分析报告》和本周报均有源代码、逐题轨迹、机器验证结果与 SHA-256 证据。

Day31 使用 100 条核心训练样本和 200 条扩展样本完成 LoRA 训练，并以独立开发集选择 checkpoint-38。100 条最终测试的 raw 首工具准确率和完整工具序列准确率均为 91%，参数正确率 89%，严格成功率 39%。这说明工具路由已较稳定，但最终答案语义与 Observation 对齐仍是主要限制。

## 2. 环境与模型血缘

| 项目 | 记录 |
|---|---|
| 基座 | 已核验的 Week4 DPO merged model |
| SFT 框架 | LLaMA-Factory 0.9.3 |
| 训练规模 | 300 train / 40 dev / 100 final test |
| 训练过程 | 2 epoch / 76 steps / exit code 0 |
| 最终 eval loss | 0.0638269 |
| 最终候选 | checkpoint-38 |
| checkpoint-38 adapter SHA-256 | `a5e348ad180f8183479d3138b40613ead7163f3b2089c9b271063b64be5cdc5c` |
| Prompt | main，SHA-256 `c16614bb753cd8c5226efde24a5ab1acfc55900023eb6f69679ec047d25f5735` |
| 推理 | do_sample=false，max_new_tokens=384，seed=42，最大 6 步 |

模型、数据、Prompt、知识库和生成配置在最终测试前写入 `v2_frozen_release.json`。最终测试不参与 checkpoint 或 Prompt 选择。

## 3. 三个工具

| 工具 | 功能 | 安全边界 |
|---|---|---|
| Calculator | 受限数值表达式计算 | AST 白名单；不调用 `eval`；限制节点、深度、指数和数值范围 |
| KnowledgeRetrieval | 查询固定本地产品知识库 | 不联网；返回明确的 success/not_found/ambiguous/error 状态 |
| CodeExecutor | Python 语法与风险 AST 检查 | 只解析、不执行代码；测试验证无副作用文件 |

三工具拥有固定名称、描述和 JSON schema，并在 Agent 初始化时联合绑定。工具返回统一包含 `status`、`data` 和 `error_code`，便于 Agent 与评测器稳定处理。

## 4. ReAct Agent 与单轮调用

Day29 将 Week4 模型接入 ReAct Agent。单轮正式轨迹证明模型能够选择 Calculator、生成合法参数、读取真实 Observation 并输出最终回答。运行记录同时保存模型身份、Prompt、工具 schema 和生成参数。

ReAct 表示“思考—行动—观察—回答”的交替过程。本项目只保存简短路由理由和可审计工具轨迹，不依赖不可复核的长推理文本。

## 5. 多步任务

Day30 的教师案例为一个可审计的三阶段业务流程：

1. KnowledgeRetrieval 查询商品价格、运费与库存；
2. Calculator 根据检索结果计算总价 719 元；
3. Agent 基于两次真实 Observation 给出购买结论。

同日加入 AST-only CodeExecutor，使工具总数达到 3。复杂任务的 trace、run spec、知识库 SHA 和代码安全审计均在提交包中。

## 6. 工具调用 SFT

训练数据采用 LLaMA-Factory ShareGPT 工具角色，保留 `human → function_call → observation → gpt` 的完整轨迹。100 条核心样本满足老师要求，200 条扩展样本增加实体、表达式、多步组合、格式与错误终止覆盖。

训练 loss 从早期约 1.2 下降到后期约 0.05–0.08，四次开发集评测分别对应 checkpoint-19/38/57/76。checkpoint-38 的严格成功率 45% 和语义正确率 50% 为四者最高，因此作为最终候选；不能仅按最低 eval loss 选择 Agent 模型。

## 7. 固定评测与错误模式

### 最终测试结果

| 指标 | Raw | Guarded |
|---|---:|---:|
| 严格成功率 | 39.0% | 27.0% |
| 首工具选择 | 91.0% | 91.0% |
| 完整工具序列 | 91.0% | 91.0% |
| 参数 schema 合法 | 100.0% | 100.0% |
| 参数正确率 | 89.0% | 89.0% |
| Observation 完整性 | 100.0% | 100.0% |
| 格式合法 | 84.0% | 84.0% |
| 语义正确率 | 49.0% | 49.0% |
| 最终回答可溯源 | 40.0% | 40.0% |
| 安全终止 | 95.0% | 95.0% |
| 完成率 | 84.0% | 57.0% |
| 无循环 | 100.0% | 100.0% |

Raw 是模型直接结果；guarded 会拒绝策略不允许的轨迹，不能用于包装模型能力。安全策略没有改变工具路由、参数和语义指标，但降低了系统完成率。

### Day32 Prompt 消融

main Prompt 在 40 条开发集上严格通过 18 条，input-fidelity ablation 通过 14 条。主要错误是最终回答语义或可溯源失败，其次是工具调用格式问题；选错工具和参数错误以次要标签出现，死循环为 0。ablation 有 1 条改善、5 条回归，因此最终采用 main Prompt。

## 8. 安全边界与限制

- CodeExecutor 仅做 AST 静态检查，不执行任意代码；
- Calculator 使用受限解析器，不接受任意 Python；
- KnowledgeRetrieval 只读取固定本地 JSON；
- Agent 设置最大步骤、循环检测、结构化参数校验和安全终止；
- Guarded 结果与 raw 指标分栏报告；
- 模型大权重、checkpoint、缓存、凭据和平台运维记录不进入教师包。

当前主要能力限制是最终答案与 Observation 的语义对齐。虽然路由达到 91%，严格成功率只有 39%，不能将“会选工具”表述为“任务整体可靠”。后续应优先使用结构化解码、最终答案校验和针对 Observation-to-answer 的监督数据。

## 9. 老师验收矩阵

| 要求 | 状态 | 证据 |
|---|---|---|
| 三个工具可调用 | PASS | Day28 两工具验证 + Day30 CodeExecutor 与联合绑定 |
| 至少三阶段复杂任务 | PASS | 检索 → 计算 → 最终结论，逐步 trace 可审计 |
| Agent 错误分析与优化实验 | PASS | 40 条开发集、预定义 taxonomy、main/ablation 配对复测 |
| 第 6 周报告与最终归档 | PASS | 本报告、验收 CSV、输入 manifest、模型引用归档 |

这里的 PASS 表示老师要求的交付项和证据完整，并不表示所有内部质量指标达到生产门槛。内部 90% 严格成功率门槛未达到，已在结果和限制中明确说明。

## 10. 模型归档与复现

最终组合为：Week4 DPO merged model + Day31 checkpoint-38 LoRA + main Prompt + 三个固定工具。

复现顺序：

1. 核验基座 manifest；
2. 核验 checkpoint-38 adapter SHA-256；
3. 加载 `adapter_config.json` 并绑定三个工具；
4. 加载 main Prompt；
5. 使用冻结的确定性生成参数与最大 6 步运行；
6. 用提交的成功谓词分别计算 raw 与 guarded 指标。

Git 与教师包只保存 adapter 配置、manifest、远端引用、大小和 SHA，不复制约 40 MB 权重。`final_agent_archive.json` 和 `report_input_manifest.json` 分别描述最终组合和报告输入完整性。

## 11. 结论

Week6 完成了老师要求的 Agent 开发闭环：3 个安全工具、ReAct 单轮调用、三阶段复杂任务、100 条核心工具调用 SFT 数据、正式 LoRA 训练、错误模式分析、Prompt 单因素实验、周报和最终模型归档。

结果显示模型在工具 schema、Observation 完整性、循环控制和路由方面较稳定，但最终回答的语义正确与证据对齐仍不足。最终归档选择开发集表现最好的 checkpoint-38 与 main Prompt，并完整保留 raw、guarded、消融和逐题失败证据，使交付可以被独立复核。
