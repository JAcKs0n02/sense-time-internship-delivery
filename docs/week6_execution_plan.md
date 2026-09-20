# 第 6 周 Agent 智能体开发完整执行计划

## 1. 文档用途

本文将老师新版 [`实习需求.pdf`](../实习需求.pdf) 中 Week 6 的 Day 28–Day 33 要求转化为可执行、可验证、可交付的工程计划。Day 28–Day 33 均已完成并通过证据一致性验证。

设计依据：[Week 6 Agent 交付设计](superpowers/specs/2026-08-24-week6-agent-deliverables-design.md)

实施计划：[Week 6 Agent Implementation Plan](superpowers/plans/2026-08-24-week6-agent-implementation.md)

## 2. 老师原始要求

| Day | 必做内容 | 老师交付 |
|---|---|---|
| 28 | 安装 LangChain；实现安全计算器；基于本地 JSON 实现知识检索 | 两个 Python 工具类 |
| 29 | 加载 Week 4 DPO 模型；使用 `create_react_agent`；测试单轮工具调用 | Agent 初始化脚本、单轮日志 |
| 30 | 实现只做 AST 语法检查的 CodeExecutor；运行复杂多步任务 | 多步推理日志、三个工具完整代码 |
| 31 | 构造 100 条 ReAct 数据；使用 LLaMA-Factory SFT | 工具调用 SFT 模型、训练数据 |
| 32 | 统计选错工具、参数错误、死循环；优化 Prompt/工具描述并复测 | Agent 错误模式分析报告 |
| 33 | 整理代码、日志和报告；撰写第 6 周报告 | 周报 |

周验收标准：三个工具可正常调用；Agent 能完成至少 3 步复杂任务；错误分析报告有深度；周报提交。

## 3. 当前已知输入

- GPU：既有 AutoDL RTX 3090 24GB；真正使用前重新验证。
- Week 4 DPO 模型：`reward_corrective_40step_merged`。
- 远端路径：`/root/autodl-tmp/qwen25-week4/best_model/qwen25-7b-week4-dpo-corrective-merged`。
- 模型 manifest SHA-256：`aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c`。
- 已验证环境：Day 28 本地 Python 3.11.14；Day 29 AutoDL Python 3.10.20；LangChain 0.3.30、LangChain Community 0.3.31、LangGraph 0.6.11。
- LLaMA-Factory 历史环境为 0.9.3；Day 31 以实际锁定 revision 的 parser 为准，不能直接假定旧配置仍兼容。

## 4. 当前进度

| Day | 状态 | 工程入口 |
|---|---|---|
| 28 | 已完成（PASS） | [LangChain 与两个基础工具](../deliverables/week6/day28/README.md) |
| 29 | 已完成（PASS） | [ReAct Agent](../deliverables/week6/day29/README.md) |
| 30 | 已完成（PASS） | [CodeExecutor 与多步推理](../deliverables/week6/day30/README.md) |
| 31 | 已完成（PASS） | [工具调用 SFT](../deliverables/week6/day31/README.md) |
| 32 | 已完成（PASS，v2 回归并不采用） | [错误分析与优化](../deliverables/week6/day32/README.md) |
| 33 | 已完成（PASS） | [周报与验收归档](../deliverables/week6/day33/README.md) |

## 5. 专业术语

| 术语 | 解释 |
|---|---|
| Agent | 由模型负责决策、由外部工具负责执行受限操作的智能体系统 |
| Tool | 带名称、描述、输入 schema 和确定输出的受控函数 |
| ReAct | 将推理路由与工具行动交替组织的方法；本项目日志只保留简短可审计的路由摘要 |
| Action | Agent 选择的工具名称 |
| Action Input | 传给工具的结构化参数 |
| Observation | 工具真实返回的结果，模型不得自行编造 |
| Tool schema | 工具参数名、类型、必填项和约束的机器可读定义 |
| AST | Python 抽象语法树；可在不运行代码的情况下检查语法结构 |
| Sandbox | 隔离并限制代码能力的环境；Day 30 并不执行代码，因此不能虚称为真实执行沙箱 |
| SFT | 使用输入和标准输出监督模型学习，本周用于增强工具选择与参数格式 |
| Function call | 模型输出的结构化工具调用消息 |
| Failure taxonomy | 对失败原因进行互斥或明确优先级分类的标签体系 |
| Dead loop | Agent 重复调用工具却无法达到终止条件 |
| Frozen evaluation | 运行前固定且训练/调参不可见的评测集合 |
| Lineage | 模型、数据、配置、代码、工具 schema 和结果哈希组成的血缘 |

## 6. 跨日数据流

```text
Day 28：Calculator + KnowledgeRetrieval + 固定知识库
  └─> Day 29：Week 4 DPO model + create_react_agent + 单轮日志
        └─> Day 30：CodeExecutor + 三工具 + 至少三步复杂轨迹
              └─> Day 31：100 条 ReAct SFT + 独立冻结评测 + adapter
                    └─> Day 32：baseline/SFT/Prompt 优化对比与错误报告
                          └─> Day 33：周报、四项验收矩阵、模型归档、教师投影
```

任何工具名称、输入字段或知识库条目发生变化，必须更新版本和 SHA，并使依赖该版本的轨迹、训练数据和评测结果失效。

## 7. 通用执行规则

1. 每日先写测试和冻结输入，再实现或运行；
2. 工具先独立验证，不能把 Agent 偶然答对当作工具正确；
3. 模型调用与工具执行分层记录，Observation 必须来自真实工具返回；
4. 固定最大步骤、超时、生成参数和随机种子；
5. 失败轨迹、异常和未通过指标如实保留；
6. 代码工具永不执行用户代码；计算器永不使用 `eval`；检索工具永不联网；
7. Git 只保存代码、配置、小型数据、日志、报告和 manifest；大权重留在 AutoDL；
8. 每次使用 AutoDL 后关机，但关机证据不进入老师提交；
9. `Submission/Week6/` 只从验证通过的工程证据投影，不直接在其中开发。

## 8. Day 28 实施摘要

- 建立隔离环境并记录 LangChain、LangGraph、Pydantic 与 Python 精确版本；
- 验证 `create_react_agent` 兼容导入路径，但本日不加载大模型；
- Calculator 使用 AST 白名单解释器，限制表达式长度、节点数、深度、指数和结果范围；
- KnowledgeRetrieval 读取固定 JSON 产品库，返回价格、运费、命中依据或 `not_found`；
- 测试正常输入、边界输入、恶意输入、空知识库、重复 ID、模糊命中和查无结果；
- 当日完成条件见 [Day 28 README](../deliverables/week6/day28/README.md)。

## 9. Day 29 实施摘要

- 重新核验 Week 4 DPO merged model 的路径、manifest 和离线加载；
- 使用真实 `create_react_agent` 绑定 Day 28 两个工具；
- 设置最大步骤、超时、循环保护和版本化 System Prompt；
- 固定测试 `计算 123 * 456`，保存模型消息、Action、参数、Observation `56088` 和最终回答；
- 当日完成条件见 [Day 29 README](../deliverables/week6/day29/README.md)。

## 10. Day 30 实施摘要

- CodeExecutor 只运行 `ast.parse` 和风险节点检查；
- 使用文件写入副作用测试证明代码没有执行；
- 冻结产品价格/运费/预算案例和预期工具序列；
- 复杂任务至少包含知识检索、总价计算、预算比较三个步骤；
- 单独验证 CodeExecutor 被正确路由和返回语法结论；
- 当日完成条件见 [Day 30 README](../deliverables/week6/day30/README.md)。

## 11. Day 31 实施摘要

- 构造正好 100 条训练数据：三个单工具各 20 条、多工具 25 条、纠错/恢复 15 条；
- 额外建立至少 20 条冻结评测，和训练数据做精确/近似污染检查；
- 内部审计保留 Thought/Action/Action Input/Observation；训练导出为 LLaMA-Factory 官方 ShareGPT tool-call schema；
- 运行真实 tokenizer 长度检查、parser smoke 和单步 GPU smoke；
- 从 Week 4 DPO merged model 进行 LoRA/QLoRA SFT，保存日志、有效配置和 adapter manifest；
- 正式 4-bit NF4 QLoRA 完成 3 epoch / 39 steps，训练退出码为 0；
- 30 条冻结评测中，首工具和完整序列正确率均由 86.67% 提升到 90.00%；参数正确率按 24 条适用题由 79.17% 提升到 91.67%，严格完成率由 80.00% 提升到 86.67%；
- 当日完成条件见 [Day 31 README](../deliverables/week6/day31/README.md)。

## 12. Day 32 实施摘要

- 冻结评测和错误 taxonomy 后运行 baseline；
- 分类选错工具、参数提取错误、死循环，并补充未处理工具错误、提前回答、编造 Observation 和格式错误；
- 只修改一个版本化 System Prompt 或工具描述集合；
- 使用相同评测、模型、工具和生成参数复测；
- 报告每类数量、代表 trace、根因、修改和前后差值，不隐藏无效优化；
- 当日完成条件见 [Day 32 README](../deliverables/week6/day32/README.md)。

## 13. Day 33 实施摘要

- 运行 Day 28–Day 32 全部验证器并冻结报告输入 manifest；
- 周报覆盖三工具、单轮、三步任务、100 条 SFT、错误分析、安全边界和限制；
- 建立老师四项验收矩阵；
- 归档模型路径、基座/数据/配置/adapter 哈希和加载 smoke；
- 创建精简 `Submission/Week6/` 并生成逐文件 SHA-256；
- 当日完成条件见 [Day 33 README](../deliverables/week6/day33/README.md)。

## 14. 最终交付清单

- 三个工具的完整 Python 代码与测试；
- Agent 初始化脚本和单轮原始日志；
- 至少三步复杂任务的原始轨迹；
- 正好 100 条 ReAct 工具调用训练数据；
- LLaMA-Factory 生效配置、训练日志和工具调用 SFT 模型 manifest；
- baseline/SFT/Prompt 优化的固定评测和错误模式报告；
- 《第 6 周：Agent 智能体开发报告》；
- Week 6 验收矩阵与提交校验清单。

## 15. 一手参考

- [LangChain Agents 官方文档](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph v1 迁移指南](https://docs.langchain.com/oss/python/migrate/langgraph-v1)
- [LLaMA-Factory 数据格式](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md)
- [Python AST 官方文档](https://docs.python.org/3/library/ast.html)

## 16. 本轮完成边界

Day 28–Day 33 均已有真实验证证据。Day 32 在相同 30 条冻结集上完成 Prompt-only 复测，v2 产生 1 条改善和 5 条回归，因此最终保留 v1；Day 33 已完成周报、四项验收矩阵、最终 Agent 引用归档和精简教师投影。
