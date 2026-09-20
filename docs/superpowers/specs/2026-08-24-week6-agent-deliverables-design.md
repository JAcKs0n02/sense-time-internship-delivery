# Week 6 Agent 智能体交付设计

## 1. 目标

将新版 `实习需求.pdf` 中 Week 6 的 Day 28–Day 33 要求转换为可逐日执行、可验证、可独立提交的工程结构。本轮只建立计划和 Markdown 入口，不安装依赖、不启动模型、不生成实验结果，也不提前创建教师正式提交目录。

Week 6 的最终目标是：

1. 实现 `CalculatorTool`、`KnowledgeRetrievalTool` 和 `CodeExecutor` 三个受限工具；
2. 使用 Week 4 最终 DPO 模型和 `create_react_agent` 构建 Agent；
3. 留存单轮工具调用以及至少 3 步复杂任务的完整轨迹；
4. 构造正好 100 条工具调用 SFT 训练数据并完成 LLaMA-Factory 微调；
5. 用冻结评测集分析选错工具、参数错误和死循环等失败模式，并复测优化效果；
6. 完成《第 6 周：Agent 智能体开发报告》。

## 2. 老师要求映射

| Day | 老师任务 | 老师交付 | 工程证据门禁 |
|---|---|---|---|
| 28 | 安装 LangChain；实现计算器与本地 JSON 知识库检索 | 两个工具的 Python 类代码 | 依赖版本锁定；工具 schema、正常/拒绝/边界测试齐全；无 `eval`、无网络检索 |
| 29 | 加载 Week 4 DPO 模型；用 `create_react_agent` 绑定两个工具；测试单轮调用 | 初始化脚本、单轮日志 | DPO lineage 与远端路径匹配；API 导入 smoke；日志含原始输入、工具名、参数、Observation 和最终回答 |
| 30 | 实现只做 AST 检查的 `CodeExecutor`；测试复杂任务 | 多步日志、三个工具完整代码 | 不执行代码；至少 3 个可审计步骤；最大步数和超时生效；三工具单测通过 |
| 31 | 构造 100 条 ReAct 数据；用 LLaMA-Factory 做 SFT | 工具调用 SFT 模型、训练数据 | 正好 100 条训练数据；额外冻结评测不计入训练；官方 tool-call schema smoke、日志、adapter 清单齐全 |
| 32 | 统计失败模式；优化 System Prompt 或工具描述并复测 | 《Agent 错误模式分析报告》 | 同一冻结评测集做前后对比；至少覆盖老师指定三类；原始失败轨迹和修复证据可追溯 |
| 33 | 整理代码、日志、报告并撰写周报 | 周报 | 四项周验收逐条映射；只汇总实际证据；模型与教师包清单完整 |

## 3. 范围与边界

### 3.1 本轮创建或更新

```text
docs/week6_execution_plan.md
deliverables/week6/README.md
deliverables/week6/day28/README.md
deliverables/week6/day29/README.md
deliverables/week6/day30/README.md
deliverables/week6/day31/README.md
deliverables/week6/day32/README.md
deliverables/week6/day33/README.md
README.md
```

### 3.2 本轮不创建

- 不创建空的 `Submission/Week6/`；
- 不创建占位 Python、JSON、训练配置、日志、模型 manifest 或报告；
- 不安装或升级 LangChain/LangGraph，不运行 GPU，不启动 AutoDL；
- 不把预期准确率、训练 loss 或 PASS 写成实验事实；
- 不修改老师的 PDF/Word 内容。

## 4. 总体架构

后续代码采用一个共享、可测试的 Agent 包，而不是按日期复制三份工具实现：

```text
deliverables/week6/source/week6_agent/
├── tools/
│   ├── calculator.py
│   ├── knowledge_retrieval.py
│   └── code_executor.py
├── agent_factory.py
├── model_adapter.py
├── trace_schema.py
└── limits.py
```

Day 28 创建并独立验证前两个工具；Day 29 增加模型适配和 Agent 工厂；Day 30 增加第三个工具及多步运行器；Day 31 复用同一工具 schema 生成训练数据；Day 32 复用同一冻结评测器；Day 33 只汇总验证后的事实。

## 5. 模型与依赖血缘

Day 29 必须加载 Week 4 已归档的最终 DPO merged model：

```text
model_id: reward_corrective_40step_merged
remote_path: /root/autodl-tmp/qwen25-week4/best_model/qwen25-7b-week4-dpo-corrective-merged
manifest_sha256: aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c
```

执行时重新验证路径、文件清单、离线加载和 manifest SHA。缺少权重时不得静默退回基座或 Week 5 VLM。

LangGraph v1 已将 `create_react_agent` 标记为弃用并推荐 `langchain.agents.create_agent`，但老师明确要求前者。因此正式路线优先锁定仍可导入 `langgraph.prebuilt.create_react_agent` 的兼容版本，并保存精确版本与导入 smoke；只有老师要求允许变更时才迁移 API。参考：[LangGraph v1 迁移指南](https://docs.langchain.com/oss/python/migrate/langgraph-v1)、[LangChain Agents 官方文档](https://docs.langchain.com/oss/python/langchain/agents)。

## 6. 三个工具的安全合同

### 6.1 CalculatorTool

- 输入只允许单个数学表达式；
- 使用 `ast.parse(..., mode="eval")` 解析并递归解释白名单节点；
- 只允许有限数字、括号和预注册算术运算；
- 禁止名称、属性、函数调用、下标、容器、导入和任意代码执行；
- 设置输入长度、AST 节点数、深度、整数位数、指数和结果幅度上限；
- 除零、超限和非有限结果返回结构化错误，不抛出未处理异常；
- 禁止 `eval`、`exec`、`subprocess` 和 shell。

### 6.2 KnowledgeRetrievalTool

- 只读项目内固定的本地 JSON；不接受任意文件路径；
- 知识库至少包含可支持 Day 30 任务的产品名、别名、价格、运费和描述；
- 启动时验证 schema、唯一 ID、非负价格和 SHA-256；
- 采用确定性的精确别名与关键词匹配；并列时返回候选而不是猜测；
- 查无结果时明确返回 `not_found`，不得编造产品或价格；
- 不访问互联网、数据库或用户隐私数据。

### 6.3 CodeExecutor

老师已明确“非真正执行”。因此该名称只表示代码检查工具：

- 使用 `ast.parse` 检查 Python 语法；
- 返回 `syntax_valid`、错误行列、节点统计和风险节点；
- 可拒绝 `Import`、`Call`、`Attribute` 等超出安全策略的结构；
- 永远不调用 `compile` 后执行，不使用 `eval`、`exec`、`subprocess` 或 shell；
- 文档、工具描述和最终报告均不得把它表述成真实代码执行沙箱。

Python 官方文档只保证 AST 解析/字面量能力，不等同于安全执行环境，参考：[Python `ast` 文档](https://docs.python.org/3/library/ast.html)。

## 7. Agent 与轨迹合同

Agent 每次运行必须固定：模型 lineage、工具 schema SHA、System Prompt SHA、生成参数、最大步骤、超时和随机种子。统一轨迹至少包含：

```text
run_id,case_id,model_id,input,step_index,thought_summary,
action,action_input,observation,status,latency_seconds,final_answer,
model_manifest_sha256,tools_sha256,system_prompt_sha256
```

`thought_summary` 只记录简短的路由理由或训练用结构化标签，不保存或要求模型暴露不可审计的私有长推理。日志不得包含 token、密码、SSH 信息或平台关机证据。

Day 30 的老师示例至少产生三个可审计阶段：

1. `KnowledgeRetrievalTool` 查询产品价格和运费；
2. `CalculatorTool` 计算总价；
3. Agent 将总价与 1000 元预算比较并生成有依据的最终回答。

此外单独验证 `CodeExecutor` 被正确选择。Agent 设置有限最大步数，重复相同 action/input 时触发循环保护。

## 8. Day 31 数据与训练设计

正式训练集正好 100 条，建议冻结分布：

| 类型 | 数量 |
|---|---:|
| Calculator 单工具 | 20 |
| KnowledgeRetrieval 单工具 | 20 |
| CodeExecutor 单工具 | 20 |
| Calculator + Knowledge 多步 | 25 |
| 参数纠错、工具异常恢复与循环终止 | 15 |
| 合计 | 100 |

另建至少 20 条冻结评测，不计入 100 条训练数据，且不得与训练 Prompt 精确或近似重复。内部审计格式保留老师要求的 `Thought/Action/Action Input/Observation`；训练导出使用 LLaMA-Factory 官方 ShareGPT tool-call 角色：`human`、`function_call`、`observation`、`gpt`，并提供 `tools` 列。官方格式说明见 [LLaMA-Factory 数据文档](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md)。

训练前必须完成：JSON/schema 验证、100 条计数、ID 唯一、角色交替、工具名与参数 schema 检查、知识库引用检查、污染检查、真实 tokenizer 长度统计和 LLaMA-Factory parser smoke。

训练使用 Week 4 DPO merged model 作为 SFT 起点，优先采用 LoRA/QLoRA 并记录基座、数据、配置、代码 revision、日志、adapter 文件大小和 SHA。准确率提升只能通过同一冻结评测集的 pre-SFT/post-SFT 对比判断，不能用 train loss 替代。

## 9. Day 32 评测与错误分析

冻结评测表至少记录：预期工具序列、预期关键参数、实际工具序列、实际参数、是否完成、步骤数、是否循环和失败标签。失败 taxonomy 至少包含：

- `wrong_tool`：选错工具；
- `argument_extraction_error`：参数缺失、字段错或值错；
- `dead_loop`：重复调用且无法终止；
- `tool_error_unhandled`：工具返回错误后未恢复；
- `premature_answer`：未获得必要 Observation 就回答；
- `hallucinated_observation`：把不存在的工具结果当事实；
- `format_parse_error`：Action/参数格式无法解析。

先冻结 baseline，再只修改一个版本化 System Prompt 或工具描述，使用相同模型、相同测试和相同生成配置复测。报告给出每类数量、代表案例、根因、修改内容和前后差值；若优化无效，必须如实保留。

## 10. 教师提交边界

`Submission/Week6/` 只在对应实验实际完成并验证后增量创建。预计最终包含：

```text
Submission/Week6/
├── Day28_LangChain_and_Basic_Tools/
├── Day29_ReAct_Agent/
├── Day30_Multi_Step_Reasoning/
├── Day31_Tool_Call_SFT/
├── Day32_Error_Analysis/
└── Day33_Weekly_Report/
```

教师材料不包含整个仓库、模型大权重、checkpoint、缓存、账号、SSH 信息、私有映射或 AutoDL 关机证据。模型只提交可复现 manifest、远端路径、哈希和加载证明。

## 11. 完成定义

本轮“Week 6 Markdown 整理完成”只表示老师要求、跨日依赖、安全边界、数据合同、验证门槛和预计目录已经冻结。Day 28–Day 33 均保持“未开始”；它不表示工具、Agent、数据、训练、错误分析或周报已经执行。
