# Day 28：LangChain 环境与两个基础工具

## 状态

**已完成并验证（2026-08-25）。** 两个受限工具、固定本地知识库、LangChain
封装和独立验证器均已实现；本状态只陈述已运行的本地证据。

执行依据：[第 6 周完整执行计划](../../../docs/week6_execution_plan.md) · [Week 6 工程索引](../README.md)

## 老师要求

1. 安装 LangChain 及其社区包；
2. 实现 `CalculatorTool`，安全执行数学表达式；
3. 实现 `KnowledgeRetrievalTool`，检索自行构建的本地 JSON 知识库；
4. 交付两个工具的 Python 类代码。

## 当日交付

- Python、LangChain、LangGraph、Pydantic 等环境与版本清单；
- `CalculatorTool` 类、输入 schema、白名单运算和安全限制；
- `KnowledgeRetrievalTool` 类、本地知识库与确定性检索规则；
- 正常、边界、恶意和错误输入的单元测试；
- 工具 schema、知识库和源码 SHA-256；
- Day 28 验证结果。

## 已验证证据

- 依赖锁定于 [`requirements.txt`](requirements.txt)：Python 3.11.14、LangChain
  0.3.30、LangChain Community 0.3.31、LangGraph 0.6.11 和 pytest 9.0.2；
  `pip check` 返回 `No broken requirements found.`
- `from langgraph.prebuilt import create_react_agent` 可导入。导入时观察到
  `LangChainPendingDeprecationWarning`（`allowed_objects` 的默认值将在未来变更），
  已如实记录在 [`environment.json`](source/results/environment.json)。
- [`day28_validation.json`](source/results/day28_validation.json) 为 `PASS`：它在
  仓库根目录独立导入两个类、重算知识库 SHA-256，并执行安全、拒绝、超限、别名和查无结果案例。
- 聚焦测试命令 `./.venv/bin/pytest -q deliverables/week6/day28/source/tests`
  实际通过 18 项测试。

## 专业术语

| 术语 | 解释 |
|---|---|
| LangChain | 用于组织模型、工具和 Agent 的应用框架 |
| LangGraph | LangChain Agent 底层可使用的有状态运行框架 |
| Tool schema | 工具名称、用途、参数类型和约束；模型依靠它选择并填写参数 |
| Pydantic | 用类型模型验证工具输入和结构化输出的库 |
| AST | Python 抽象语法树；可以检查表达式结构而不运行任意代码 |
| 白名单 | 只允许明确列出的节点和运算，未列出的全部拒绝 |
| Deterministic retrieval | 相同知识库和查询始终返回相同排序结果的检索方式 |
| Structured error | 带固定状态和错误码的错误返回，便于 Agent 判断是否重试 |

## 安全设计

### CalculatorTool

只允许数字常量、括号、一元正负和预注册算术操作。使用 `ast.parse(expression, mode="eval")` 后递归解释节点，不使用 `eval`。必须限制：

- 输入字符数；
- AST 节点数与嵌套深度；
- 整数位数和指数上限；
- 除零、非有限值和结果绝对值；
- 计算耗时。

名称、函数调用、属性、下标、导入、容器和字符串全部拒绝。`__import__`、文件操作、网络、shell 和子进程必须在测试中证明无法触发。

### KnowledgeRetrievalTool

知识库路径由构造函数固定，不从用户输入读取路径。每条记录至少包含：

```text
product_id,name,aliases,price_cny,shipping_cny,stock,description
```

检索先做标准化精确名/别名匹配，再做可解释关键词匹配。并列结果返回候选列表；无结果返回 `not_found`。不联网，不编造价格，不包含真实客户或个人数据。

## 详细实施步骤

1. 建立 Week 6 隔离环境，记录 Python 与包版本；
2. 验证 `create_react_agent` 的实际导入路径和弃用状态，只记录事实；
3. 先编写 Calculator 正常/拒绝/超限测试；
4. 实现 AST 白名单解释器并通过测试；
5. 构建用于后续多步任务的虚构产品知识库；
6. 编写知识库 schema、唯一性和检索测试；
7. 为两个类增加 LangChain Tool 接口和 Pydantic 参数 schema；
8. 生成 manifest 并运行联合验证；
9. 仅在验证通过后更新状态并投影老师所需代码。

## 计划文件结构

```text
deliverables/week6/day28/source/
├── data/knowledge_base.json
├── results/environment.json
├── results/day28_validation.json
├── scripts/validate_day28.py
└── tests/
    ├── test_calculator.py
    └── test_knowledge_retrieval.py

deliverables/week6/source/week6_agent/tools/
├── calculator.py
└── knowledge_retrieval.py
```

## 完成门禁

- [x] 环境版本来自实际安装，`pip check` 无冲突；
- [x] 两个工具类可独立导入和调用；
- [x] Calculator 正确计算 `123 * 456 = 56088`；
- [x] Calculator 的恶意表达式、除零和资源超限均被结构化拒绝；
- [x] 源码中不存在 `eval`、`exec`、`subprocess` 或 shell 执行路径；
- [x] 知识库 JSON schema、ID、价格和 SHA 校验通过；
- [x] 精确命中、别名命中、歧义和查无结果行为均有测试；
- [x] 老师要求的两个 Python 类、LangChain 输入 schema/封装和必要依赖说明齐全。

## 故障处理

- **API 导入失败**：保存版本与堆栈，依据官方迁移说明选择兼容版本；不猜测导入路径。
- **Calculator 可执行名称/调用**：停止交付，收紧 AST 白名单并增加失败用例。
- **大整数或指数导致卡顿**：在运算前检查位数、指数和节点规模，而不是只靠超时。
- **检索返回多个产品**：返回候选并要求澄清，不自动选择价格更低或排名第一的产品。
- **知识库内容缺失**：返回 `not_found`，不让模型从常识补写价格。

## 下一日交接

Day 29 只接收通过验证的两个工具类、知识库 SHA、工具 schema SHA、环境版本和测试结果。工具名称或参数字段变化时，Day 29 必须重新绑定和复测。
