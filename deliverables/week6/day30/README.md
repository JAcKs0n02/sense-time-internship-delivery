# Day 30：AST-only CodeExecutor 与多步推理

## 状态

**已完成。** 本地 57 项 Week6 回归测试通过；AutoDL 上使用已校验的 Week4 DPO adapter 完成真实三工具 Agent 运行。老师指定多步案例通过，CodeExecutor 危险代码案例通过且未产生文件副作用。

执行依据：[第 6 周完整执行计划](../../../docs/week6_execution_plan.md) · [Day 29 ReAct Agent](../day29/README.md)

## 老师要求

1. 实现第三个工具 `CodeExecutor`；
2. 只使用 Python AST 做语法检查，不真正执行代码；
3. 测试多步复杂任务，例如查询产品价格、加运费、计算总价、判断是否超过 1000 元；
4. 交付多步推理日志和三个工具完整代码。

## 术语与安全边界

| 术语 | 解释 |
|---|---|
| Syntax check | 判断 Python 源码能否被解析成 AST，并定位语法错误；不运行代码 |
| Risk node | 在本项目策略中需要拒绝或提示的 AST 节点，例如 Import、Call、Attribute |
| Side-effect test | 用文件、进程或网络标记证明被检查代码没有实际执行 |
| Multi-step trace | 包含多个有顺序依赖的 Action、Observation 和最终回答的轨迹 |
| Success predicate | 预先定义的机器判定规则，例如工具序列、总价和预算结论均正确 |
| Premature answer | 未获得必要 Observation 就提前生成结论 |

`CodeExecutor` 只是沿用老师命名。对外描述必须写“AST 代码检查工具”，不能写成“安全执行任意 Python”。禁止 `eval`、`exec`、`compile` 后执行、`subprocess`、shell、容器执行和网络调用。

## CodeExecutor 输出合同

所有输出沿用共享 `ToolResult(status, data, error_code)`：

```text
status: success | rejected
error_code: null | invalid_source | syntax_error | risk_nodes_detected | limit_exceeded
data.syntax_valid: boolean
data.error_line/error_column/error_message: nullable
data.node_count/max_depth: integer
data.risk_nodes: list[string]
```

输入长度、AST 节点数和深度必须受限。通过语法检查不代表代码安全，也不代表代码可以运行。

## 多步任务设计

老师示例固定为一个虚构产品，知识库中预先保存价格和运费。成功轨迹至少包含：

1. KnowledgeRetrieval：找到正确产品、价格和运费；
2. Calculator：使用检索结果计算总价；
3. Final reasoning：将总价与 1000 元比较并引用工具结果回答。

另建至少两个不同产品/预算的变体，以及一个需要 CodeExecutor 的独立任务。所有案例在模型运行前冻结 expected tool sequence、关键参数和成功谓词。

## 详细实施步骤

1. 先写 AST 语法正确、语法错误、风险节点、超限和副作用测试；
2. 实现解析、节点遍历、风险分类和结构化返回；
3. 用恶意文件写入字符串证明检查过程不产生文件；
4. 将第三个工具绑定到 Agent，更新工具 schema SHA；
5. 冻结复杂任务、知识库条目、预期步骤和预算答案；
6. 先 smoke 一条，再运行全部正式多步案例；
7. 保存每一步 Action/Input/Observation、最终回答、步骤数、耗时和终止原因；
8. 检查重复调用、错误恢复、提前回答和编造 Observation；
9. 生成三工具完整代码清单和 Day 30 验证结果。

## 计划文件结构

```text
deliverables/week6/source/week6_agent/tools/code_executor.py
deliverables/week6/day30/source/
├── configs/agent_config.json
├── data/multistep_cases.json
├── results/multistep_traces.jsonl
├── results/multistep_summary.json
├── results/day30_validation.json
├── results/teacher_multistep_trace.json
├── results/code_executor_safety_trace.json
├── results/teacher_acceptance.json
├── results/transitive_integrity_audit.json
├── scripts/initialize_day30.py
├── scripts/run_multistep.py
├── scripts/validate_day30.py
└── tests/
    ├── test_code_executor.py
    └── test_multistep_trace.py
```

## 已落实的安全设计

- `ast.parse(..., mode="exec")` 只建立语法树，不产生可执行对象；
- 没有 `eval`、`exec`、`compile`、`subprocess`、shell、网络或任意路径读取入口；
- `Import`、`ImportFrom`、`Call`、`Attribute`、`With`、`AsyncWith`、`Global` 和 `Nonlocal` 会触发结构化拒绝；
- 源码长度上限为 4096 字符，AST 节点数和深度沿用共享限制 128/32；
- 专门的副作用测试把文件写入语句作为字符串输入，并断言目标文件不存在。

## 冻结正式案例

| 案例 | 工具序列 | 预期结果 |
|---|---|---|
| 老师示例：星云机械键盘 | `knowledge_retrieval → calculator` | `699 + 20 = 719`，未超过 1000，剩余 281 |
| 彗星鼠标 | `knowledge_retrieval → calculator` | `249 + 12 = 261`，超过 250，共超 11 |
| 轨道耳机 | `knowledge_retrieval → calculator` | `899 + 0 = 899`，未超过 1000，剩余 101 |
| 合法 AST | `code_executor` | 语法正确，无风险节点 |
| 文件写入字符串 | `code_executor` | 语法正确但含 `Call/Attribute`，拒绝且无副作用 |

工具调用是可审计的 Action/Observation；最终预算比较是第三个可审计阶段。文档不记录或声称模型的隐藏思维链。

## 完成门禁

- [x] CodeExecutor 可区分合法语法、语法错误、风险结构和资源超限；
- [x] 副作用测试证明代码没有执行；
- [x] 三个工具的类代码、输入 schema 和测试齐全；
- [x] 正式多步案例在运行前冻结且未因结果删题；
- [x] 老师示例包含至少三个可审计阶段；
- [x] 价格、运费、总价和预算结论均可从工具返回重算；
- [x] 最大步骤、超时和重复调用保护已有本地回归测试；
- [x] 完整原始轨迹和失败轨迹均已保存。

## 正式结果

老师示例真实轨迹为：

1. `knowledge_retrieval({"query":"星云机械键盘"})` 返回价格 699 元、运费 20 元；
2. `calculator({"expression":"699 + 20"})` 返回 719；
3. 最终回答明确说明总价 719 元，没有超过 1000 元预算。

教师要求口径为 `PASS`，见 `teacher_acceptance.json` 和 `teacher_multistep_trace.json`。原始 run spec 直接绑定模型身份、三个工具源码/schema、Day30 Prompt、冻结案例和 Day30 脚本；共享运行模块与知识库文件另由运行后 `transitive_integrity_audit.json` 补充审计。两种证据范围明确区分，不把运行后审计倒写成原始 run spec 字段。

`requirements.txt` 记录正式 AutoDL 运行实际版本；本地 Python 3.11 回归环境使用 Day28 锁定依赖，二者不能混称为同一环境。

额外压力测试共 5 条，通过 3 条：`orbit_within_budget` 因 0 元运费而跳过 Calculator；`code_ast_valid` 在 CodeExecutor 后误调用 KnowledgeRetrieval。两条失败均保留在内部 `multistep_traces.jsonl` 与 `day30_validation.json` 中，不重跑、不删题，也不把扩展集 60% 写成全通过。老师提交目录只投影必需的多步日志、CodeExecutor 安全证据和三个工具代码。

## 故障处理

- **AST 检查意外执行代码**：立即停止，移除执行路径并添加副作用回归测试。
- **模型跳过检索直接编价格**：判为失败，强化 Observation 约束，不把答案巧合正确视为通过。
- **工具参数不是合法 JSON**：保存原始输出并标记格式错误，不能人工改写后算成功。
- **复杂任务只有两个步骤**：按冻结步骤定义判为未通过，不通过拆分日志文字虚增步数。
- **循环或超时**：由运行器终止并记录 `dead_loop`/`timeout`，不无限监控或重试。

## 下一日交接

Day 31 必须使用经过 Day 30 验证的三个工具名称、描述、输入 schema 和 Observation 格式构造训练数据；任何接口改变都需要重建数据。
