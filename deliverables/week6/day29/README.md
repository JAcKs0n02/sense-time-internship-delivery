# Day 29：加载 Week 4 DPO 模型并构建 ReAct Agent

## 状态

**已完成并验证（2026-08-26）。** 已在 AutoDL RTX 3090 上以 Week 3 合并基座加
Week 4 DPO PEFT 权重的等价运行方式加载指定 DPO 模型，实际构建 ReAct Agent、调用
Calculator，并由独立验证器给出 `PASS`。

执行依据：[第 6 周完整执行计划](../../../docs/week6_execution_plan.md) · [Day 28 基础工具](../day28/README.md)

## 老师要求

1. 加载第 4 周产出的 DPO 模型；
2. 使用 `create_react_agent` 构建 Agent；
3. 绑定 Calculator 与 KnowledgeRetrieval；
4. 测试单轮调用，例如“计算 123 * 456”；
5. 交付 Agent 初始化脚本和单轮日志。

## 实际模型血缘

| 字段 | 实际值 |
|---|---|
| model_id | 已核验的 Week4 DPO merged model |
| runtime mode | `base_plus_peft_merge` |
| base path | `/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged` |
| DPO adapter | 由 Week4 模型归档 manifest 绑定 |
| adapter SHA-256 | `d7a932ee4f28c8950db289126381f5d4dd30a037b238852e87ad8e5a241f9e52` |
| runtime identity SHA-256 | `d9be5c285591d7fcf0a8e91d5ec351ecefabe241d0372e1aa21d067e400d28ef` |
| archived merged lineage SHA-256 | `aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c` |

目标 GPU 实例没有保存 15 GB 的 Week 4 合并目录，因此正式运行没有把 Week 3 SFT
基座冒充为 DPO 模型，而是加载同一基座与 Day 20 的精确 DPO adapter，并在内存中执行
`merge_and_unload()`。验证器同时约束基座、adapter 文件、Week 4 源 manifest 和历史合并
模型归档 SHA；详见 [`model_preflight.json`](source/results/model_preflight.json)。

## 正式运行结果

- 固定输入：`计算 123 * 456`；
- 模型原始结构化调用：`calculator({"expression": "123*456"})`；
- 工具 Observation：`56088`；
- 最终回答：`计算结果为 56088`；
- 工具调用次数：1，未调用 KnowledgeRetrieval；
- Run Spec SHA-256：`05b465941d1051eba123b9a0f677e5e922b28d765c138e9f31b8ce16029db354`；
- [`day29_validation.json`](source/results/day29_validation.json) 为 `PASS`，所有 9 项
  绑定检查与 9 项消息检查通过，`failed_checks` 为空；
- Day 28 + Day 29 聚焦测试共通过 44 项，其中 Day 29 为 26 项。

## 专业术语

| 术语 | 解释 |
|---|---|
| ReAct Agent | 模型根据当前消息和 Observation 选择下一工具或给出最终回答的循环 |
| Model adapter | 将本地 Transformers/vLLM 模型包装成 Agent 所需聊天模型接口的适配层 |
| `bind_tools` | 把工具名称、描述和参数 schema 提供给聊天模型的方法 |
| Tool call | 模型生成的结构化工具名和参数，不是工具执行结果 |
| Observation | 工具执行后的真实返回，由运行器写回 Agent 状态 |
| Max steps | 一次任务最多允许的 Agent 决策/工具调用步数 |
| Loop guard | 检测重复 action/input 并终止无效循环的保护机制 |
| Lineage gate | 确保实际加载模型和 Week 4 归档模型完全一致的门禁 |

## API 兼容策略

LangGraph v1 官方已把 `create_react_agent` 标为弃用，但老师明确要求该 API。本项目优先验证并锁定：

```python
from langgraph.prebuilt import create_react_agent
```

精确 LangChain/LangGraph 版本、导入结果和最小调用 smoke 必须进入环境证据。不能在文档中写着 `create_react_agent`，实际却只调用其他构造器。

## 详细实施步骤

1. 在 AutoDL 重新验证 GPU、模型目录、文件 manifest 与离线加载；
2. 固定 LangChain/LangGraph 版本并运行 API 导入 smoke；
3. 实现本地聊天模型适配层，确认工具 schema 能传入模型；
4. 编写 Agent 工厂，仅绑定 Day 28 两个工具；
5. 固定 System Prompt、生成参数、最大步骤和超时；
6. 使用 fake model 测试工具绑定、轨迹记录和循环终止；
7. 正式运行一次“计算 123 * 456”；
8. 保存原始模型消息、Action、Action Input、Observation、最终回答、耗时与所有哈希；
9. 验证最终值为 56088 且仅调用 Calculator；
10. 同步小型证据后关闭 AutoDL，关机证据不进入教师提交。

## 计划文件结构

```text
deliverables/week6/source/week6_agent/
├── agent_factory.py
├── model_adapter.py
└── trace_schema.py

deliverables/week6/day29/source/
├── results/model_preflight.json
├── results/single_turn_trace.json
├── results/day29_validation.json
├── scripts/run_single_turn.py
├── scripts/validate_day29.py
└── tests/test_agent_factory.py
```

## 完成门禁

- [x] 基座、DPO adapter 与 Week 4 归档血缘均由哈希门禁验证；
- [x] DPO 权重在新进程中离线加载并在内存合并成功；
- [x] 实际调用 `create_react_agent`，版本与导入证据齐全；
- [x] Agent 只绑定两个已验证工具；
- [x] 单轮 Prompt、原始模型消息和完整工具轨迹已保存；
- [x] Action 为 Calculator，参数正确，Observation 与最终答案均为 56088；
- [x] 最大步骤、超时和循环保护测试通过；
- [x] 日志不包含凭据、SSH 信息或不可提交的长推理内容。

## 故障处理

- **模型缺失/哈希不符**：停止并恢复 Week 4 指定模型，不静默换模型。
- **模型不生成结构化工具调用**：先核对 chat template、工具 schema 和适配层；保留原始输出后再调整，不手工伪造 tool call。
- **API 已移除**：依据官方版本历史锁定仍提供该 API 的兼容版本，并记录原因；不同时维护两条未经验证的主路线。
- **计算结果由模型心算而未调用工具**：判为未满足老师要求，优化工具描述或 Prompt 后重新运行新版本案例。
- **Agent 循环**：触发最大步数并保存失败轨迹，不无限重试。

## 下一日交接

Day 30 接收模型 lineage、依赖锁、Agent 工厂、两个工具 schema、单轮验证和统一轨迹格式；新增 CodeExecutor 后必须重新计算三工具 schema SHA。
