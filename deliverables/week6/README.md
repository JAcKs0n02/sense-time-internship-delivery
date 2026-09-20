# Week 6 Engineering Archive

Week 6 从模型训练转向 Agent 智能体工程：让 Week 4 DPO 模型在明确安全边界内选择计算器、本地知识库和 AST 检查工具，完成单轮与多步任务，再通过 100 条 ReAct 工具调用数据进行 SFT，最后执行错误分析和周报归档。

## 执行入口

- [第 6 周完整执行计划](../../docs/week6_execution_plan.md)
- [Week 6 交付设计](../../docs/superpowers/specs/2026-08-24-week6-agent-deliverables-design.md)
- [老师新版需求 PDF](../../实习需求.pdf)

## 当前状态

| Stage | Status | Planned engineering deliverable | Teacher requirement |
|---|---|---|---|
| Day 28 | 已完成（PASS） | [安全 Calculator 与本地 JSON KnowledgeRetrieval](day28/README.md) | 两个工具 Python 类 |
| Day 29 | 已完成（PASS） | [Week 4 DPO 模型、ReAct Agent 与单轮日志](day29/README.md) | 初始化脚本、单轮测试日志 |
| Day 30 | 已完成（PASS） | [AST-only CodeExecutor 与三步任务](day30/README.md) | 多步日志、三个工具完整代码 |
| Day 31 | 已完成（PASS） | [100 条核心数据、200 条扩展数据、LoRA 训练与 100 条最终测试](day31/README.md) | 工具调用 SFT 模型、训练数据 |
| Day 32 | 已完成（PASS） | [失败模式统计与单因素 Prompt 消融](day32/README.md) | Agent 错误模式分析报告 |
| Day 33 | 已完成（PASS） | [第 6 周报告、验收矩阵和模型归档](day33/README.md) | 周报 |

## 固定口径

| 项目 | 规划口径 |
|---|---|
| Agent 模型 | 已核验的 Week 4 DPO merged model |
| 模型加载 | Day 31 使用逐文件核验的 Week 4 DPO merged model；仅在 merged 不可用时回退到 Week 3 基座 + Day 20 精确 DPO adapter |
| Agent API | 优先锁定仍提供 `langgraph.prebuilt.create_react_agent` 的兼容版本 |
| 工具数量 | 3：Calculator、KnowledgeRetrieval、CodeExecutor |
| 代码工具边界 | AST 解析/风险检查，不执行代码 |
| 复杂任务 | 至少 3 个可审计步骤，且设置最大步骤与循环保护 |
| SFT 数据 | 100 条核心训练数据 + 200 条扩展训练数据；40 条开发集；100 条最终测试 |
| 状态原则 | 缺少原始日志、模型 lineage 或语义验证即保持未完成 |

## 跨日接口

- Day 28 产生两个工具、知识库、工具 schema 与单测；
- Day 29 只消费 Day 28 通过验证的工具，并绑定 Week 4 DPO 模型；
- Day 30 增加第三个工具，复用同一 Agent 工厂与轨迹 schema；
- Day 31 使用三工具的真实名称、描述和参数 schema 构造训练数据；
- Day 32 使用固定 checkpoint、开发集和工具代码比较 main Prompt 与单规则 ablation；
- Day 33 只汇总 Day 28–Day 32 已验证事实。

## 目录边界

- `deliverables/week6/`：完整工程说明，后续加入代码、配置、小型数据、原始结果、测试和审计证据；
- `deliverables/week6/source/week6_agent/`：三工具、模型适配、Agent 工厂、运行限制和统一轨迹 schema；
- `Submission/Week6/`：包含已验证的 Day 28–Day 33 教师交付投影，不包含整个仓库；
- `/root/autodl-tmp/`：模型、checkpoint、缓存和大文件；
- Git 与教师提交均不得包含凭据、SSH 信息、真实个人数据、模型大权重或平台关机证据。

## 更新规则

1. 每日开始前读取总计划和当天 README；
2. 工具 schema、Prompt、评测案例和成功判据必须在正式运行前冻结；
3. 先单测工具，再运行 Agent；
4. 所有 Observation 必须来自真实工具返回；
5. 不删除失败轨迹，不把 train loss 当作工具准确率；
6. 当日验证通过后再更新状态和创建教师投影。
