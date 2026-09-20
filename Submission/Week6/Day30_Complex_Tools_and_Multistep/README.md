# Day 30：复杂工具与多步任务

## 交付结论

老师要求已完成：

- `CodeExecutor` 只调用 `ast.parse` 做语法与风险节点检查，不执行输入代码；
- 三个工具 `Calculator`、`KnowledgeRetrieval`、`CodeExecutor` 的完整源码均在 `Code/week6_agent/tools/`；
- 真实 Week4 DPO 模型完成“查询价格 → 加运费 → 计算总价 → 比较预算”任务；
- 老师指定案例结果为：699 元 + 20 元运费 = 719 元，未超过 1000 元预算；
- 危险文件写入语句被识别为 `Call/Attribute` 风险并拒绝，目标文件未产生。

教师要求验收状态见 `Results/teacher_acceptance.json`，结论为 `PASS`。

## 文件说明

| 路径 | 内容 |
|---|---|
| `Code/week6_agent/tools/` | 三个工具的完整实现 |
| `Code/week6_agent/day30_agent.py` | 三工具 ReAct Agent、System Prompt 与证据清单 |
| `Code/week6_agent/day30_validation.py` | 从 Observation 复算结果的语义验证器 |
| `Data/multistep_cases.json` | 模型运行前冻结的 5 条案例及预期值 |
| `Results/teacher_multistep_trace.json` | 老师指定案例的完整消息、Action、Input、Observation 与最终回答 |
| `Results/code_executor_safety_trace.json` | CodeExecutor 拒绝危险代码且无副作用的真实轨迹 |
| `Results/teacher_acceptance.json` | 按老师要求整理的验收结果及源码 SHA-256 |
| `Results/model_preflight.json` | Week4 DPO adapter 与基座模型身份校验 |
| `Results/run_spec.json` | 冻结配置、用例、工具和 Prompt 的运行规格 |
| `Results/transitive_integrity_audit.json` | 对共享运行模块和知识库文件的运行后补充哈希审计 |
| `Scripts/` | 初始化、正式运行和独立验证脚本 |
| `Submission_Map.json` | 本提交文件到仓库正式证据的逐文件映射 |

## 多步轨迹

本项目只记录可审计的工具步骤，不声称或导出模型隐藏思维链：

1. `knowledge_retrieval({"query":"星云机械键盘"})`：返回价格 699 元、运费 20 元；
2. `calculator({"expression":"699 + 20"})`：返回 719；
3. 最终回答：总价 719 元，没有超过 1000 元预算。

每条工具调用均保存模型原始输出、结构化参数和工具真实 Observation，可由验证器重新计算。原始 run spec 直接绑定模型、三个工具、Prompt、案例和 Day30 脚本；共享模块与知识库文件是运行后补充审计，二者没有混写。

## CodeExecutor 安全边界

- 输入只经过 AST 解析与遍历，不调用 `eval`、`exec` 或执行后的 `compile`；
- 不提供 shell、子进程、网络或任意文件读取接口；
- 源码长度、AST 节点数和深度均有上限；
- `Import`、`ImportFrom`、`Call`、`Attribute`、`With`、`AsyncWith`、`Global`、`Nonlocal` 被列为风险节点；
- “语法正确”仅表示能建立 AST，不代表代码可以安全执行。

## 扩展测试说明

除老师指定案例外还冻结了 4 条压力案例，完整扩展集通过 3/5。两个保留失败分别是：0 元运费时模型跳过 Calculator；合法代码检查后模型误调用 KnowledgeRetrieval。它们不影响老师指定案例和 CodeExecutor 安全边界的验收，但如实记录在 `teacher_acceptance.json`，没有删题或伪造全通过。

## 复现

`requirements.txt` 是 AutoDL 正式运行的实际版本快照，其中 CUDA 版 Torch 需要按目标平台选择对应软件源，不能视为跨平台的一键安装文件。

`Scripts/` 是与正式工程字节一致的证据副本，保留了仓库相对路径约定；应先按 `Submission_Map.json` 恢复仓库布局，再在含对应模型和 adapter 的环境运行：

```bash
python Scripts/initialize_day30.py --help
python Scripts/run_multistep.py --help
python Scripts/validate_day30.py --help
```

正式运行需要传入基座模型路径、Week4 DPO adapter 路径、adapter 来源 manifest、输出目录和冻结的 run spec。模型权重未放入教师提交。
