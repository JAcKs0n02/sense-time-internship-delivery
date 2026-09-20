# Day 29：ReAct Agent

## 交付结论

已使用 Week 4 DPO 权重完成真实单轮 Agent 运行并通过独立验证：

- Prompt：`计算 123 * 456`
- Action：`calculator`
- Action Input：`{"expression":"123*456"}`
- Observation：`56088`
- Final Answer：`计算结果为 56088`
- 验证结果：`PASS`，`failed_checks` 为空

目标实例没有保存 15 GB 的 Week 4 合并目录，因此正式运行采用等价的
“Week 3 合并基座 + Day 20 精确 DPO PEFT adapter + 内存合并”方式。adapter
SHA-256 为 `d7a932ee4f28c8950db289126381f5d4dd30a037b238852e87ad8e5a241f9e52`，
验证器同时绑定历史 Week 4 合并模型归档血缘，没有使用 SFT-only 模型替代。

## 文件说明

| 目录/文件 | 内容 |
|---|---|
| `Code/week6_agent/` | Agent 工厂、模型适配器、轨迹格式和两个工具 |
| `Config/agent_config.json` | 冻结的正式运行配置 |
| `Scripts/initialize_agent.py` | Agent 初始化和模型预检脚本 |
| `Scripts/run_single_turn.py` | 正式单轮运行脚本 |
| `Scripts/validate_day29.py` | 独立验证脚本 |
| `Results/single_turn_trace.json` | 老师要求的真实单轮日志 |
| `Results/day29_validation.json` | 最终验证结果（PASS） |
| `Results/` 其他 JSON | 环境、模型、源码、工具和 Run Spec 证据 |
| `Provenance/Adapter_Manifest.json` | Week 4 DPO adapter 来源证据 |
| `Submission_Map.json` | 扁平文件与正式实验源文件的哈希映射 |

`Scripts/` 中是正式 GPU 运行时使用的字节一致原件，其模型血缘与源码哈希已写入
`Results/`。它们依赖正式工程相对路径以及 AutoDL 上的本地模型，不是脱离工程与权重
即可运行的演示脚本。正式复现需要对应的基座与 DPO adapter；模型权重未纳入教师提交。
项目内部测试保留在正式工程目录中，不重复放入教师提交。
