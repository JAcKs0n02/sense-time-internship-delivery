# Day 32：Agent 错误分析与 Prompt 消融

## 任务与结论

本日按照老师要求统计 Agent 失败模式，并只改变一条 System Prompt 规则进行配对复测。模型固定为 Day31 checkpoint-38，开发集固定为 40 条，生成参数与工具实现保持不变。

结论：继续采用 main Prompt。input-fidelity ablation 的严格成功率从 45.0% 降至 35.0%，语义正确率从 50.0% 降至 47.5%；共有 1 条改善、5 条回归。

## 专业术语

| 术语 | 含义 |
|---|---|
| Error taxonomy | 预先定义的错误分类规则，用同一规则分析全部轨迹 |
| Primary label | 一条失败轨迹的主要错误类型，按固定优先级确定 |
| Secondary label | 同一轨迹同时出现的其他错误现象 |
| Ablation | 只改变一个因素，以确认指标变化来自哪个因素 |
| Regression | 原本通过的案例在新版本中失败 |
| Strict success | 该案例所有适用检查同时通过 |
| Grounded answer | 最终答案中的关键事实可以追溯到真实 Observation |

## 失败类型

| 标签 | 判定规则 | Main 主要标签 | Main 总出现数 |
|---|---|---:|---:|
| `wrong_tool` | 首工具或完整工具序列不符合预期 | 0 | 3 |
| `argument_extraction_error` | 工具参数缺失、类型或关键值错误 | 1 | 4 |
| `dead_loop` | 重复相同调用或超过最大步骤 | 0 | 0 |
| `tool_error_unhandled` | 工具报错后既未处理也未安全终止 | 0 | 0 |
| `premature_answer` | 未获得必要 Observation 就给出答案 | 0 | 0 |
| `final_answer_error` | 语义或可溯源性检查失败 | 17 | 22 |
| `hallucinated_observation` | 引用了工具从未返回的事实 | 0 | 0 |
| `format_parse_error` | 工具调用格式无法解析 | 4 | 4 |

老师点名的选错工具、参数提取错误和死循环均有确定性规则与统计。该开发集中检测到选错工具和参数问题，死循环为 0；“未观察到”不等于没有设置检测。

## 配对实验

| 指标 | Main | Ablation | 差值 |
|---|---:|---:|---:|
| 严格成功率 | 45.0% | 35.0% | -10.0pp |
| 首工具选择 | 92.5% | 95.0% | +2.5pp |
| 完整工具序列 | 92.5% | 95.0% | +2.5pp |
| 参数正确率 | 90.0% | 92.5% | +2.5pp |
| 格式合法 | 87.5% | 85.0% | -2.5pp |
| 语义正确率 | 50.0% | 47.5% | -2.5pp |
| 最终回答可溯源 | 45.0% | 35.0% | -10.0pp |
| 无循环 | 100.0% | 100.0% | 0pp |

ablation 稍微改善路由和参数，却削弱了最终答案与 Observation 的一致性。严格成功率由多项检查共同决定，因此不能只看首工具准确率。

## 文件说明

- `source/results/v2_dev_prompt_main.json`：main Prompt 的 40 条原始轨迹；
- `source/results/v2_dev_prompt_ablation.json`：ablation 的 40 条原始轨迹；
- `source/results/v2_prompt_selection.json`：Prompt 选择规则与结论；
- `source/results/v2_analysis/`：自动分类、逐题 CSV 与配对差异；
- `source/results/AGENT_ERROR_ANALYSIS.md`：老师要求的错误模式分析报告。

## 完成检查

- [x] 固定模型、数据、工具、生成参数和最大步骤；
- [x] 仅修改一条 Prompt 规则；
- [x] 统计选错工具、参数错误和死循环；
- [x] 保留主要标签与总出现数两种口径；
- [x] 保留改善与回归案例 ID；
- [x] 最终采用开发集严格成功率更高的 main Prompt。
