# Week 5 Engineering Archive

Week 5 从纯文本训练转向视觉语言模型（VLM）实践：使用 Qwen2-VL 完成图文推理、跨模态注意力可视化、视觉幻觉测试和冻结视觉塔的 LoRA 微调，并已形成第 5 周报告与模型归档。总验收为 `3 PASS / 1 FAIL`，失败项是微调后明显提升。

本目录保存 Week 5 的可执行工程说明与实验归档。Day 22–Day 25 已完成并通过验证；Day 26 的数据、训练、冻结审计和盲评流程已完成，但预注册效果总门槛未通过；Day 27 已完成周报、验收矩阵和最终模型 lineage，Week 5 总验收保持 `3 PASS / 1 FAIL`。

## 执行入口

- [第 5 周完整执行计划](../../docs/week5_execution_plan.md)
- [老师新版需求 PDF](../../实习需求.pdf)

## 当前状态

| Stage | Status | Planned engineering deliverable | Teacher requirement |
|---|---|---|---|
| Day 22 | 已完成（PASS） | [固定模型 revision、环境、离线 smoke 与五张图片](day22/README.md) | 模型下载确认、图片素材包 |
| Day 23 | 已完成（PASS） | [25/25 原始回答、逐条复核与能力边界](day23/README.md) | 25 条图文推理结果记录 |
| Day 24 | 已完成（PASS） | [第 20 层真实权重、4 案例原始数组与可复现绘图](day24/README.md) | 4 张注意力热力图（要求至少 3 张） |
| Day 25 | 已完成（PASS） | [10/10 原始回答、严格二元评分与 70% 幻觉率](day25/README.md) | 幻觉检测报告 |
| Day 26 | 实验完成；效果门槛 FAIL | [200 条训练、20+20 隔离评测、冻结视觉塔 LoRA 与两轮盲评](day26/README.md) | 微调后的 VLM、微调日志；LoRA 12/20 胜、幻觉率 55%，均分仅 +0.0125 |
| Day 27 | 报告完成；总验收 FAIL | [v8 双候选终止、全周报告与模型 lineage](day27/README.md) | 周报完整；v8-E/F 均未通过开发门禁 |

## 固定模型与资源口径

| 项目 | 冻结口径 |
|---|---|
| GPU 路线 | 已实际验证 RTX 3090 24GB，满足 7B 路线 |
| 正式模型 | `Qwen/Qwen2-VL-7B-Instruct` |
| 模型版本 | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` |
| 测试图片 | 表格截图、自然风景、Logo、手写公式、UI 界面，各 1 张 |
| Day 23 规模 | 5 张 × 5 类问题 = 25 条 |
| Day 24 规模 | 至少 3 张跨模态注意力热力图 |
| Day 25 规模 | 固定 10 题，分母始终为 10 |
| Day 26 数据 | 200 条训练 + 20 条开发 + 20 条最终评测，图片 SHA 零重叠 |
| 效果门槛 | 均分增益 ≥0.50/5、LoRA 胜出 ≥12/20、幻觉率不高于 Base，三项同时满足 |

## 跨日数据流

```text
Day 22：模型 revision、环境和 5 张冻结图片
  ├─> Day 23：25 条 Base 图文推理
  ├─> Day 24：3+ 注意力热力图
  └─> Day 25：10 题 Base 幻觉检测

独立 200 条训练数据 + 独立 20 条评测
  └─> Day 26：冻结视觉塔的 LoRA 训练与 Base/LoRA 对比

Day 22–Day 26 已验证证据
  └─> Day 27：周报、验收矩阵、最终模型归档与教师投影
```

后续日期只消费上游已经冻结的 revision、配置和 SHA。任何上游图片、模型或生成参数变化都必须产生新版本标识，并重新运行所有受影响的下游实验。

## 目录边界

- `deliverables/week5/`：完整工程说明，以及后续增加的脚本、配置、小型数据、原始结果、测试和审计证据。
- `Submission/Week5/`：只收录已经实际执行并有可审计证据的教师交付；当前已建立 Day 22–Day 26 教师投影，Day 26 明确保留效果 FAIL，不美化为 PASS。
- `/root/autodl-tmp/`：模型、checkpoint、缓存和其他大文件的 AutoDL 持久化位置；实际路径在执行时写入非敏感 manifest。
- Git 不保存模型大权重、checkpoint、optimizer state、缓存、账号、密码、SSH 连接信息、真实个人数据或私有盲评映射。
- AutoDL 每次使用后都必须关机，但平台关机证据不属于老师交付。

## 更新规则

1. 每天开始前读取 [完整执行计划](../../docs/week5_execution_plan.md) 和当天 README；
2. 输入、Prompt、ground truth、评分规则和门槛必须在正式运行前冻结；
3. 只有脚本、日志、原始结果、数量、哈希和完成门禁全部通过后，才能修改当天状态；
4. 失败 run 与低质量回答保留在工程归档中，不通过删题、换答案或改分母美化结果；
5. `Submission/Week5/` 从验证通过的工程归档增量投影，不直接在教师目录里开发；
6. Day 27 只汇总可追溯事实，不把预期结果、train loss 或可加载性替代老师的质量验收。
