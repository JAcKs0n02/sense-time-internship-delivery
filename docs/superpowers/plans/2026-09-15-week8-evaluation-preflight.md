# Week8评估输入预检执行计划

**Goal:** 在首次新训练前固定benchmark数据和配置，检查与已验收SFT数据的交叉，并明确尚缺的裁判和GPU身份。

**Architecture:** 继续既有任务6接续计划；只读冻结阶段1产物，新增独立阶段2证据目录。所有下载绑定官方来源、精确revision与SHA-256；实际数据检查通过也不自动解除训练门禁。

**Tech Stack:** Python、官方benchmark数据、OpenCompass 0.5.3配置。

**Spec:** reports/week8/phase1_task6/next_stage_plan.json；docs/week8_data_protocol.md。

## 步骤

- [x] 获取C-Eval/CMMLU官方不可变版本的本地快照、保留来源及文件哈希。
- [x] 固定OpenCompass 0.5.3具体代码与数据配置，核对52/67科目、全部评估题和few-shot输入。
- [x] 使用已冻结的词面规则检查SFT train/validation各轮与benchmark问题，记录命中，不自动改动已冻结数据。
- [x] 汇总输入锁定状态、污染结论与缺项；收到用户明确的GPU实例与裁判信息后才能核验对应目标。

## 验收边界

不运行训练、裁判调用或付费实例启动。不能用历史分数替代新推理，不能将文件下载称为GPU评估通过。发现污染则保留命中明细并暂停训练放行，按原协议升版重做数据。用户要求分步执行，此轮聚焦评估前置条件，不串行执行完整SFT/DPO。

## 执行结果

数据/配置文件身份与交叉检查完成；实际OpenCompass配置接入、全题提示长度、裁判身份和GPU目标仍未验收，详见../../week8_phase2_evaluation_preflight.md。勾选表示完成本轮列出的静态输入预检，不代表运行环境或训练通过。
