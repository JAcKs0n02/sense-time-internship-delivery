# Day 19 Submission：DPO 配置、启动日志与 Rewards 纠偏

本目录对应老师 Day 19 的 DPO 配置、训练启动日志和 Rewards 监控要求。最终选用的纠偏 run 已完成 `40/40` optimizer steps：Chosen 的末 20-step 均值高于首窗口，Rejected 的末窗口低于首窗口，严格满足老师要求的方向。

## 最终采用文件

- [DPO_Config.yaml](DPO_Config.yaml)：最终纠偏 run 的实际配置，`pref_beta=0.1`。
- [Launch_Log.txt](Launch_Log.txt)：最终纠偏 run 的原始启动、训练、评估和保存日志。
- [DPO_Training_Summary.json](DPO_Training_Summary.json)：状态、两次验证、Rewards 窗口、数据哈希和 adapter lineage。
- [DPO_Metrics_By_Step.csv](DPO_Metrics_By_Step.csv)：40 个连续 optimizer step 的原始指标和 20-step 移动平均。
- [Adapter_Manifest.json](Adapter_Manifest.json)：最终 adapter 路径、大小和 SHA-256；本提交不含权重。
- [Corrective_Development_Review.json](Corrective_Development_Review.json)：只使用隔离开发集完成的安全、正常任务和业务门禁。
- [Reference_Model_Decision.json](Reference_Model_Decision.json)：显式 reference 到 LoRA adapter-disabled reference 的显存决策。
- [DPO_Explicit_Reference_Config.yaml](DPO_Explicit_Reference_Config.yaml)：最初按老师字面要求设置 `ref_model` 的配置。

## 最终训练参数

| 参数 | 值 |
|---|---|
| framework | LLaMA-Factory 0.9.3 |
| stage / method | DPO / QLoRA |
| policy / logical reference | Week 3 最优 merged SFT |
| corrective train / validation | 783 / 87 |
| beta / loss | 0.1 / sigmoid |
| LoRA rank / alpha / target | 8 / 16 / all |
| quantization | bitsandbytes 4-bit NF4，BF16 compute |
| cutoff length | 2048 |
| optimizer steps | 40/40 |
| learning rate | 2e-6，constant with 29-step warmup |
| micro batch / accumulation | 1 / 8 |
| eval / save interval | 20 / 40 steps |
| seed | 42 |

逻辑参考模型仍是冻结的 Week 3 SFT。显式加载第二份 7B reference 在 24GiB GPU 的真实长样本上 OOM，因此成功 run 使用 LLaMA-Factory LoRA DPO 的 adapter-disabled reference：计算 reference log-probability 时临时禁用正在训练的 LoRA adapter，不会把更新后的策略当作参考。

## Rewards 严格结果

| 指标 | 首 20-step 均值 | 末 20-step 均值 | 变化 | 验收 |
|---|---:|---:|---:|---|
| Chosen reward | 0.002147 | 0.034006 | +0.031859 | 上升，PASS |
| Rejected reward | 0.008553 | 0.004417 | -0.004136 | 下降，PASS |
| Reward margin | -0.006405 | 0.029590 | +0.035995 | 扩大，PASS |

Validation margin 从 step 20 的 `0.000667` 上升到 step 40 的 `0.050445`，reward accuracy 从 `0.517241` 上升到 `0.701149`。状态为 `completed`、return code 为 `0`，指标均为有限值。

最终 adapter：

- 路径：`/root/autodl-tmp/qwen25-week4/day20-remediation/runs/reward-corrective-40step/attempt_001/trainer_output`
- 大小：80,792,096 bytes
- SHA-256：`d7a932ee4f28c8950db289126381f5d4dd30a037b238852e87ad8e5a241f9e52`

## 纠偏选择与隔离

老师的 10 道安全题和 5 道业务题没有用于选择该 candidate。隔离开发集结果为：高危拒绝 `18/20 = 90%`、正常任务帮助 `10/10 = 100%`、业务均分 `3.555`，相对 SFT 的 `3.580` 下降 `0.025`，没有超过预设 `0.10` 容忍线。满足 Rewards、开发集安全、正常帮助和业务保持门槛后，candidate 才进入 Day 20 正式合并和老师题评测。

## 初始 240-step run 的纠偏说明

初始 run 完成 240/240 steps 且安全题为 10/10，但 Rejected reward 的末 20-step 均值高于首窗口，因此没有继续作为最终模型。纠偏不是更换 GPU 后的数据同步修复，而是基于严格验收门禁重新选择训练阶段。教师目录只保留最终 40-step 模型的配置、日志和指标；初始 run 的原始历史证据留在工程归档，不混入最终教师材料。
