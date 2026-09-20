# Week8：320机三步训练执行与审核记录

## 结论与边界

2026-09-16 在指定320机 `be044ebe99-be706b14` 完成一次 SFT 3步 → 独立模型合并 → DPO 3步。目标入口给出 `SMOKE_PASS_REVIEW_REQUIRED`，回收证据的本地复核给出 `GPU_SMOKE_PASS_LOCAL_REVIEWED`。没有运行完整训练、正式评测或裁判API；`full_training_allowed=false`、主预检中的 `training_allowed=false` 均保留。

这证明指定环境下前向、反向、参数更新、checkpoint保存、SFT合并以及基于合并模型的DPO链路可运行。三步没有覆盖完整数据，也没有验证泛化能力，不能将本次adapter或合并模型当作正式模型提交。

## 逐项验收

| 环节 | 本次真实证据 | 审核结果 |
| --- | --- | --- |
| 实例与开机保护 | 仅320机，价格¥1.48/小时；约13:25 UTC启动，关机保护设为页面时间15:20（约14:20 UTC）；另有2400秒进程上限 | 通过 |
| 部署 | 新目录 `/root/autodl-tmp/week8-training-smoke-runtime-20260916`；包SHA256与本地一致，31份冻结输入逐项一致，输出使用全新 `run-01` | 通过 |
| 环境与基座 | RTX3090 24GB、CUDA BF16可用；60GiB cgroup内存限额；部署前约64.5GiB可用磁盘；原始模型15文件哈希一致 | 通过 |
| 并行任务排查 | 开始前无GPU进程，无既有smoke目录；预检快照中的Python GPU进程是本次CUDA检查自身；结束后GPU进程列表为空 | 通过 |
| SFT数据与配置 | 实际日志1422训练样本；冻结158验证样本保留且关闭周期评测；长度2048、4bit NF4、rank8/alpha16、batch1、累积4，恰好3步 | 通过 |
| SFT更新 | 三步loss为1.1983、1.9822、2.5578；梯度有限；392个adapter张量有限，196个LoRA B张量非零；392份优化器参数状态均为step3 | 通过 |
| SFT合并 | CPU导出4个分片；339个权重张量与索引键集合完全匹配、数值有限；分片哈希已记录 | 通过 |
| DPO数据与配置 | 实际日志783训练样本；以本次 `models/smoke_sft` 为基座；长度2048、4bit NF4、rank8/alpha16、batch1、累积8，保留29步预热，恰好3步 | 通过 |
| DPO更新 | 三步loss为0.6931、0.6931、0.6891；所有记录数值有限；392个adapter张量有限，196个LoRA B张量非零；392份优化器状态均为step3 | 通过 |
| 本地复核 | 回收30份原始日志、配置和验收文件及清单；压缩包与文件逐项SHA256相符；实际YAML与冻结计划一致；两个CLI返回码均0；checkpoint与最终训练步骤一致 | 通过 |
| 结束与资源回收 | 约13:33 UTC关机，页面确认“已关机”；清除本轮关机定时器；`week8-320`监控暂停 | 通过 |

Trainer报告的SFT训练耗时6.3625秒，DPO训练耗时32.3996秒，均不包含完整的部署、文件校验、模型加载和导出时间。不能以此估算正式全流程总耗时或费用。实际账单本轮未另行核对。

## 环境与注意事项

Python 3.10.20，torch 2.5.1+cu121，transformers 4.50.0，LLaMA-Factory 0.9.3，PEFT 0.15.1，TRL 0.9.6，accelerate 1.2.1，datasets 3.2.0，bitsandbytes 0.43.3。

- 预检时发现继承的 `OMP_NUM_THREADS` 无效。本轮启动前将 `OMP_NUM_THREADS` 和 `MKL_NUM_THREADS` 显式设为14，记录在 `launch_environment.json`；没有改变数据或训练超参数。
- 三段日志均保留SDPA sliding-window提示，日志中实际模型配置 `use_sliding_window=false`。本次短程验收不因该提示失败，但不把数值有限等同于所有实现语义或长序列行为已验证。
- SFT最后记录的调度器学习率为0，DPO保留29步预热，符合冻结短程配置。三步验收关注是否有有效梯度和非零更新，不能据不同batch的loss变化判断训练效果。
- 本地交叉核对最初要求最终 `trainer_state` 与checkpoint逐字相等，发现最终状态额外附加训练汇总行。审核脚本改为核对全部非日志字段、完整三步日志及仅追加的step3汇总行；没有修改原始证据。

## 证据位置与复核方式

本地证据目录：`reports/week8/phase2_training_smoke/target-20260916/`。其中包含原始SFT/DPO训练日志、两份实际训练YAML、导出日志/配置、checkpoint/最终Trainer状态、adapter配置、分阶段验收和最终状态。`evidence_manifest.json`逐文件记录大小及哈希。

- 部署包SHA256：`3e75d310e532bbe6b7bf59202d14589f1ee8c53d8939b830fe981e736bbae081`
- 冻结锁SHA256：`7c9bc62c7870b5a652eb098a6451319728271902ab280d46eef078769d2be882`
- 回收证据包SHA256：`6fdeee44fde063e0e62f71215c137fd5971b3954bdecc046ac9d464ffb99a941`
- 本地审核：`reports/week8/phase2_training_smoke/target_review.json`
- 关机记录：`reports/week8/phase2_training_smoke/target_session_close.json`

原始adapter、optimizer和约15GB合并权重保存在目标机独立目录，没有下载这些大体积二进制文件。其张量检查在目标机完成，结果和哈希已回收；本地检查不冒充一次对全部权重的独立重算。

可重跑本地证据审核（不启动训练）：

```bash
/tmp/week8-oc-fix-20260915/bin/python reports/week8/phase2_training_smoke/review_target_evidence.py
```

原 `verification.json` 保留为61项本地测试及部署前状态的历史证据；最新状态由主预检的 `training_smoke.target_review` 和本报告承接。

## 下一步

准备正式训练放行方案：依据本轮真实配置和资源占用确认完整SFT与DPO的步数、保存策略、预算及关机保护，为正式模型建立独立输出和身份记录，再按已冻结的评测协议执行后续评测。短程通过不会自动开启旧完整训练入口。原始数据、锁文件和本轮产物继续保留。
