# Week8 学生蒸馏训练实测记录

2026-09-18，AutoDL 320（be044ebe99-be706b14）单卡 RTX3090 上完成学生训练、LoRA 合并、逐张量检查与 CUDA 冷加载。训练技术验收通过；CEval 和同协议吞吐对比尚未执行，不能据此宣称蒸馏提升质量或已达到最终验收指标。

## 输入与训练设置

教师为已核验的 Week4 corrective DPO 7B 模型。学生为 Qwen/Qwen2.5-0.5B-Instruct，固定 revision `7ae557604adf67be50417f59c2c2f167def9a775`。本次采用教师答案监督的序列级蒸馏，不是 KL 软标签蒸馏。

200 条教师生成结果中，191 条生成完整，质量筛选后保留152条；正式学生 tokenizer 清洗后分为137条训练、15条验证，无提示交集。远端预检和训练入口各自生成的数据文件，均与上一阶段本地准备产物逐文件哈希一致。质量审核、筛选清单及目标答案哈希也已核对。

沿用批准配置：BF16、LoRA rank8/alpha16、all target、学习率5e-5、batch1、梯度累积8、cutoff1024、seed42。实际训练两轮，每轮18次更新，总36步，每轮末验证。内部 Trainer 循环上界不作为实际训练轮数；最终状态确认为 epoch2.0。

## 实际结果

| 项目 | 结果 |
|---|---|
| 优化器更新 | 36/36 |
| 实际轮数 | 2.0 |
| 第一条/最后一条训练日志 loss | 1.6044 / 0.6115 |
| 第1轮验证 loss（step18） | 1.7604635953903198 |
| 第2轮验证 loss（step36） | 1.753207802772522 |
| 更新的 LoRA 目标模块 | 168 |
| adapter 张量 / 合并后张量 | 336 / 290 |
| 权重合并检查 | TENSOR_MERGE_VERIFIED |
| CUDA 冷加载 | COLD_LOAD_PASS；BF16；有限 logits |
| 冷加载生成 | 2 token，“你好！”；仅加载冒烟检查，不计基准成绩 |

训练日志末尾 loss 不是整轮平均值，也不是独立测试成绩。15条验证集很小，以上结果只支持本次训练与导出流程有效，不能证明泛化改善。

会话起止为 UTC 19:40:25.951202 至19:42:10.508781，约105秒，涵盖预检、训练、导出和验证，不等同于GPU完整计费时间。会话结束后回传证据并通过控制台确认320“已关机”。启动时显示单价1.48元/小时，实际费用以平台账单为准。会话设置了超时及平台定时关机保护，没有自动重试。

## 模型与证据位置

合并模型保存在320数据盘：

```text
/root/autodl-tmp/week8-distillation-student-20260918/run-01/student/final_distilled
```

`model.safetensors` 大小988097824字节，SHA256：

```text
61a741a2b366253ace843f74add4c80468df729bfb7ea91bb56b32770a7c5386
```

仓库 `reports/week8/phase3_distillation_student_gpu/` 保存执行计划、上传清单、会话信息、52份小文件回传证据、离线复核脚本及结果。大权重和完整 tokenizer 仍在远端，未宣称已在本地备份。

- `retrieved/student/input_binding.json`：输入、模型身份、配置与数据绑定。
- `retrieved/student/training/attempt-01/adapter/trainer_state.json`：完整训练状态；train.log 保存实际训练日志。
- `retrieved/student/student_tensor_verification.json`：远端张量计算结果及完整模型文件清单。
- `retrieved/student/student_cold_load.json`：CUDA 冷加载回执。
- `received_bundle.json`：从远端回传并按终端显示SHA固定的证据包。
- `checks.json`、`verification.json`：本地复核结果和本轮交付文件哈希。

离线复核需带 PyYAML 的 Python，无需 GPU。此次使用已有环境运行：

```bash
/tmp/week8-student-tokenizer-20260918/bin/python reports/week8/phase3_distillation_student_gpu/verify_receipts.py
```

本地复核重新检查52份文件哈希、两轮训练状态与有限日志、两轮验证时点、数据输出与输入审核绑定、合并与冷加载回执关联。张量计算及真实 CUDA 加载在320执行，本地复核不冒充重复加载大模型。

## 下一项

固定相同题集、评分规则与生成参数，对原始0.5B和蒸馏0.5B执行 CEval 前后对比；按相同硬件、精度、提示、生成长度及计时范围进行推理吞吐比较。报告需区分学生蒸馏前后变化与7B教师/0.5B学生的体量差异。当前未得到这些比较结果，保持 `STUDENT_TRAINING_VERIFIED_COMPARISON_PENDING`。


## 2026-09-18 比较完成续记

前后完整CEval与同协议测速现已完成，本地243份证据重验通过。CEval53.7147%→53.1204%，速度基本持平；上述“待比较”为训练阶段历史状态。详见[比较报告](week8_distillation_comparison.md)。320已关机。
