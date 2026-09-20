# Week8 三模型正式评测准备

本阶段已完成本地候选配置、目标机CPU身份审核、tokenizer差异核对及三模型独立评测锁放行，尚无正式分数。准备过程没有启动GPU或调用裁判API。下方“审核发现”保留候选阶段发现问题的过程，最新结论见文末更新。

## 已复核的输入

再次运行正式训练证据审核：90份回收文件、40项冻结输入、10个阶段进程均通过。SFT完成1775步，DPO完成40步，两次合并与独立加载已通过。大权重仍在320机，本轮本地审核不等于重新读取目标权重。

复核现有评测锁的304项依赖、冻结题目、裁判profile及原放行记录。为三个模型分别生成候选配置、模型身份清单和未放行的runtime lock：

| 模型 | 目标位置 | 身份文件数 |
| --- | --- | --- |
| original_base | `/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct` | 15 |
| final_sft | `/root/autodl-tmp/week8-full-training-runtime-20260916/logs/full-run-01/models/final_sft` | 14 |
| final_dpo | `/root/autodl-tmp/week8-full-training-runtime-20260916/logs/full-run-01/models/final_dpo` | 14 |

候选配置除模型路径外与原已核验配置一致。每个新锁把模型目录下各文件的绝对路径和SHA256加入现有入口实际检查的dependencies，避免只有模型路径、没有权重身份。所有候选仍为`verified=false`，正式入口应拒绝执行。

## 正式比较范围

每个模型完成CEval验证集52学科1346题、CMMLU测试集67学科11582题，以及同一份custom20。三模型合计38784条客观题推理、60条开放回答。

客观题使用冻结的OpenCompass配置，BF16、batch1、2048上下文、最多32个新token、贪心解码、seed42。custom20沿用现有入口的最多512个新token及贪心解码。客观题按标准答案核算准确率；custom20由已校准的DeepSeek裁判评分，明确标注为AI评分。裁判单题最多两次尝试，不因分数低重评；原入口每模型最多40次尝试和余额保护保持不变。

原基座作为同协议对照重新评测；不混入旧周次、不同题目或不同提示词的历史分数。选择本次训练最终模型，不用评测集挑checkpoint。SFT验证loss从约0.929上升至1.566需如实保留，不能以训练成功推断效果提升。

## 审核发现与下一步

1. 新合并模型的`tokenizer.json`哈希与冻结评测tokenizer相同；回收checkpoint的tokenizer配置与冻结版本语义一致。
2. **合并模型的`tokenizer_config.json`哈希不同于checkpoint**：合并文件为`056f2c…db491`，checkpoint为`8a5e6a…b98e2`。本地没有合并后的原文件，尚不能判断是元数据、格式还是行为差异。不得把checkpoint的一致性当作合并模型聊天模板一致的证据。
3. 在320机上读取三个模型实际tokenizer与generation配置，逐字段解释差异；用相同custom20输入对照chat template、token ID和停止规则。若存在实际行为差异，先固定公平评测协议再更新配置并复核。
4. 目标机重新核验43个身份文件、依赖和配置，保存新回执；通过后分别放行。现有入口只消费一个runtime lock，应串行安装各自的锁、使用独立输出目录，不能并行覆盖同一个预检文件。
5. GPU启动前确定执行时间窗口和关机保护；先验证实际评测入口与吞吐，再全量执行。失败保留证据，不盲目从头重跑，不删历史模型；任务结束回收结果、复核逐题覆盖和分数，关机。

## 可复现检查

```bash
/tmp/week8-oc-fix-20260915/bin/python reports/week8/phase2_full_training_target/review_evidence.py
/tmp/week8-oc-fix-20260915/bin/python reports/week8/phase2_three_model_eval/prepare.py
```

产物在`reports/week8/phase2_three_model_eval/`。`verification.json`的PASS仅表示本地候选准备通过；目标tokenizer语义检查、权重重验和正式推理均未完成。

## 2026-09-16 目标机CPU审核更新

在320机无卡模式执行，平台显示0.5核CPU、2GB内存、¥0.10/小时。已设置22:30 UTC自动关机保护；本阶段不加载模型权重进行推理。

`target_audit.json`及本地`target_review.json`记录以下结果：

- 三个模型共43个身份文件逐文件核对大小及SHA256，并核对目录文件集合；全部一致。
- 三份评测配置除模型路径外一致；数据、代码、裁判profile及冻结依赖通过目标机审核。
- 原基座、SFT、DPO分别对20道custom20执行实际chat template及tokenize，共60次输入token ID对照，全部与冻结版本一致；词表与聊天模板也一致。
- 合并模型的tokenizer配置仅`padding_side`由`right`变为`left`。当前入口逐题生成且不padding，此差异不改变本次输入。原基座配置仅缺省`extra_special_tokens`和`padding_side`，实际输入也通过相同比较。
- 三者实际generation配置除版本元数据外一致，入口覆盖后均为贪心、num_beams=1、max_new_tokens=512、repetition_penalty=1.05、eos=[151645,151643]、pad=151643。不能因为文件哈希不同就推断生成行为不同。
- 目标机空闲数据盘约20.30GiB；权重留在原路径，未复制大权重、未删除历史模型。

本轮不修改tokenizer、模型或冻结推理参数。上述结论只适用于当前逐题协议；后续如引入批量padding，需要重新核对。

复核命令：`python3 reports/week8/phase2_three_model_eval/review_target.py`。原始回执SHA256为`7b897741b926728fa3989213d5d5a80bf823927ec957aa5348cdbcb65f7e1525`，已下载并验证传输一致。

### 独立评测锁放行结果

目标机`release_target.py`将CPU原始回执及本地复核回执纳入锁的依赖，分别调用原有`step3_eval.require_evaluation_lock`，原基座322项依赖、SFT321项、DPO321项全部通过。检查包含重新流式读取权重哈希，没有调用`fresh()`，没有裁判API请求。原来的全局预检文件已逐字节恢复，当前没有新评测在运行。

三份正式锁已安装于目标机：`/root/autodl-tmp/week8-judge-runtime-20260916/reports/week8/phase2_three_model_eval/{original_base,final_sft,final_dpo}/runtime_lock.json`。对应配置、身份清单及回执也已部署。后续必须选择正确模型对应的锁安装到全局预检，再执行原评测入口；不可同时覆盖全局预检。每次运行入口仍会重新检查全部依赖。

本地三份`runtime_lock.json`与`release_receipt.json`已从目标机下载并核对SHA256。归档`released_locks.tar.gz`的SHA256为`e6fceacae9cc98d43455c0fc938e8ff6d93dbe11681dea1f72e6cec7418ea28b`。

下一步为GPU执行准备与真实推理：先确认当前资源、实际显存和吞吐，据此确定完整评测窗口及平台关机保护，再执行全量题目。小批推理只用于运行核验，不替代全量分数，也不用于挑选模型或调参。无卡检查完成后关闭实例，避免等待时继续计费。

会话收尾已完成：控制台确认320机“已关机”，本轮22:30 UTC定时已取消，页面恢复“设置定时关机”。记录见`session_close.json`；没有启动GPU评测，没有调用裁判API。
