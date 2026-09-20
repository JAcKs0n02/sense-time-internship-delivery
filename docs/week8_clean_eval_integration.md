# Week8 干净环境评估入口与 Gemini 接入

日期：2026-09-19。范围：完成上一轮准备方案的本地代码接入、模拟验证和独立代码审查。真实GPU评估和新裁判请求尚未执行；目标环境运行锁未放行。

## 已实现的行为

新增`scripts/week8_clean_eval.py`，并通过`run_pipeline.sh --clean-eval`进入。旧`step3_eval.py`、旧Gemini评分器及历史评分计划保持原字节，避免破坏已有实验的哈希证明。新入口复用其中的基准逐题核验、Google原生API请求执行和已校准评分协议。

运行分为GPU生成与本机评分两段。前段自动执行正式数据准备、已有final_dpo的119科基准推理、custom20新答案生成；后段在产物回收且320停机后生成新Gemini计划、评分并输出三行汇总CSV。这满足分段执行的设计，不把历史成绩复算当作新推理。

| 阶段 | 具体行为和检查 |
|---|---|
| candidate | 根据已审核配置生成候选，绑定14个模型文件、数据、代码和配置哈希；固定final_dpo；始终输出CANDIDATE_NOT_RELEASED |
| audit | 核对候选依赖、配置/模型/题型、完整模型文件及哈希；不启动推理、不自动放行；不替代目标GPU forward/prompt实测 |
| generate | 要求目标320及准确仓库根目录、已通过目标审计、有效截止时间与平台关机窗口；跳过训练，运行数据准备、基准和20题生成 |
| prepare-score | 核对完整回收清单、原始基准结果、15个数据产物、新答案身份、已校准Gemini协议及320停机回执，生成独立新评分计划 |
| score | 本机执行20次Google原生评分；通过子进程监督限制总墙钟时间7200秒；记录请求、响应、token用量并独立重计分 |
| verify-score | 零调用复核已完成的新评分，重建/核对pipeline_summary.csv，包括空白列和完整表头 |

主控`--clean-eval generate`内部自动执行数据准备并跳过训练，不能再与`--quick`、旧`--score-only`、`--skip-train`等其他模式混用。原有模式继续独立可用。旧`--skip-train`的fresh分支仍属历史DeepSeek入口，新的Gemini流程使用上述显式入口，不把替换模型名视为兼容切换。

## 本轮生成的候选

[本机候选](../reports/week8/clean_eval_integration_20260919/generation_candidate.json)绑定本地final_dpo备份，仅用于核验配置与依赖。它没有有效目标审计、执行截止时间或平台关机窗口，不能启动GPU。目标根目录和模型位置确认后必须在新独立候选中重建配置与计划。

以下配置准备命令不加载模型、不调用API，每次使用新输出路径：

```bash
python scripts/build_week8_formal_eval_config.py \
  --model "$WEEK8_MODEL_ABSOLUTE_PATH" \
  --output-dir reports/week8/clean-eval-target-config-001
python scripts/week8_clean_eval.py candidate \
  --binding reports/week8/clean-eval-target-config-001/runtime_lock_candidate.json \
  --output reports/week8/clean-eval-generation-candidate-001.json
```

后续已放行的目标运行使用下面的入口；这些是接口说明，本轮没有生成可启动的`WEEK8_RELEASE`：

```bash
bash run_pipeline.sh --clean-eval generate \
  --eval-plan "$WEEK8_RELEASE" --eval-plan-sha256 "$WEEK8_RELEASE_SHA256" \
  --run-dir logs/clean-eval-generation-001
```

完成回收与停机后，本机准备新评分计划：

```bash
python scripts/week8_clean_eval.py prepare-score \
  --source "$WEEK8_RETRIEVED_GENERATION" \
  --receipt-sha256 "$WEEK8_GENERATION_RECEIPT_SHA256" \
  --gpu-release "$WEEK8_GPU_STOP_RECEIPT" \
  --output reports/week8/clean-eval-gemini-plan-001.json
bash run_pipeline.sh --clean-eval score \
  --eval-plan "$WEEK8_SCORE_PLAN" --eval-plan-sha256 "$WEEK8_SCORE_PLAN_SHA256" \
  --run-dir logs/clean-eval-scores/run-001
```

最后把`score`改为`verify-score`可进行零调用复核。产物路径必须与第一次一致；更换目录不能绕过同一答案/协议的付费claim。不存在评分结果时，verify-score直接报错，不隐式调用API。

## 运行锁与证据要求

目标审计须包含`CLEAN_EVAL_TARGET_PASS`、目标host、候选规范化JSON SHA256、pip_check/cuda_bf16/model_forward/prompt_check/shutdown_confirmed检查结果，以及完整environment_freeze文件路径和哈希。审计还须绑定准确deadline_utc及platform_shutdown_utc。候选规范化计算使用`scoring.canonical`，不等同于磁盘JSON文件SHA；主控的`--eval-plan-sha256`则始终是磁盘文件SHA。

新环境采用Python3.10，核心运行版本固定torch2.5.1+cu121、Transformers4.50.0、OpenCompass0.5.3、MMEngine0.10.7。运行时重新pip check、读取实际包版本、生成freeze并与已审核版本逐字节比较；pip check成功本身不足以通过。正式配置和所有依赖在加载Python配置前先核对哈希。

GPU生成持有实例级文件锁及与旧会话兼容的仓库锁。基准最长14400秒，总生成窗口最多21600秒；平台关机最多在程序截止后600秒作为故障兜底。因此极端故障情况下预算还需预留最多10分钟实例计费，而非承诺6小时绝对平台扣费上限。成功提前完成则回收产物后立即停止实例。

生成回执列出所有数据、推理、日志和环境文件；回收后拒绝文件遗漏、额外文件、符号链接、哈希变化。CEval/CMMLU继续从逐题原始结果重算，不接受手写均分。新20题按原题序、20个唯一ID绑定答案；空答案、非EOS结束或超过512个新token均拒收。

停机回执须有`instance: "320"`、`state: "stopped"`、本次generation_receipt_sha256、带时区的observed_utc，以及原始观察证据的相对路径/哈希。它应在实际看到停机并回收本次结果后建立，不由本地模拟生成。程序核验其绑定及完整性；平台真实状态仍依赖执行时取得的原始观察证据。

API请求沿用既有校准协议和估算规则；旧6美元准入预算不是当前供应商报价或硬扣费上限。实际放行前仍需确认所用预览版本可用及价格适用。未知请求保留started标记，失败不自动重试；完整响应可复算。GPU端禁止执行付费评分。

## 本地验证与代码审查

测试使用模拟裁判响应和临时产物，不产生API消费，也不加载GPU。完整流程测试借用历史基准文件检查解析及来源核验，明确标记为合成测试，不是新GPU评估证据。新增测试覆盖新答案请求绑定、缺题/截断拒收、源文件变化、运行锁拒绝、模型与配置依赖、重复收费/未知请求保护、CSV核验及API子进程监督。

独立代码审查发现并修复：旧进程清理函数可能遗留孙进程、缺少跨计划运行锁、评分未绑定停机回执、环境仅记录未比对、API缺少进程级墙钟限制、CSV空白列未校验。新入口独立处理这些问题，未修改冻结旧脚本。父进程退出而孙进程忽略SIGTERM的情况已用真实本地进程复现，再验证整个进程组被终止。

最终测试数量、命令、退出码、候选状态和文件哈希见[本轮回执](../reports/week8/clean_eval_integration_20260919/verification.json)。本轮未重导既有发布候选、未启动320、未训练或收费评分、未提交推送。

## 下一步

将新入口、测试和文档纳入新的独立发布候选，复验安装/数据入口与必需文件白名单；随后在320准备目标配置、环境审计、模型forward与prompt核对、准确运行时间窗口和平台关机证据，生成新放行计划。只有这些通过后才开始正式GPU评估。真实评估结束前，老师的干净环境真实评估项仍保持待验收。
