# Week8：前序复核与正式评测配置

2026-09-15。本轮完成前序证据复核、正式评测配置生成、真实 OpenCompass CLI dry-run 和入口修正。**本地配置验收通过；尚未放行正式推理、裁判调用或训练。**

## 前序复核

- 第一阶段归档 `deliverables/week8/phase1/SHA256SUMS.txt` 重新逐文件核对，138项全部通过。
- 全部119科、12928道计分题重新走真实OpenCompass提示流程，逐题token与独立v2预期一致，五条示例保留，最长1361 token，截断差异0。收据保存在 `reports/week8/phase2_formal_config/prompt_recheck/verification.json`。
- 配置构建前重新核对290个benchmark CSV、输入清单、统一记录、原提示配置、专用模板及独立预期的SHA-256。
- 本轮新增回归测试后，完整测试集38项全部通过。此前35项通过的记录保留，没有用新增测试结果覆盖历史记录。

320机的基础CUDA检查与1580条数据加载属于前次目标机证据；本轮未重新开机，不能把历史检查写成本次远端复验。完整模型forward、修复后的目标Linux提示核验仍未完成。前几周既有质量缺口也没有因这次检查而消除。

## 本次发现并修复的问题

OpenCompass 0.5.3 `utils/run.py:get_config_from_arg` 在指定完整配置时优先返回配置，忽略`--hf-path`和其他模型快捷参数。原`step3_eval.py`同时传配置和模型快捷参数，未来解除运行锁后可能评测旧配置中的模型路径。之前运行锁一直关闭，此问题没有导致实际错误模型评测；但不能据此前测试通过断言入口已完整验证。

修正后，模型路径直接绑定到完整配置和运行锁；入口要求CLI传入的模型路径与锁一致。运行命令仅传配置文件、工作目录、单worker和详细计分输出参数，子进程显式设置仓库工作目录及PYTHONPATH，以便导入专用模板。运行前同时核验配置、题目记录、模板代码、290个CSV、tokenizer等依赖文件。任意已绑定文件发生变化即拒绝运行。

测试先验证旧代码未拒绝模型路径不一致，再修复；失败和通过日志分别见`tests_before.log`、`tests_after.log`。

## 正式配置及CLI核验

新增 `scripts/build_week8_formal_eval_config.py`，从已通过提示验证的完整119科配置生成正式权重加载配置，输出`opencompass_formal.py`及`runtime_lock_candidate.json`。候选锁始终`verified=false`，不会自动写入或解除全局运行锁。

| 设置 | 固定值 |
|---|---|
| 数据 | C-Eval 52科val；CMMLU 67科test |
| 示例 | 每科dev前5条，专用非递归模板 |
| 模型 | 明确传入绝对路径；本次候选使用此前核对过身份的320机原始Qwen2.5-7B-Instruct路径 |
| 加载 | tokenizer_only=false，BF16，local_files_only=true，trust_remote_code=false |
| tokenizer | 第一阶段核对过的原始tokenizer，文件哈希纳入依赖 |
| 生成 | do_sample=false，num_beams=1，max_out_len=32 |
| 输入 | max_seq_len=2048 |
| 资源 | 单GPU、batch size 1、CLI单worker |

真实OpenCompass CLI使用`--dry-run`成功退出，载入119科并完成任务划分；保存的有效配置保留了上述模型与数据设置。此模式没有启动模型推理，不能代替forward或实际输出质量验收。

候选中的数据/tokenizer路径绑定当前仓库位置，模型路径绑定320机；**不能直接把本地候选文件复制到远端执行**。需将证据与代码传入目标仓库、核对哈希，然后在目标目录重新生成配置及候选锁。更换为后续SFT/DPO模型也必须重新生成、绑定并验收，不靠`--hf-path`覆盖。

复现构建（在对应主机的仓库根目录运行，输出目录必须未存在）：

```bash
python scripts/build_week8_formal_eval_config.py \
  --model /root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct \
  --output-dir reports/week8/formal-config-new
```

## 下一步及未通过条件

1. 将修复代码和固定输入传到320机，逐文件校验，在目标目录生成配置并复验全部提示。
2. 验证目标模型身份、实际加载与最小forward；按既定关卡进行后续短步smoke，不提前开始完整训练。
3. 补齐custom20裁判准确模型名称、版本及base URL；不猜测裁判，不要求在聊天中提供密钥。
4. 将目标机收据和完整依赖身份纳入运行锁后再审核放行。当前`training_allowed=false`，全局`runtime_lock`仍未配置，正式入口继续拒绝执行。

主要证据目录：`reports/week8/phase2_formal_config/`。汇总收据为`verification.json`。
