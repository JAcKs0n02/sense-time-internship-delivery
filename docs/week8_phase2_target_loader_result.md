# Week8：320机真实数据加载复验与评测阻断

2026-09-15。结论：**TARGET_LOADER_PASS_OC_PROMPT_BLOCKED**。

## 会话与逐步验收

监控于12:42 UTC发现320机GPU充足，核对价格为¥1.48/小时后启动。开机记录约12:43:54 UTC；设置服务器时间14:35（13:35 UTC）自动关机，提前于约12:57 UTC关机，随后确认“已关机”。本次少于1小时，未查询实际账单。只操作be044ebe99-be706b14；结束后暂停week8-320监控。

第一步，传输校验通过：独立目录`/root/autodl-tmp/week8-recheck-20260915`，318个输入文件全部通过SHA-256检查，上传包哈希为`30c47d20b816d1e77f99185c3403813ee7425ef4d1de8841eaa2ba8d9ee1141b`。开始时GPU无计算进程，没有发现训练或评测进程。

第二步，实际LLaMA-Factory加载通过：使用目标Linux环境原有`llm_exp`，运行与本地相同的`check_week8_loader.py`，退出码0。此检查在目标机器上明确禁用CUDA，只审核数据预处理，不执行模型forward或训练。不能将它写成CUDA模型训练验证。

| 检查 | 训练集 | 验证集 |
|---|---:|---:|
| 记录数 | 1422 | 158 |
| 监督token | 85596 | 8702 |
| assistant轮数 | 1441 | 158 |
| 多轮记录 | 13 | 0 |
| 最长token | 2048 | 1949 |
| 逐数组差异 | 0 | 0 |

1580条ShareGPT与Alpaca加载结果一致，合成system多轮检查通过。训练与验证token/labels输出文件SHA-256分别为`346c9067ccfeb6acb05b1ca8300876e4ad7d7056f3694c5fd20f4ad769c54055`、`583d0a5681d139c1752dfa0c718b6a32cc8dd7ff915c61e4a1832c7031f23e3c`，均与本地attempt-02重新计算值一致。PEFT0.15.1与Accelerate1.2.1的环境差异保留，没有升级原环境。

第三步，真实OpenCompass提示复验**未通过**：实际调用Config.fromfile、注册Dataset加载、FixKRetriever、PromptTemplate、GenInferencer提示生成和HuggingFacewithChatTemplate的tokenizer_only模式。没有加载模型权重或生成回答。

- attempt-01/02在首科发现独立重建缺少换行。默认SYSTEM也映射到HUMAN，空SYSTEM与HUMAN合并时产生额外换行；原先独立重建未计入。修正版保存在`scripts/audit_week8_benchmark_prompts_v2.py`和`reports/week8/phase2_eval_hardening_v2`，原始证据不覆盖。修正版仍是独立检查，不代表运行验收。
- attempt-03/04在上传未完成时读取预期文件，分别出现不存在、JSON不完整错误；保留日志。后续确认远端文件完整SHA-256与本地一致后才再次运行。后续传输必须以完整哈希为门槛，不能以上传控件返回为完成。
- attempt-05通过前面若干科目后，在C-Eval高等数学第14题（row_index=13）被token哈希差异拦截。官方`safe_format`逐字段执行replace，会再次替换已插入题目中的`{C}`，将积分下标替换为选项C文本。该问题不能通过修改预期来接受。
- 在本地用官方函数源码复现，扫描fewshot与计分源记录，发现11条计分题同类冲突（C-Eval3条、CMMLU8条）。见`reports/week8/phase2_gpu_monitor/recursive_placeholder_hits.json`。这是源文本替换检查，不是全题实际框架验收通过。

## 证据与边界

本地`reports/week8/phase2_gpu_monitor/observed_receipt.json`保存通过已登录Jupyter编辑器读取的收据字段、依赖、计数、日志哈希与异常。它是明确标注的摘录，不冒称完整原始文件的字节副本。完整日志、原始收据和token输出已打包保存在远端`/root/autodl-tmp/week8-recheck-20260915/evidence.tar.gz`，**完整压缩包尚未下载至本地**。本地保留了运行脚本、传输清单、输入包和独立预期。

下一步优先在本地实现并测试“只替换模板占位符、不递归替换插入内容”的适配，覆盖上述11题和空SYSTEM换行，再做119科12928题真实框架复验。正式完整评测配置和裁判具体身份仍缺失。评测运行锁、正式训练以及短步SFT/DPO继续关闭；本次没有新模型分数。

后续更新：本地非递归模板已修复，并通过119科12928题真实框架核验及配置重载，见[修复报告](week8_phase2_prompt_fix.md)。上述远端失败记录保留；本次本地通过不等于目标Linux修复复验或模型推理通过。
