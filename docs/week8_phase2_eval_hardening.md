# Week8 第二阶段：评测入口加固与提示预核验

日期：2026-09-15。承接320机基础预检，本轮完成三个本地子步骤，逐项验收。当前结论为 **LOCAL_EVALUATION_HARDENING_PASS_RUNTIME_PENDING**；不代表正式评测或训练通过。

## 步骤一：题目计数与计分校验——通过

旧入口使用 `len(details)` 作为题数。OpenCompass 0.5.3 的 `OpenICLEvalTask.format_details` 会在补充生成详情时加入 `type: GEN`，这不是题目。C-Eval的AccEvaluator经此路径输出；本轮所选CMMLU的AccwithDetailsEvaluator已自带详情，使用另一组字段，不应机械减一。历史1398/11649与当前1346/11582的差值分别是52/67，符合每科多计一个字段的特征，但没有逐份核对历史原始结果前，不宣称两个历史数字的完整成因均已证实。

修复后的入口只接受连续的0至N−1题目索引，以及可选的`type: GEN`。按冻结快照逐题核对参考选项，支持两种官方详情格式，独立重算正确题数及准确率；拒绝缺题、额外题、错位、陌生元数据和不一致的正确标记。汇总为总正确数／总题数，不再将错误题数用于加权。还要求科目名称集合与冻结快照完全相同。

验收：新增测试覆盖上述两种格式和异常情形。先运行失败，再修复通过。所有原有测试一起重跑，共31项通过。记录在`reports/week8/phase2_eval_hardening/tests_before.log`、`tests_after.log`和`full_tests.log`。

## 步骤二：未完成运行锁时禁止新评测——通过

`step3_eval.py`的fresh入口在调用OpenCompass、加载模型或调用裁判前，要求存在已验证的运行锁；检查配置与基准记录文件的SHA-256，以及裁判模型和地址的一致性。未完成时立即报错，失败状态仍会记录。历史replay保留原有含义。

已移除入口中`ceval_gen/cmmlu_gen`默认别名拼装；后续只能从锁定配置路径进入。**可执行完整配置、真实框架展开和目标环境复验仍待完成，本轮没有把候选配置视为运行锁。** 当前配置不含可放行的runtime_lock，新评测和正式训练继续关闭。拒绝提前启动的测试明确断言没有调用子进程。

## 步骤三：全题五样本提示独立重建——本地检查通过，框架等价性待验收

使用已校验的0.5.3官方配置源码，将导入类型保留为字符串进行静态展开，生成119科候选数据配置。此过程不是mmengine真实注册或配置加载。C-Eval选择`ceval_gen_5f30c7`，CMMLU选择`cmmlu_gen_c13365`，均取dev前5条，计分split分别为val和test。

按照官方CSV读取方式处理换行，36条记录与保留原始CRLF的读取结果不同；原始文件未改动。使用冻结Qwen tokenizer，逐题构造5组问答加当前问题，并验证chat_template直接tokenize与渲染后tokenize逐token相同。输出每题token数与token序列哈希。

- 119科，12,928道计分题，全部参与检查。
- 最长输入1,361 token；加32 token输出为1,393，小于2,048。
- 输入超2,048为0，输入加输出超2,048也为0。
- 维持评测上下文2,048、输出上限32的候选设置；不改变SFT的2,048限制。

OpenCompass的GenInferencer存在过长时减少示例的代码，模型包装器也存在截断路径。独立重建足以说明当前候选没有显式长度溢出，**尚不足以证明实际框架提示相同、没有减示例或截断**。下一次必须让真实Dataset、FixKRetriever、PromptTemplate和模型模板解析器生成全部提示，与本轮逐题token哈希对照。模型生成设置、种子、批量、数据路径和包版本也须在完整展开配置中验收后，才能签发runtime_lock。

可复现命令（使用现有CPU审核环境）：

```bash
/tmp/internship-week8-loader-20260915/bin/python scripts/audit_week8_benchmark_prompts.py
/tmp/internship-week8-loader-20260915/bin/python -m unittest discover -s tests
```

## 下一步顺序与通过条件

1. 实际OpenCompass配置与全题提示复验：119科、12,928题、每题5示例，token哈希对照一致；禁止用静态展开替代。
2. 320机实际1580条数据加载复验：与本地1422/158拆分、input_ids、labels逐项相同，记录目标环境依赖差异。上次320机会话已结束；本轮未开机或发生新的GPU任务。
3. 固定裁判模型具体版本和API地址，完成裁判协议验收；当前用户尚未提供，不使用猜测的模型或地址。
4. 以上条件完成后，再按此前顺序做短步SFT/DPO冒烟；正式训练仍需独立放行。

没有新模型分数、训练结果或目标Linux数据加载结果产生。第一阶段冻结归档不做就地更新。
