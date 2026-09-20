# Week8 正式评测格式修复与续跑方案

## 本次已完成

正式任务在2026-09-16 23:43 UTC停于原基座客观题验收。CMMLU真实结果的 `details` 是列表，行内 `pred/refr/is_correct` 均是单元素列表，且含 `example_abbr`。此前单元测试使用了标量字典的替代样例，未覆盖实际输出，造成正式入口错误拒绝了完整结果。

已修复 `scripts/step3_eval.py` 的 `validate_subject_result`。原有字典校验保持有效；列表格式只允许绑定CMMLU学科，逐项验证明细长度、`cmmlu-{subject}_test_{index}`题号、三个字段的单元素列表结构、布尔正确性、冻结标准答案和独立重算准确率。不会改变或重写原始结果，不调整推理参数、题目或评分规则。

新增回归测试覆盖真实格式、原对象不变、缺少学科绑定，以及缺行、多行、重排、重复题号、错误学科、空/多预测、标量reference、错误gold、错误/非布尔/缺失正确性标记、虚假准确率等拒绝路径。真实输出已在修复前复现 `missing benchmark details or golds`，修复后通过。

真实产物复验脚本：`reports/week8/phase2_eval_schema_fix/review_recovered.py`。它先核对374份文件的manifest大小及哈希、冻结benchmark records哈希，再调用修复后的生产校验函数检查119学科、12,928题。结果为CEval 1049/1346，CMMLU 9246/11582，与上轮独立核验一致。输出保存在同目录 `recovered_review.json`。

## 冻结绑定仍然阻止直接运行

旧裁判profile和三个runtime lock绑定旧脚本哈希。修复后的脚本必须重新审核并建立新版本绑定，不能直接沿用旧放行状态。本轮保留所有已放行profile、runtime lock及历史失败产物，不在原记录上伪造“已完成”。测试使用临时的当前代码绑定，并验证脚本哈希错误仍被拒绝。

320机保持上轮已确认的关机状态，监控暂停。本次只做本地修复和离线验收，没有启动GPU，也没有调用裁判API。

## 受控续跑入口设计（本地实现已完成，尚未部署）

1. 新建独立输出目录和专用续跑入口，禁止覆盖失败会话；拒绝默认走`fresh`而重复调用OpenCompass。输入须显式绑定旧会话、原基座模型路径、完整manifest、模型/数据/推理配置身份，以及新审核的代码和裁判profile。
2. 在启动GPU或付费裁判调用之前，验证来源manifest及文件集合，确认确为本次原基座的119学科12,928题，并执行本轮已通过的逐题校验。验证旧会话未生成custom20、未发出裁判请求。来源不完整、错模型或内容变化一律停止。
3. 仅复用原基座的客观题结果，重新生成其20条custom20回答并完成裁判评分。新汇总必须明确客观题来自旧会话，custom20来自续跑会话，分别保留哈希和来源；不能宣称全部推理发生于同一新运行。
4. 原基座完成后，再依次执行尚未启动的SFT和DPO全量评测。仍使用相同题目、BF16和已冻结推理参数、裁判模板、限额及停止条件。
5. 先在本地用无GPU、无API的测试验证：有效证据能走到custom20边界，所有异常来源在加载模型或付费前失败，且不会调用OpenCompass或覆盖旧结果。然后审核新版绑定，在目标机完成身份检查后才启动GPU。

以下本地实现已完成；三模型正式评测仍未完成。


## 受控续跑入口实现与本地验收

入口为 `scripts/week8_resume_eval.py`，提供显式 `--source/--output-dir/--model/--judge-model/--judge-url` 参数。它首先使用正式入口的运行锁校验，再验证绑定manifest的哈希、精确文件集合、全部374份文件的大小及哈希，拒绝软链接、额外文件或缺失文件。随后核对旧放行锁、原始launch、模型/数据/推理配置、旧依赖与新依赖的一致性（允许已审核的评测脚本变更），确认旧运行正是本次格式失败且未尝试custom20，最后重算全部客观题分数。

新增入口不含OpenCompass启动路径。正常评测与续跑共享 `validate_benchmarks` 和 `complete_custom20`，custom20仍使用同一BF16加载、seed42、greedy、512新token、题目/模板及裁判协议。新summary中的客观题明确标为 `reused_verified_opencompass`；新status标记 `resumed_custom20` 与 `benchmark_inference_reused`，记录旧manifest/旧锁与新运行锁的哈希。

原始证据保持只读，新输出目录必须新建且不能位于来源目录内。付费阶段前创建 `logs/resume-claims/{manifest_sha256}.started` 独占记录，失败后仍保留；重复调用会被阻止，不可自动删除该记录后重跑。

本地87项测试全部通过，包括有效来源进入共享custom20边界、错误来源在GPU/API前被拒绝、OpenCompass零调用，以及失败后重复启动被阻止。GPU生成与付费评分在这些边界测试中使用mock，没有执行真实GPU/API操作。真实374份文件和12,928题来源校验另行执行通过。

证据目录：`reports/week8/phase2_eval_resume/`。其中 `local_review.json` 记录当前代码及测试日志哈希，`source_validation.json` 记录来源检查。新的裁判profile候选已通过本地参数与校准绑定检查；三个锁候选均为 `verified:false`，不具备执行放行资格。旧profile、旧runtime lock和历史失败证据未改写。

下一步：在目标机核对新代码、候选profile、旧证据包和三个模型依赖；审核并生成新版放行锁；配置新的有截止时间的串行协调会话，先原基座custom20续跑，再SFT/DPO全量评测。旧协调脚本具有过期时限和旧锁哈希，不能直接再次运行。最后才启动GPU并恢复监控。

## 目标机放行更新

2026-09-17无卡部署、模型依赖核验和新版锁放行均已完成；GPU空闲0，实例已关机等待，监控已恢复。最新状态见`docs/week8_resume_target.md`。上文“尚未部署”描述保留为本地实现阶段的历史状态，不能作为当前运行状态。正式续跑尚未启动。
