# Week8下一阶段：评估输入预检

日期：2026-09-15。当前完成benchmark快照和词面交叉核验，**评估运行环境与GPU试跑尚未通过**。状态为BENCHMARK_INPUTS_CHECKED_RUNTIME_AND_TARGET_PENDING。

## 本轮完成

从官方数据仓库下载160份文件，绑定精确revision和SHA-256；生成适配本地CSV读取布局的290份数据文件并独立逐字段复核。原始快照、转换文件、统一题目记录均保留。

| 数据集 | 计分范围 | 计分题 | 5-shot示例 | 额外保护 |
|---|---|---:|---:|---:|
| C-Eval | 52科的val | 1346 | 260 | test的12342条，仅作保护，不计分 |
| CMMLU | 67科的test | 11582 | 335 | — |
| 总计 | 119科 | 12928 | 595 | 12342 |

版本：[C-Eval官方快照](https://huggingface.co/datasets/ceval/ceval-exam/tree/617524a00b307ff6f9933702f724131fe12ca7ce)、[CMMLU官方快照](https://huggingface.co/datasets/haonan-li/cmmlu/tree/efcc940752ea4a1ea94d2727f11f83858d64fc8e)。合计25865条题目/示例；C-Eval未公开答案的test不当作有标签计分集。

使用冻结协议的NFKC、去空白、casefold、精确/长串包含/5-gram Jaccard≥0.8规则，对本周1580条数据全部3198轮检查。比较benchmark题干及“题干+四选项”两种视图，共51328份去重比较文本，**词面命中0**。独立匹配谓词与原实现进行了4005组对照。该结果不证明无语义泄漏。

独立验证再次解析所有官方原文件，逐条核对25865条内容、选项和答案；160份下载和290份本地文件哈希通过，第一阶段15份数据及137份归档保持不变。原协议、划分和训练门禁未改动。

## 发现的问题

**CMMLU默认别名与短输出配置不相称。** 实际检查OpenCompass 0.5.3 wheel发现cmmlu_gen引用cmmlu_0shot_cot_gen_305931，提示要求逐步思考并在最后一行给答案；现有scripts/step3_eval.py却只允许32个输出token。这是运行前发现的截断风险，本轮没有运行模型，因此不是已实测截断。

已固定显式候选配置ceval_gen_5f30c7与cmmlu_gen_c13365，两者都用dev的第0–4条做5-shot并直接输出选项；各自完整提示和答案抽取代码均已保存。这些文件与[OpenCompass 0.5.3官方提交](https://github.com/open-compass/opencompass/tree/a25fdd25dbe6a6e1da355332fa52e82dcfc3a1ee)逐字节一致。**尚未将候选配置接入正式评估入口，也未实际执行OpenCompass，不能称为运行锁完成。** 下一步必须使用固定本地数据、核验展开配置与所有题目的实际提示长度；2048上限和32输出预算需经过真实渲染验收，不允许静默截断。所有新模型和同次重测基座使用同一配置，不能与旧结果直接拼接比较。

**历史汇总的题数不同。** 历史记录为1398/11649，比本轮官方解析值分别多52/67，差值与科目数相同。原因尚未复核，不推断旧模型分数一定错误；本轮以实际题目和每题预测计数，不能将元数据项或表头当作题目，也不再照抄历史总数。

## 证据入口

- [机器状态与缺项](../configs/week8_evaluation_preflight.json)：训练放行false，GPU/candidate judge仍为null，尚未接入训练或评估入口。
- [benchmark输入锁](../reports/week8/phase2_preflight/benchmark_input_lock.json)：数据revision、wheel/config哈希、具体科目及split、few-shot和解码候选。
- [全量交叉核验](../reports/week8/phase2_preflight/benchmark_overlap_verification.json)与[独立快照审计](../reports/week8/phase2_preflight/independent_verification.json)。
- [下载清单](../reports/week8/phase2_preflight/download_manifest.json)、[转换清单](../reports/week8/phase2_preflight/local_data_manifest.json)、[官方配置对照](../reports/week8/phase2_preflight/official_config_verification.json)。

脚本位于同一证据目录：download_benchmarks.py、prepare_snapshot.py、check_overlap.py和verify_snapshot_independent.py。解析/独立核对使用任务5 CPU环境中的pyarrow；重叠检查使用标准库及冻结规则。未运行新的训练回归测试，因为本轮没有修改训练实现；实际执行的是快照逐条核验和全量词面比较。

本轮独立CSV验证第一次使用默认换行转换读取，遇到原题内CRLF被读取为LF而失败；改为newline=''后证实转换文件保留了原内容。真实OpenCompass读取可能采用其自身换行处理，需在实际渲染验收中记录。初次官方源码请求误用根目录configs路径返回404，随后以tag内实际opencompass/configs路径取得源码并验证一致；均未作为通过记录掩盖。

## 继续条件

本轮已询问用户GPU实例和自定义20题裁判模型/版本/接口地址，尚未收到。不要提交密码或API密钥到聊天、仓库或日志。确认目标后还需实时核验连接、软件、完整原始7B权重及CUDA；历史320实例记录不作为本次可用证明。

下一步先完成OpenCompass固定配置/本地数据接入与提示长度检查，并补齐裁判身份；完整运行锁写入后，重新核对其绑定的污染收据，再进行GPU和原始基座预检。任何前置条件未过都不能进入3步SFT/DPO试跑。
