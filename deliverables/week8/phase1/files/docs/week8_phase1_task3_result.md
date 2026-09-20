# 第8周第一阶段任务3：全量数据准备与机器验收

完成日期：2026-09-14。状态：**TASK3_MACHINE_CHECKS_PASS_NOT_DATA_READY**。本次完成数据处理入口、全量生成、重复运行与独立机器校验，任务4内容抽查、任务5真实loader和任务6最终发布仍待完成。

## 实际结果

| 项目 | 结果 |
|---|---:|
| 正式源 | 1580条 |
| 合法且不超过2048 token | 1580条 |
| 精确去重后 | 1580条 |
| 保留评估题命中后隔离 | 0条 |
| 最终训练 / 验证 | **1422 / 158，严格9:1** |
| 历史验证集 | 83条继续隔离，未并入新数据 |
| 本轮改写 / 截断 / 拒绝 | 0 / 0 / 0 |
| 训练 / 验证最大长度 | 2048 / 1949 token |
| 训练 / 验证多轮对话 | 13 / 0 |
| 训练 / 验证system样本 | 0 / 0 |
| 去重后的保留问题 | 504条 |
| 与保留问题的词面命中 | 0 |
| 跨训练/验证词面关系命中 | 0 |
| 重复运行产物 | 15份JSON逐文件字节哈希一致 |

源正文保持不变指与本次核定的1580条源文件一致，不追溯宣称更早的原始网络数据从未被处理。504条包含83条历史验证的用户问题和清单中的历史/后续诊断题，按规范化文本去重；不包含尚未锁定本地快照的全量公共benchmark题目。

| 来源 | train | validation | 合计 |
|---|---:|---:|---:|
| alpaca_gpt4_zh | 687 | 74 | 761 |
| coig_pc | 669 | 76 | 745 |
| sharegpt_zh | 66 | 8 | 74 |

训练token中位数97、P95为390；验证中位数97、P95为543。共1342个问题组，其中75组含多条记录，最大组27条。当前协议比较候选所有轮次（包括回答），因此组数与历史只围绕问题的划分不能直接相比；近重复关系用于完整分组，没有删除这些记录。新验证集没有多轮样本，不能单凭它评价多轮能力；13条训练多轮在任务4专项抽查，生成能力仍需后续固定评估验证。

## 代码和入口变化

- `scripts/week8_data.py`实现冻结协议：输入与tokenizer哈希检查、严格角色解析、超长整条隔离、精确去重、历史关系保留、全轮次词面分组、保留题整组隔离、确定性子集和划分、可逆双格式导出及审计。
- `scripts/step1_data_prep.py`正式模式默认加载协议；`run_pipeline.sh`同样使用冻结源。正式模式拒绝input/tokenizer/seed/长度参数覆盖，旧5000条流程只能显式使用quick或legacy。
- `dataset_info.json`登记ShareGPT。sample_id、源记录哈希、来源和完整血缘放在独立`lineage.json`，通过split及从0开始的row_index与两种文件对齐；训练正文不含审核批注或记录ID。
- 训练入口明确拒绝任务3状态，避免将“机器处理成功”当作“可以正式训练”。此次未启动训练、GPU或外部裁判。

协议v1.1仅修正三份评估元数据计数（items结构200/200条、prompts结构15条），并显式绑定已要求保留的历史分组文件哈希，处理规则未变。原v1清单与协议均保留快照。解析器对未知/空评估结构失败，不再默认为0题。

## 产物与复现

主结果：[run-a目录](../logs/week8-protocol-data-20260914/run-a/)；总入口独立复跑：[run-b/data目录](../logs/week8-protocol-data-20260914/run-b/data/)。

四份数据文件：`train_sharegpt.json`、`validation_sharegpt.json`、`train_alpaca.json`、`validation_alpaca.json`。其余包括dataset_info、lineage、groups、group_edges、lengths、exclusions、protected_hits、protected_coverage、input_receipt、protocol和statistics。空排除及命中清单仍保存为JSON数组，不能把文件不存在当作零命中。

在已安装锁定依赖的Python环境中执行，目录必须尚不存在：

```bash
python scripts/step1_data_prep.py --protocol configs/week8_data_protocol.json \
  --output-dir logs/your-new-run-a
PIPELINE_PYTHON=python bash run_pipeline.sh --skip-train --skip-eval \
  --run-dir logs/your-new-run-b
python reports/week8/phase1_task3/verify_outputs.py \
  logs/your-new-run-a logs/your-new-run-b/data \
  --receipt logs/your-independent-verification.json
```

正式运行前取消旧DATA_INPUT/TOKENIZER_PATH覆盖。本次实际解释器、版本、脚本与协议哈希见[执行清单](../reports/week8/phase1_task3/execution_manifest.json)。

## 验证证据

新增12项回归测试和原9项流水线测试合计 **21项通过**，覆盖多轮/system/字面空白保留、非法角色和空回答、评估items/prompts解析、归一化和近重复、回答命中后的整组隔离、长度边界、精确重复代表选择、历史关系保留、按条数分组划分、哈希篡改、参数冲突及未发布数据的训练拒绝。

独立导出审计脚本不导入准备实现。它从输出重新复查源正文、双格式可逆性、角色、条数、样本归属、所有跨划分词面关系、所有保留题词面关系及真实tokenizer长度，并比较两次文件哈希。证据见[独立验证收据](../reports/week8/phase1_task3/verification.json)、[测试记录](../reports/week8/phase1_task3/tests.txt)。

局限：词面检查不证明无语义泄漏；本地tokenizer长度不替代LLaMA-Factory实际labels/EOS/模板验收；当前没有system实例，相关逻辑只通过合成边界测试。公共benchmark数据快照与裁判身份仍须在首次新训练前锁定，并纳入污染复检。本次不宣称数据最终发布或模型质量改善。

## 下一步

只进入任务4：分层抽查约30–50条，覆盖三种来源、13条多轮、长短样本、代码和数学；记录正文、上下文、答案完整性问题。当前无截断或隔离样本，此类抽查记为不适用。抽查完成后再进行任务5真实LLaMA-Factory加载验证。
