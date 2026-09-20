# 第8周数据处理与评估协议 v1

冻结日期：2026-09-14。完成第一阶段任务2；机器协议：`configs/week8_data_protocol.json`。本协议优先于旧执行计划中的默认输入示例。此次只锁定输入和规则；实现、生成、抽查和真实loader验收属于任务3–6。

## 输入和身份

正式源为技术预检的 `train.sharegpt.jsonl`（1580条），绑定 `train_lineage.jsonl`。父集合为共同KEEP原文1663条，旧83条验证集仅作受保护历史诊断。所有路径和哈希从 `reports/week8/phase1/input_inventory.json` 引用，执行前重新核验；文件缺失或变更必须失败，不自动回退旧5000条。

tokenizer锁定技术预检 `remote_evidence/smoke_adapter/` 的本地保存版本（不是使用adapter权重），使用fast tokenizer、离线加载，transformers 4.50.0，seed/data_seed均42。Chat Template文本及五份tokenizer文件均记录SHA-256。词表与历史原始基座词表哈希相同；tokenizer.json与已交付版本相同，但tokenizer.json和tokenizer_config.json均不同于原始基座文件哈希。不能据此直接宣称行为完全等价。任务5须对当前LLaMA-Factory的qwen模板逐token核对；若不同，先修订协议和重跑数据，不静默替换模板。模型权重身份在GPU预检时单独核验。

## 处理顺序和边界

1. **解析与最小清洗。** UTF-8解析JSONL，检查每条ID/血缘、角色、非空内容；原文不进行全局strip、HTML删除、反转义或空白折叠，避免破坏代码、公式及字面反斜杠。仅将合法源字段映射到标准角色；无效记录隔离并记录原因。原始正文与清洗后正文哈希分别留存。
2. **长度检查与超长处置。** 对完整system和全部对话应用冻结模板，含终止符，不加生成提示；上限2048 token。v1选择超长整条隔离，禁止截断半个回答或删多轮上下文。截断策略明确为“不截断原文”，报告truncated=0、overlength_rejected数量及原因；2048恰好可保留。若以后必须启用截断，另立版本、保留前后内容并专项抽查。
3. **精确去重。** 标准角色序列与正文按UTF-8、JSON排序键、紧凑分隔符计算SHA-256，重复保留sample_id字典序最小者，其余记录duplicate_of。提示相同而回答不同不能当作全文重复删除。
4. **问题关系与受保护集合隔离。** 仅比较键使用NFKC、去全部Unicode空白和casefold；正文保持原样。比较所有对话轮次：非空精确匹配；长度至少32字符的较短串包含于较长串；双方至少32字符的字符5-gram Jaccard≥0.8，任一命中连边。合并历史split_group_links关系，用连通分量为问题组；近重复保留但不得跨划分。与83条或清单中的评估题命中时，整个相关候选组隔离，记录源文件、题目ID、命中规则及得分。评分rubric只固定评分，不当作题目文本。该规则是词面保护，不证明无语义泄漏。
5. **9:1确定性分组划分。** 对隔离后的N条，验证目标 floor(0.1×N+0.5)，train取余数。组ID为成员sample_id排序后以换行连接的SHA-256；按SHA-256("week8-v1:42:"+组ID)升序排列。用可达子集和选择完整组，使验证总条数最接近目标，同距取较小条数；同一可达条数采用排序遍历中第一次得到的组合。剩余组为train。这样固定取整、分组和同分处理，不拆组凑比例。划分后分别按sample_id排序，报告来源和多轮分布；不根据评估结果反复换seed或换分组。
6. **双格式导出与审计。** sample_id、血缘、排除原因放在独立审计文件，训练正文不混入审核批注。输出ShareGPT/Alpaca各一份train和validation，共四份JSON；每对文件逐ID、逐角色、逐正文可逆对齐。隔离样本不进入任一训练或验证交付文件。

## 训练表示

正式训练用ShareGPT：conversations中的human/gpt映射user/assistant，system保留为独立system字段（仅允许最前system；其他位置或工具轮次不支持时隔离，不压成文字）。LLaMA-Factory登记formatting=sharegpt，template=qwen，train_on_prompt=false、mask_history=false、packing=false、cutoff_len=2048。监督所有assistant回答及其EOS，非assistant标签为-100，至少一个有效监督token。

Alpaca作为可逆交付表示：最后一条user为instruction，input为空，最后assistant为output，此前成对轮次放history，system保留。禁止把13条多轮样本扁平化成单问单答。任务5需验证ShareGPT真实加载结果；Alpaca如需参与训练，必须另做等价加载验证。

## 评估口径（训练前固定）

- 新9:1 validation：以实际监督token加权的NLL为主要损失统计，另报逐样本均值及分布；固定模板与标签规则。83条保持独立同口径历史诊断。损失下降不直接等于问答质量提升。
- 模型比较：同一原始7B基座、新SFT最终完整训练checkpoint的合并模型、新DPO最终完整训练checkpoint的合并模型。正式运行前固定总步数/epoch，不根据20题或benchmark结果挑中间checkpoint；故障恢复必须记录，未完成训练不伪装最终模型。
- C-Eval和CMMLU：OpenCompass全科目（52/67），分别按题数加权汇总准确率，同时保留科目分数、样本数、预测与失败记录；任何缺科目、漏题或推理异常不得声称完整评估通过。各模型使用同一份数据、few-shot/模板、答案抽取逻辑和环境；禁止将旧结果回放标为新推理。
- 自定义20题：固定Week3 Day14的evaluation_questions.json和evaluation_rubric.json及其哈希，沿用system_message与题目顺序；greedy、do_sample=false、max_new_tokens=512、seed42。五维0–5分：准确性30%、完整性25%、逻辑20%、安全15%、格式10%；保存逐题理由、原始回答、评分请求和响应，模型总分为20题加权分算术平均。评分失败标失败，不能丢弃失败题后平均。
- AI评估与历史人工评分分开报告。裁判不能根据哪个模型得高分而更换；运行前记录其精确模型版本、提示词、参数和服务端身份。同一裁判盲化模型名后评全部模型；无法确认版本时报告限制。此次不调用外部裁判、不发送数据。
- 质量结论预先限定：完整运行、证据齐全仅代表工程完成。只有新模型在上述两个完整benchmark和自定义20题三个聚合指标均不低于同次重测基座，且至少一项更高时，才可表述为“本次固定评估整体改善”；否则逐项报告改善与回退，不换题掩盖。该条件不代替统计显著性，不声称隐藏测试或全面泛化。

公共benchmark本地快照、OpenCompass具体版本/配置和裁判部署标识目前尚未解析锁定。因此这里冻结方法和指标，**不声称评估运行环境已完全冻结**。必须在首次新训练前补充运行锁文件，包含benchmark实际题目/配置哈希、软件版本、few-shot及解码参数、裁判身份；将实际benchmark题目纳入污染检查并对最终数据重新验收。若新增排除或模板变化，则提升数据版本、重跑划分，不能直接沿用旧数据。此项是后续训练前的硬性检查，不是等待结果后补口径。

## 任务3接续清单

先让数据脚本读取协议、显式传入正式源并绑定血缘和排除清单；替换旧的默认Day6/Alpaca训练选择。随后在独立运行目录执行准备、确定性复跑及新增诊断题污染检查。报告实际条数和所有隔离记录，再进行任务4的分层抽查和任务5的真实loader核对。所有这些尚未执行，当前不能标记DATA_READY。
