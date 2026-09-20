## 2026-09-17：custom20诊断与人工盲评材料已完成

60条回答完成辅助检查；18份代码已在本机隔离Docker中执行冻结断言。原基座代码通过3/6，SFT与DPO各1/6；数值末值匹配分别4/7、2/7、2/7。两份各60条匿名人工评分表已生成且核验，人工评分仍为空。无GPU启动，无API调用。详情见docs/week8_custom20_diagnostic_review.md；验收reports/week8/phase2_custom20_review/verification.json。

下一步：实际人工独立盲评；并继续整合Week8报告、盘点部署/压缩/干净环境剩余验收。不得将空白盲评表当作人工完成或自动重新训练。

---

## 2026-09-17 22:39 UTC：三模型 Gemini 评分完成，比较结论须谨慎

SFT 与 DPO 各20题完成评分并通过原始响应、请求标记、分数、均分及用量复核，共40次新调用、零重试、153932 token。原基座结果未重跑；320保持已关机，GPU监控保持暂停。

custom20 AI 均分：原基座3.9775，SFT 3.8275，DPO 3.8700。**不能认定DPO主观质量优于SFT**：12条完全相同回答中7条评分不同，最大差0.8；相同回答评分波动对总均分差贡献+0.075，大于观察到的总差+0.0425。评分流程验收通过，排名解释存在限制。

详细报告：docs/week8_trained_gemini_scoring.md；逐题对照：reports/week8/phase2_gemini_trained_release/comparison.json；各模型验收回执：reports/week8/phase2_gemini_score_final_sft/verification.json、phase2_gemini_score_final_dpo/verification.json。

下一步：核对确定性自动检查和人工评分要求，整理逐题错误类型及人工盲评材料。不得自动重跑GPU、重评刷分或将单次AI评分冒充两人评分。

---

## 2026-09-17 21:00 UTC：SFT/DPO 生成与结果验收完成，320 已关机

两模型均完成完整 CEval 52科/1346题、CMMLU 67科/11582题及 custom20 各20条回答。远端与本地分别重新按冻结标准答案核对238个学科明细与正确数；40条自定义回答题号唯一、非空、顺序及SHA一致。

| 模型 | CEval | CMMLU | 自定义回答 |
|---|---|---|---|
| SFT | 1072/1346，79.6434% | 9260/11582，79.9516% | 20/20 |
| DPO | 1077/1346，80.0149% | 9264/11582，79.9862% | 20/20 |

750份原始文件已回收至 reports/week8/phase2_trained_generation/retrieved，并逐文件核验大小及SHA256。归档8255948字节，SHA256为23c763762011c03c3de6a522a2af00af7368f1f5f152ca5be9122dd875bebf4d。打包清单为合法JSON，末尾为普通换行；已保留原始字节，模型结果文件没有修改。独立本地验收：generation_verification.json；下载审核：retrieval_verification.json。

协调器及本任务计算进程已退出，GPU计算进程为空；内嵌浏览器确认320显示“已关机”，监控week8-320已暂停。没有重试推理，没有调用裁判API。关机证据：gpu_shutdown_receipt.json。

下一步：适配Gemini仅评分入口，使其验证此次SFT/DPO的新来源与验收回执，再按与原基座相同协议各评20题。不能直接复用原基座专用prepare，也不能把生成完成写成主观评分或全部实习任务完成。

---

## 2026-09-17 18:18 UTC：320 正式 SFT/DPO 会话已启动

协调器PID1593，启动时间18:18:16.607557 UTC。已部署独立 released_plan.json 与 session_release.json 并逐文件核验发布包哈希；原候选计划和原审核回执保持不变。远端CUDA、BF16和GPU空闲检查通过，RTX3090一张。终端原有OMP_NUM_THREADS无效警告仅见于预检查；正式协调器启动环境明确设OMP/MKL线程14。

运行目录：`/root/autodl-tmp/week8-judge-runtime-20260916/logs/trained-generation-20260917`。协调日志：`/root/autodl-tmp/week8-judge-runtime-20260916/reports/week8/phase2_trained_generation/coordinator.log`。顺序为final_sft→final_dpo，逐模型完整CEval52科/1346题、CMMLU67科/11582题与custom20回答；不重跑原基座，不调用裁判API。已观察到SFT进入OpenCompass：划分119个学科任务，GPU利用率24%、占用14794 MiB；当时尚无完成结果，不能声称评估通过。

保护窗口：最晚启动2026-09-17 18:40 UTC（北京时间9月18日02:40）；程序截止9月18日03:10 UTC（北京时间11:10）；平台关机03:15 UTC（页面伦敦时间04:15，北京时间11:15），开机前及开机后实例行均确认存在。开机前余额16.40元，GPU单价1.48元/小时，平台估算本窗口13.31元；实际以账单为准。

既有监控week8-320已替换旧窗口指令并恢复ACTIVE，每5分钟只监控本会话，不再派发、不重新开机、不自动重试。完成或失败后回收轻量证据并核验，及时关机、暂停监控；不应仅凭文件计数/退出码判定验收完成。旧窗口、旧DeepSeek入口及原基座运行均不再触发。下一阶段是审核新回答和完整基准结果，再适配并执行本地Gemini同协议评分。

本轮验证：6项针对性unittest测试通过；released_plan本地非权重依赖全部通过；发布包远端哈希通过；GPU预检查通过。详见 reports/week8/phase2_trained_generation/launch_observation.json、platform_shutdown.json、session_release.json。历史CPU审核与关机记录仅代表上一轮，不代表当前实例状态。

---

## 2026-09-17 17:40 UTC：目标 CPU 审核已通过，GPU 正式启动待发布

已在320无卡模式部署 generation-only 候选入口，逐文件核验 final_sft/final_dpo 的完整文件集合、大小和 SHA-256，并核对旧运行锁与依赖。目标回执为 `TARGET_GENERATION_AUDIT_PASS`，审计计划 SHA 为 `a1dcb2a9d71b98db4fd6a9892dec3dc5f3a4f7185459f7a2a13b0226ceb6067e`。

首次检查发现新运行目录缺少历史 target_review.json；补齐三份与旧锁哈希一致的历史回执后重新进行只读审核，通过。两次日志均保留，未进行推理或裁判付费重试。三份远端证据回收到本地后再次核对 SHA-256 一致。

目标环境包版本为 torch 2.5.1+cu121、transformers 4.50.0、opencompass 0.5.3；剩余数据盘 21,538,770,944 字节。这里验证的是文件及包版本，尚不等于 GPU 模型加载测试通过。

新增本地会话协调器 `scripts/week8_trained_session.py`，按 SFT→DPO 执行，检查发布哈希、窗口和 GPU 占用，超时清理进程组，失败不自动重试。6项针对性 unittest 测试通过（生成入口3项、进程组清理3项）；未声称完整仓库测试通过。最初 pytest 调用因系统 Python 没有 pytest 未执行，随后用这些测试原生使用的 unittest 运行通过。

`window_candidate.json` 是未启用的9小时窗口提案，按当时页面 GPU 单价1.48元/小时估算13.32元；启动前仍须核对余额、资源及实际单价。旧绝对时间窗口已过期，旧监控保持暂停。没有设置新的平台定时关机，没有发布协调器，没有启动 GPU 推理。

收尾：已通过内嵌浏览器确认320显示“已关机”；344未操作。

下一步：审核并部署协调器及独立发布计划，配置新绝对时间窗口与平台关机保护，再启动 SFT/DPO 推理。回答生成后另行做来源完整性审核与 Gemini 评分。目标审核证据位于 `reports/week8/phase2_trained_generation/{target_audit.json,target_review.json,audit.log,audit-2.log}`。

---

## 2026-09-17 最新进展：SFT/DPO独立生成入口本地准备完成

已识别旧入口会调用DeepSeek，新增generation-only入口，保留旧模型和数据锁，后续回答回本地按Gemini统一评分。本地非权重依赖审核及3项测试通过；320目标权重本轮尚未复核。计划保持候选状态，不能启动GPU。旧窗口15:00 UTC最晚启动时间已过，须重新安排运行窗口与自动关机保护；没有沿用旧启动器开机。详情见docs/week8_trained_generation_preparation.md。

---

## 2026-09-17 最新结果：原基座custom20 Gemini正式评分完成并复核

Gemini正式仅评分入口已实现，执行器8项测试及解析器2项测试通过。运行来源20条回答、17例校准、协议和代码依赖均重新核验后放行。20次请求全部首次返回有效结果，零重试，未重新推理、未启动GPU、未混合旧DeepSeek分数。

独立复核重新生成放行计划、核对20份请求与原始响应、返回模型身份、已保存分数和用量，加权均分3.9775/5。各维均分：准确性3.30、完整性4.30、逻辑性3.70、安全性4.85、格式4.45。输入52603、最终输出5289、思考21225，共79117 token。按标准价不计缓存折扣估算约$0.423374，不是实际账单。

结果标记ai_judge_not_human，不能当作两位人工评分员。低分样本包括MATH-04（1.3）、REASON-07（1.5）、CODE-04（1.6）；完整逐题结果及理由均有原始响应。MATH-05题干的潜在断句歧义仍须在分析中说明，不据单题认定模型优劣。

独立回执：reports/week8/phase2_gemini_score_original_base/verification.json，状态VERIFIED_COMPLETE。逐题分数：同目录summary.json。执行计划：reports/week8/phase2_gemini_score_release/plan.json。实现与约束：docs/week8_gemini_formal_scoring.md。

下一步是为SFT/DPO统一采用该Gemini协议准备正式推理与评分计划，先复核模型产物、来源和运行锁，再处理320实例GPU可用性。原基座已完成，不应再次产生其20题推理或评分费用。当前没有启动SFT/DPO任务，旧GPU监控仍不应直接以旧裁判配置自动续跑。

---

## 2026-09-17 最新状态：Gemini修订口径17例校准通过

明确“内容覆盖”和“结构数量”的独立归因，增加3个隔离样本；原14例输入与预设阈值不变。Gemini 3.1 Pro Preview保持high思考、16384输出上限，17次请求均一次通过，无重试、无JSON修补。独立复核冻结manifest、候选答案、原始评分、预设阈值、配对差异与累计用量后，结论SMALL_BATCH_PASS_NOT_FORMAL_RELEASE。

三个隔离样本正确分别扣格式、完整性或两者。MISSING-REASON本轮完整性3、格式3，各有不同证据；REASON-06布尔false被正确理解为不能宣布有效，五维均5。两组配对的所有维度与加权分差均为0。输入37295、最终输出4135、思考11450，共52880 token。相关离线测试13项通过，不代表全套环境回归通过。

本轮保持旧试验失败记录，不替换旧成绩。小样本通过不保证未来请求永不失败，诊断样本也不是独立无偏测试集。MATH-05仍含“对折后价格”的潜在断句歧义，本轮符合预设阈值不构成消除歧义的证据；正式报告应保留该题的解释限制，不能以这一题单独证明模型优劣。

候选协议已导出reports/week8/phase2_gemini_calibration_v2/judge_profile_candidate.json，明确formal_entrypoint_integrated=false、formal_scoring_approved=false。未启动GPU、未更改正式运行锁、未进行正式20题评分。

下一步：将该候选接入独立仅评分入口，验证20条来源答案、协议哈希、返回模型身份、用量、防重复付费与失败恢复，然后再发布新的运行锁。禁止直接重用旧DeepSeek执行器或把历史部分成绩混进新总分。

详细口径：docs/week8_gemini_dimension_calibration.md。独立证据：reports/week8/phase2_gemini_calibration_v2/verification.json。

---

## 2026-09-17 Gemini接入成功；校准停在维度归因分歧

本地Key权限与格式检查通过，实际调用模型gemini-3.1-pro-preview成功。采用high思考、maxOutputTokens=16384、JSON Schema、每例一次且零自动重试。两个兼容性样本EQ-TEXT、STRICT-BOOL均通过。完整14例校准前5例通过，第6例MISSING-REASON结构有效但预设format<=3检查失败，已停止，后8例未调用。

本轮共8次请求（探针2+校准6），所有8份响应通过严格结构解析，未出现旧服务的截断/缺括号/额外包装。此为小样本观察，不能宣称接口永不失败。合计用量{"promptTokenCount": 15726, "candidatesTokenCount": 1972, "thoughtsTokenCount": 5030, "totalTokenCount": 22728}，包含思考token，不等于实际账单。Gemini解析与Key测试4项、既有裁判加固5项，共9项通过；不宣称全套环境测试通过。

失败样本要求恰好3个理由字符串，答案仅给2个且遗漏“无对照组”。Gemini给完整性3、格式5，理由明确将数组少一项归于内容遗漏。预设检查只检查格式<=3。原rubric完整性要求覆盖显式要求，格式也要求严格满足全部形式约束；现有操作口径要求避免跨维度补扣，却没有清晰说明“遗漏实质理由同时违反显式数组长度”这一交叉情形。因此本次证据首先指向口径边界未充分操作化，不能简单归结为新模型能力不足，也不能直接改阈值使现有结果通过。

建议下一步先冻结通用归因规则：明确数组数量/字段类型为格式约束；实质依据是否覆盖为完整性；两者可以同时存在，但必须分别引用不同证据。设计三个隔离样本：内容全部覆盖但合并为2个字符串；3个字符串但重复一个而漏重要依据；2个字符串同时缺重要依据。再检查旧样本与新样本，不改动本批原始结果、不将失败升级为通过。此建议尚未作为新协议发布或调用。

证据：reports/week8/phase2_gemini_probe/verification.json、reports/week8/phase2_gemini_calibration/verification.json、reports/week8/phase2_gemini_calibration/diagnosis.json。模型回答仍可复用，未启动GPU或正式评分。

---

## 2026-09-17 最新状态：Gemini本地接入准备完成，等待Key配置

用户继续推荐路线。已新增scripts/setup_week8_gemini_key.py，采用隐藏输入、仓库外私有文件、拒绝覆盖和本地检查；4项相关测试通过。本地检查确认Gemini Key尚未配置，未调用Gemini API或开通付费服务。步骤见docs/week8_gemini_setup.md。正式评分和GPU继续暂停。

---

## 2026-09-17 最新状态：同服务Pro候选失败，等待独立服务接入选择

已测试deepseek-v4-pro，保持严格工具协议、思考强度、预算和评分规则不变。STRICT-BOOL单例通过；完整v9校准在第2例EQ-TEXT因额外arguments包装而停止。3次调用、零重试，共10177 token。未修复后计分，GPU仍未启动，正式评分未放行。

独立复核证据：reports/week8/phase2_judge_calibration_v9/verification.json。替代方案、接入要求及后续校准门槛已写入docs/week8_judge_backend_selection.md。推荐候选Gemini 2.5 Pro需另一个服务的账户/凭据；已询问用户是否可用，未开通账单或调用该服务。

---

## 2026-09-17 严格工具调用复核：未放行

根据官方strict工具调用文档，测试Beta接口https://api.deepseek.com/beta/chat/completions，function.strict=true，所有对象字段required且additionalProperties=false。保留deepseek-flash、thinking enabled、low推理强度、8192上限；使用auto工具选择，因为官方说明思考模式不支持required或指定工具。输出只作为submit_scores参数解析，不执行工具。相较v7改变的是输出通道及对应提交指令，未调整评分标准或预设检查。

两个曾失败样本EQ-TEXT、V3-01的独立兼容性探针均通过。之后v8完整14例校准的前三例EQ-BOOL、EQ-TEXT、WRONG-CONCLUSION通过，两种同义答案五维完全一致。第4例STRICT-BOOL失败：finish_reason为tool_calls，但arguments含不允许的logic_placeholder字段与非法双冒号，标准JSON解析失败。实际completion_tokens=716，远小于8192，不能归因于上限。程序停止，后10例未请求，没有补全或修复JSON，也未重试。

本轮探针2次、校准4次，共6次请求、零重试。合计输入14864、输出4358、总计19222 token。独立复核了冻结manifest、输入与原答案一致性、所有原始响应、保存评分、同义对照和用量。余额不是精确账单，未据余额变化推断单次价格。新增严格工具解析器3项测试通过（包括多种异常子例），并复跑Responses解析器6项和旧裁判加固5项，共14项通过；不等于完整Week8环境回归通过。

结论：当前实际strict工具响应也未满足正式评分要求。不能仅凭接口接受strict参数或两次探针成功宣称结构可靠，也不能因前三例分数正常把整批判为通过。GPU未启动、正式锁未更改、旧回答及旧成绩未覆盖。继续暂停正式评分，不再重复同类提示词/token/接口试跑。

后续应先确定替代裁判后端，或者有明确新依据的不同协议，再统一跑同一套校准；不要重新生成已保存的20条GPU回答。替代后端需要单独核实可用性和接入凭据，当前没有擅自购买或开通其他服务。

证据：reports/week8/phase2_judge_strict_probe/verification.json；reports/week8/phase2_judge_calibration_v8/verification.json；reports/week8/phase2_judge_calibration_v8/turn_summary.json。
参考：https://api-docs.deepseek.com/guides/tool_calls/ 与 https://api-docs.deepseek.com/api/create-chat-completion/。

---

## 2026-09-17 最新复核：评分语义对照通过，输出可靠性仍未通过

已按批准的顺序完成评分口径修订、对照校准和结构化输出兼容性检查。正式评分继续暂停，未启动GPU、未更改正式运行锁、未混合不同协议成绩。

### 评分口径（v6）

新增规则明确：题干的显式约束优先；参考答案中的具体字符串、布尔类型不能自动成为额外要求。题干未指定decision类型时，结合全文将false理解为“不能宣布有效”；题干明确限定字符串枚举时则严格检查。真正错误结论与遗漏要求仍须扣分。

冻结14例中，6个新对照全部首次通过：布尔/文字同义正确答案、真正错误结论、严格类型下的违规与合规答案、缺少一个理由。两种同义答案五维评分完全一致。这是有针对性的局部证据，不代表裁判普遍准确，也不是全套校准通过。

第7例V3-01两次收到非法JSON后停止，均正常stop且分别只用1014、888个输出token，缺少最外层闭合括号，不能归因于8192上限不足。累计8次调用、1次预先允许的格式重试，输入15077、输出9474、合计24551 token；余下7例未运行。原始失败未修补。

### 结构化输出验证

根据DeepSeek官方Responses API文档，单独验证 /responses 的text.format=json_schema，保持deepseek-flash、low思考、8192上限和v6语义规则。一次V3-01兼容性探针通过，输入2134、输出1128、合计3262 token。单次成功没有被当作放行依据。

随后v7全套校准只执行2例：EQ-BOOL通过；EQ-TEXT返回completed却仍缺最外层闭合括号，严格解析失败，立即停止，无重试。累计输入4454、输出1084、合计5538 token。此次实际输出441 token，仍不是预算截断。说明当前实际返回不能满足我们对结构化评分的可靠性要求；不据此断言服务对所有用户均无效。

v7冻结plan沿用了两项v6说明字段（max_attempts_per_case=2及retry_only），与明确automatic_retries=0存在冗余冲突。冻结执行器实际每例只发一次，原始证据确认零重试；该plan不得用于正式发布，证据保持原样，后续新版本须清理字段。

新增Responses解析器拒绝异常状态、用量不一致、重复JSON键、非法/缺失维度和异常消息，不自动补括号。新增6项离线测试通过；本轮同时复跑旧裁判2项、来源审计6项、仅评分8项，合计22项通过。这些测试不代表缺依赖的全套Week8回归已通过。

### 后续门槛

暂不继续加token、循环改提示词或重试。下一阶段应单独评估真正受约束的输出通道（例如官方strict工具调用，需明确思考模式兼容性），或另一个裁判后端；必须先做有界兼容性测试，再通过统一的完整校准，之后才接入正式仅评分入口。当前仍可复用保存的20条模型回答，不需要重新进行GPU推理。

证据：reports/week8/phase2_judge_calibration_v6/verification.json、reports/week8/phase2_judge_schema_probe/verification.json、reports/week8/phase2_judge_calibration_v7/verification.json。
官方依据：https://api-docs.deepseek.com/zh-cn/api/create-response/

---

# Week8 裁判截断后的评分恢复准备

## 最新结论：low思考强度校准失败，未放行

`reports/week8/phase2_judge_calibration_v5`在调用前冻结8个案例：两道曾截断的题及原6个校准案例。相对8192/high协议，唯一请求变化为reasoning_effort=low，thinking仍enabled。官方说明medium映射到high，因此未采用无实际变化的medium：https://api-docs.deepseek.com/guides/thinking_mode/ 。

实际调用5次、重试0次：两个截断题分别使用2915、3310 completion token返回完整五维评分；V3-01、V3-02通过原预设检查。V3-03虽返回finish_reason=stop，但logic对象缺少闭合括号，JSON不可解析，严格拒收并停止；剩余3个案例未请求。本批输入8533、输出10621、总计19154 token，包含失败响应；独立重算与哈希回执见verification.json。末次余额显示¥9.54，不是精确本批账单。

不能把“截断题成功”称为新协议通过，也不能将格式失败算成模型能力0分。未修补JSON，未生成新版正式锁，未重跑20题，未开启GPU。high协议12份有效评分与所有失败证据继续保留。

### 评分口径复核发现

REASON-06题干要求合法单行JSON、固定字段、3项理由及uncertainty枚举，但未明确decision的类型或精确值。参考答案和自动检查则要求字符串insufficient_evidence。原基座输出decision:false，其语义对“能否宣布有效”的回答是否合理不能只靠字符串不匹配判定。low裁判在准确性和格式扣分时直接援引参考字符串，存在把参考表示形式当成显式要求的风险。因此后续需制定题干与参考冲突的统一判分原则，并加入对应校准对照；不能根据已经看到的某一模型答案临时改题或选分数。

### 后续步骤

先离线扩充校准对照（同义有效回答、实际误判回答、参考格式与显式题干要求的区分），保持冻结题目和模型回答。随后再评估统一提示词及输出策略。可考虑沿用已有的“只有已收到完整响应但JSON/结构不合格时，原请求最多再试一次”的有界策略；这是待评估方案，本次失败批次不恢复重试。身份、用量、网络状态不明或截断仍须停止。只有新方案全部通过预设检查，才考虑发布并对所有模型采用同一评分协议。当前不继续盲目提升token上限或启动整批付费运行。

以下为之前阶段记录。


## 最新结果：仅评分批次在REASON-06停止，12份有效评分已核验

正式评分入口已实现并通过33项相关测试，在线实际请求13次、重试0次。前12题原始响应、请求体和保存评分全部重新核验通过。第13题REASON-06返回finish_reason=length，completion_tokens=8192，最终content为空；程序按规则立即停止，7题尚未请求，没有生成完整summary或正式模型总分。

本次输入26,607、输出39,480，合计66,087 token（含失败请求），用量已从13份原始响应独立重算。末次余额显示¥9.64，仅为时点观察，不等于精确本批费用。MATH-05本次使用5833个输出token后成功，证明4096不足以容纳这次响应；但REASON-06再次截断说明8192的小批量校准不能推导为全量可靠。

证据位于`reports/week8/phase2_score_only_original_base/failure_review.json`，完整回答仍复用之前的20条；旧评分、校准试验与本批12份有效评分分别保存。失败状态及独占claim保留，禁止直接重跑或更换输出目录绕过防重复保护。全程没有启动GPU。

下一步应先分析REASON-06题目及裁判请求，比较降低推理强度、使用非思考模式等有界候选。不能直接以连续加大token上限代替可靠性验证。任何新协议都须统一校准并处理跨模型可比性，不能因为12题已有分数就把不同协议混入一个正式总分。

以下为此前实施过程记录。


## 当前进展：仅评分入口已放行并启动

新增`scripts/week8_score_only.py`，来源39文件、20回答、校准请求与原始响应、代码依赖、评分协议均在执行前复核。冻结运行清单位于`reports/week8/phase2_score_only_release/plan.json`，SHA256 `525f94fb4a473e9e120cf96da30e5a3cff518ba2d6cfe24794338ae986308a00`。本次仅放行原基座custom20新协议评分，不代表SFT/DPO运行锁更新。

新入口采用每题单次请求、不自动重试；问题目录写入付费标记后才发请求。完整有效响应先保存并fsync，再计算派生评分。同一回答+协议绑定唯一输出目录，换目录不可重付；并发受文件锁阻止。恢复前核验所有已有响应，缺失响应、篡改结果或付费记录丢失立即停止。已有成功结果可核对后复用，失败状态不可自动重跑。输出不覆盖历史回答或评分，GPU推理不重复。

共33项相关测试通过，包括新增8项执行/恢复/预算/防重复与真实放行清单检查。原完整环境缺依赖的限制仍存在。本次在线运行目录`reports/week8/phase2_score_only_original_base`，最多20次新请求，余额保留至少¥1、观察到本批余额下降达到¥1停止；余额结算可能延迟，因此次数和输出长度也设上限。最终成功仍须独立核验20份响应，不凭进程退出判断。

以下为之前阶段记录。


## 最新进展：8192-token小批量校准通过

2026-09-17已在本地直接调用DeepSeek完成截断题试验及6个原校准样本，无需AutoDL开机。7次请求全部首次返回严格有效JSON，原6案例预设检查全部通过；重复干扰案例各维差不超过1、加权总分差0.35，不超过预设0.5。

七份请求与历史请求逐项比较，唯一请求参数变化为max_tokens从4096改为8192。MATH-05本次输出3567 token，因此不能把单次成功直接归因于上限提高，也不能声称已消除截断风险。试验评分只作为校准证据，未并入正式模型总分。

独立复核已从所有原始响应重算评分、检查阈值、累计7次调用用量并保存文件哈希。输入11,508、输出12,514、合计24,022 token。开始与结束余额均显示¥9.76，可能存在延迟结算，不能视为免费或精确费用。相关25项离线回归测试再次通过；上一节记录的完整环境缺依赖限制仍然存在。

证据：`reports/week8/phase2_judge_calibration_v4/verification.json`，结论SMALL_BATCH_PASS。候选配置`judge_profile_candidate.json`明确formal_scoring_approved=false、formal_entrypoint_integrated=false。旧正式适配器仍固定4096，不能将该候选直接替换正式锁。下一步是接入独立的仅评分执行入口，绑定已核验的20条回答和新协议，验证防重复付费与失败恢复，再发布新的评分运行锁。GPU监控继续暂停。

以下是校准前的离线准备记录。


当前阶段：离线产物核验完成；8192-token候选协议尚未经过线上校准，尚不能用于正式比较。没有启动GPU、调用裁判API、删除claim或修改正式锁。

## 原因与可保留的成果

2026-09-17第二窗口中，MATH-05的裁判返回`finish_reason=length`，4096个completion token用尽，最终content为空。现有裁判适配器按已冻结规则拒绝截断响应且不自动重试，协调器正确终止。问题不在模型推理失败，而是裁判输出预算不足和缺少独立评分恢复入口。

DeepSeek官方接口说明将`length`解释为输出上限或上下文上限导致的截断；本次请求输出用量恰好达到4096且输入远小于上下文限制，支持输出预算耗尽的判断。参考：https://api-docs.deepseek.com/api/create-chat-completion/

新离线工具`scripts/week8_score_recovery_audit.py`检查完整文件清单、逐文件大小与SHA256、来源运行锁、题目与rubric绑定、20条回答的ID与顺序、所有已付费请求与原答案的一致性、原始响应重算的评分、逐请求及累计token用量。工具无网络调用入口，不读取密钥，不加载GPU，不修改历史产物；输出须在来源目录外，且拒绝覆盖已有文件。

真实产物结果：39份文件通过校验；原基座20条回答完整。MATH-01至04的4份有效评分与原始响应一致；MATH-05仅1次截断请求；15题尚未请求。MATH-03因JSON无效发生过第二次请求，故累计6次请求。总用量27,922 token，其中输入12,114、输出15,808。该用量不能直接当作实际账单金额。

## 下一阶段的明确方案

1. 先用保存的MATH-05答案进行一次8192-token候选协议验证。保留模型、思考强度、系统提示、五维rubric及权重，只提高输出上限。若仍失败，保存证据并停止，不无限增加上限或循环重试。
2. 在原校准样本上检查评分口径及完整输出，再发布独立新版本。一次MATH-05请求成功不能替代校准。记录全部实际请求、token和余额，并保留原预算保护。
3. 原基座直接复用20条GPU回答。为了不把不同裁判协议混入同一正式比较，新协议校准通过后，对三个模型统一采用8192-token协议；原基座20题按新协议评分，原4份旧评分仅作为历史与校准对照，不悄悄并入新结果。这会多出4题裁判调用，但不增加原基座GPU推理。
4. 建立仅评分的受控执行入口，以来源manifest、回答哈希、新协议哈希和独占运行记录绑定。每题持久化请求标记与原始响应，已有成功响应须验证后复用；未知是否付费的请求、输入变化或失败状态禁止自动重复。旧GPU续跑claim不得删除。
5. 原基座新评分完成并审核后，再放行SFT/DPO全量推理和同协议评分。不得把部分成绩当作三模型结论。

`reports/week8/phase2_eval_resume/score_recovery_audit.json`包含20份候选请求体及历史分类，明确`formal_scoring_ready=false`。它是离线准备材料，不是校准成功回执或正式运行授权锁。

## 验证与限制

新增6项测试通过，覆盖真实失败分类、文件篡改、额外文件、已付费候选答案变更、伪造评分和错误累计用量。相关裁判、恢复与集成共25项测试通过。原正式评测、裁判适配器及GPU续跑脚本SHA256保持不变。

系统Python尝试运行全套Week8测试共90项，结果为3项失败、24项错误，输出显示缺少OpenCompass、PyTorch或PyYAML，或因上述依赖导入失败使预期错误分支未执行。先前`/tmp/week8-oc-fix-20260915`环境已不存在。本次不宣称全套回归通过，需恢复完整依赖环境后另行验证相关完整流程。
