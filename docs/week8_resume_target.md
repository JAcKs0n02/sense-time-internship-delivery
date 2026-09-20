## 2026-09-18 Day42首轮准备完成

已核对历史教师来源，选出200条训练候选提示，并修复蒸馏数据入口和上下文保留；2项针对性测试通过。尚未GPU放行，学生身份、CEval比较及训练证据检查仍需加固。详情见week8_distillation_preflight.md。旧推理会话和GPU监控保持关闭。

---

## 2026-09-18 人工评分已完成并整合

两份各60条匿名评分已核验，评分日期2026-09-18；不收集姓名，不要求额外过程声明。人工均分原基座3.98250、SFT3.64875、DPO3.62875。已同步综合报告、雷达图与结果CSV，原始分数不改动。下一实验任务为Day42蒸馏入口与资产预检，随后单独安排新的GPU实验；已完成推理会话和旧监控保持关闭。

---

> 2026-09-18人工评分更新：已收到并核验两份各60条填写表，人工均分原基座3.98250、SFT 3.64875、DPO 3.62875。文件汇总完成，匿名代号保留，评分日期2026-09-18，归档完成。详见[人工评分结果](week8_human_scoring_result.md)。旧“人工待填写”说明为此前快照。

## 2026-09-18：报告整合及逐项验收整理

当前工作为本地报告与提交索引整理，未启动GPU、未调用付费API。正式训练/评估已完成，人工评分待用户安排；蒸馏、部署联调、统一CUDA复现及最终发布仍是独立缺口。最新清单docs/week8_acceptance_audit.md；提交入口Submission/Week8/README.md。旧监控保持暂停，不重启已完成会话。

---

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

## 2026-09-17 19:38 UTC：SFT 生成结束并完成远端只读复核，DPO 正在推理

SFT 已逐科重新核对冻结标准答案，119个学科覆盖完整，保存的基准汇总与重算一致：CEval 1072/1346（79.6434%），CMMLU 9260/11582（79.9516%）。custom20 的20条回答唯一、非空、题号顺序与冻结题集一致，回答SHA与状态回执一致；裁判调用为0。此次为远端只读审核，完整原始产物尚待下载并核验SHA，不能宣称本地回收验收或Gemini评分已完成。

协调器PID1593继续运行，19:37 UTC观测DPO已有15份学科预测，GPU利用率91%。没有重复派发或重跑。原程序截止03:10 UTC与平台关机03:15 UTC保持不变。审核记录：reports/week8/phase2_trained_generation/sft_remote_review.json。

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

# Week8 受控续跑目标部署与资源等待

最新补充：low思考强度候选完成5次请求后因V3-03非法JSON停止，校准失败未放行；另发现REASON-06题干与参考格式约束不一致，需统一判分原则。见`docs/week8_score_recovery.md`及`reports/week8/phase2_judge_calibration_v5/verification.json`。没有恢复GPU或全量评分。


当前最新状态：仅评分入口已完成并执行，13次请求获得12份有效评分；REASON-06耗尽8192-token上限后停止，无自动重试。完整评分尚未完成，GPU未开启、原监控保持暂停。证据见`reports/week8/phase2_score_only_original_base/failure_review.json`，后续方案见`docs/week8_score_recovery.md`。


最新补充：8192-token截断题试验及6个原校准样本已完成，7/7有效响应、预设检查通过，独立审核SMALL_BATCH_PASS。详情见`docs/week8_score_recovery.md`及`reports/week8/phase2_judge_calibration_v4/verification.json`。正式评分入口尚待接入，GPU监控仍暂停，不使用旧入口重跑。


最新补充：评分恢复离线审核已完成，39份来源产物、20条回答、4份有效评分及6次请求用量均通过核对。8192-token仅为待校准候选，正式运行仍暂停；完整下一阶段方案见`docs/week8_score_recovery.md`。


## 当前状态：第二窗口失败已回收，GPU已关机

2026-09-17 11:14:05 UTC，原基座custom20裁判在MATH-05失败，协调器停止，没有启动SFT/DPO。20条回答完整生成；前4题取得有效评分，累计6次裁判请求。MATH-05实际只有1次请求，返回finish_reason=length、completion_tokens=4096、最终content为空，仅有推理内容，因此评分JSON校验失败。不能将缺失评分作为0分或报告完整基座成绩。

39份产物已下载至`reports/week8/phase2_eval_resume/window2_failure`并逐项校验大小与SHA256；压缩包SHA256为`35c267cc94a2284c86a2dbbc3f4c691b7da73ec64a9061cc059c93196ae0a185`。session_exit确认preflight恢复，相关进程已退出，GPU无计算进程。11:18 UTC后页面核对320已关机、定时已清除，监控已暂停。未自动重试，也未删除一次性claim。

下一步应先在本地分析裁判token上限与拒收策略，设计保留既有20回答和4份有效评分的受控评分恢复；若调整裁判协议，须说明跨模型可比性并重新审核，不能直接修改锁或绕过claim重新执行原流程。故障审核见`window2_failure_review.json`。以下为历史运行记录。


## 最新状态：第二窗口已启动（2026-09-17 11:09 UTC）

320机出现空闲GPU后已启动。实际报价¥1.48/小时、开机前余额¥18.33，平台关机保护已核对为20:35 UTC（页面21:35）。单张3090、BF16、无其他计算进程、磁盘20.06 GiB及输出/claim不存在均通过；新版协调器已在目标生成并核对哈希与语法。11:09:46 UTC协调器PID1566启动，原基座子进程PID1585，当前RUNNING/original_base；随后核验已生成4/20条新回答，GPU显存14916 MiB、利用率95%，进程存活；裁判评分尚未开始。详情见window2_launch_observation.json。禁止重复启动；后续监控仅检查该会话，失败不自动重试。


## 当前状态：第二窗口等待GPU（2026-09-17 11:01 UTC）

用户要求继续，已恢复同一监控，每5分钟检查320机。11:00 UTC实际查询仍为空闲0卡、实例已关机。第二窗口最晚15:00 UTC（北京时间23:00）启动，进程截止20:30 UTC，平台关机20:35 UTC。运行窗口届满仍未开始则暂停，不自动延长。

新版 `reports/week8/phase2_eval_resume/run_resume_session_window2.py` 仅将旧协调器截止时间从08:30改为20:30；逐字节核对其余内容完全一致，语法编译通过，SHA256为 `308dabb2fd65c66f49b3daca8e51f4591b90090a384769e36eb9c5c2d15bd4b9`。本地审核见 `window2_review.json`，**目标端尚未部署新版协调器**，等待GPU可用后先部署并核对哈希，再执行原有全部启动检查。旧协调器、模型锁和target_release审核回执保留不变；新旧窗口共用同一输出与来源claim，禁止重复启动。

资源可用后重查实际价格、余额、剩余运行时间及自动关机保护；不为时间调整单独开启CPU实例。以下是第一窗口历史记录，不作为当前启动时间依据。


## 最新状态：本次窗口已结束，未启动续跑

2026-09-17 03:01 UTC（北京时间11:01）复核320机仍为已关机，页面显示“设置定时关机”，无残留定时。最近一次02:55 UTC资源查询为空闲0卡。已超过03:00 UTC最晚初始启动时间，`week8-320`监控已暂停；本次没有启动GPU续跑，也未产生本轮裁判调用。目标部署和审核回执仍保留。后续需要重新安排足够长的运行窗口，并更新协调器时间限制及相应审核记录后再启动。以下为本次窗口的部署与保护记录。

2026-09-17 00:14 UTC：320机`be044ebe99-be706b14`已完成无卡部署与审核，当前已关机、定时已清除。尝试查询GPU开机返回主机空闲0卡，因此尚未启动正式续跑。沿用`week8-320`每5分钟监控，GPU可用即按以下已核验配置执行。

## 目标核验结果

部署包`reports/week8/phase2_eval_resume/resume-deploy-20260917.tar.gz`，386份文件，SHA256 `e8dfb7a42bd02515ef7b9edd0336ede9a65858f22f19673d1f604cef695c15af`。目标暂存`/root/autodl-tmp/week8-resume-stage-20260917`，实际runtime `/root/autodl-tmp/week8-judge-runtime-20260916`。旧脚本保存在新审核目录`step3_eval_before.py`，旧锁和历史结果未覆盖。

目标审核重新通过正式锁入口：原基座324项依赖，SFT/DPO各322项；原基座374份来源产物哈希与119学科12,928题全部通过。没有GPU推理或裁判API调用。全局preflight在审核后逐字节恢复。下载回执包SHA256 `9e9443fee85ffca50c6a45ee9c039c0e0d07684172749955e8c98a0d9167560f`，本地核对全部新版锁与候选依赖变化一致。

本地`reports/week8/phase2_eval_resume/target_review.json`记录独立复核。`target_release.json` SHA256为`b4c64a370110cf6b10e124341598e3a2c96acb337b4a909e78353c5fae680ca2`。新版放行锁位于runtime的`reports/week8/phase2_eval_resume/`：

| 模型 | 锁文件 | SHA256 |
|---|---|---|
| 原基座 | original_base_runtime_lock.json | ce2adde9027af06c96dad05b668db954e1ef3fcf27b52affd7c6371627b4aee3 |
| SFT | final_sft_runtime_lock.json | 4f65d0757e777cbdce37c16154175d3be2d9dafe221bd2d5031cb243d3488487 |
| DPO | final_dpo_runtime_lock.json | ae261518099604cdf91b5633d5cc432ebad3fc9a46a2495ddf0a9c7eaf5c8fa5 |

## 待执行与保护

协调器为runtime下`reports/week8/phase2_eval_resume/run_resume_session.py`，SHA256 `3d5d73f09c502e9fe7c07d0dbec80d875d521291044606c06947d99cb97c2947`。启动时须将上述target_release SHA作为唯一位置参数，Python使用`/root/autodl-tmp/conda/envs/llm_exp/bin/python`。新输出`logs/formal-resume-20260917-01`，协调日志同级`formal-resume-20260917-01-coordinator.log`。

执行顺序：原基座只补custom20，再SFT和DPO各自全量评测。原基座客观题来源明确记录为复用。GPU启动前核对输出和一次性claim均不存在，禁止重复启动；不删除claim绕过失败保护。

当前窗口要求03:00 UTC前启动，进程截止08:30 UTC，平台关机保护08:35 UTC（页面若保持UTC+1为09:35）。不允许自动延长。若03:00仍无GPU，监控暂停并报告，避免启动后剩余时间不足。历史GPU报价¥1.48/小时，本次无卡页面余额¥18.34；实际GPU开机前重新核对报价、余额和关机保护。

只等待320，不克隆、不迁移、不操作344。等待期间保持关机；正常无资源时安静，仅在启动、完成、异常或需用户操作时通知。完成或失败后均回收核验产物，关闭实例并暂停监控。
