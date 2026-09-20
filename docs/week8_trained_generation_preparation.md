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

# SFT/DPO 推理与裁判分离准备

2026-09-17：原基座已完成，不再进入新GPU运行计划。旧step3_eval.fresh及旧协调器会在生成后调用DeepSeek，且旧窗口最晚启动15:00 UTC已过，因此未直接开机续跑。

新增scripts/week8_generation_only.py，仅允许final_sft/final_dpo。它保留旧已发布运行锁作为模型、基准数据、OpenCompass配置和依赖身份的来源证据；不改写旧锁中的历史DeepSeek字段，也不执行旧裁判调用。新计划明确generation_only、judge_calls=0，新的Gemini协议和原基座已完成回执另行哈希绑定。

本地检查已核对两模型旧锁及全部本地非权重依赖。远端权重需要在320上按原model_identity清单逐文件大小/SHA核验，本次未声称已完成。生成设置与基座保持一致：BF16、seed42、贪心解码、max_new_tokens512、相同chat template和题集；基准采用原OpenCompass配置与全部119科目。每模型完成后保存benchmark结果和20条回答，裁判在本地进行。

3项测试通过：真实本地证据核验、错误模型/运行锁/裁判调用模式拒绝、依赖篡改拒绝。计划保持CANDIDATE_TARGET_AUDIT_REQUIRED，执行器拒绝候选计划启动推理。未进行CUDA导入测试或远端运行测试，不宣称目标环境通过。

待完成门槛：
1. 部署新脚本和计划至320，在CPU检查阶段核验模型文件与完整运行环境。
2. 为两模型安排新的充足GPU窗口、协调器进程组超时清理与平台自动关机保护；旧15:00最晚启动/20:30截止计划不再可直接使用。生成脚本自身的benchmark timeout不替代完整会话保护。
3. 目标审核通过后生成独立已发布计划与回执，不仅手改release_state。按SFT→DPO顺序执行，失败保留且不自动重试。
4. 下载并核验回答/基准manifest，扩展Gemini仅评分入口对这两份新来源的验证（现入口prepare仍专用于原基座window2_failure），再以同一评分协议各评20题。

内嵌浏览器页面本次显示320已关机、GPU充足；这不等于已预留资源。未操作344、未启动GPU、未改旧监控。

证据：reports/week8/phase2_trained_generation/plan.json、local_review.json。实现文件：scripts/week8_generation_only.py，测试：tests/test_week8_generation_only.py。
