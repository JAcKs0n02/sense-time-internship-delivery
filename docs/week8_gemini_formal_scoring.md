# Gemini 原基座正式仅评分入口

本入口复用window2_failure中已核验的20条回答。来源manifest：22665b1f4040f239aeb258367a9a1552520f5399ce2fb8b827b3747aaa878a22。评分协议来源phase2_gemini_calibration_v2，17例全通过且独立复核。模型gemini-3.1-pro-preview，high思考、16384输出上限、JSON Schema。输入题目、rubric与候选答案均绑定来源，不复用DeepSeek旧分数。

执行器scripts/week8_gemini_score_only.py保留独占文件锁、答案+协议身份绑定唯一输出目录、请求前持久化付费标记、原始响应先落盘、已有结果全量核对后恢复、未知响应及失败禁止自动重发。20题各一次，无自动重试。生成错误或评分解析失败立即停止。与旧DeepSeek入口分文件，避免修改已冻结历史协议。

预算：每请求JSON最多32768字节，输出上限16384，最多20次。按每个请求字节都计入输入token的保守规划估计，以输入$2/M、输出含思考$12/M估算本批约$4.412344，低于执行器$6规划门槛。此为请求规模保护，不是服务方账单硬上限；实际价格与账单由服务方决定。不启用搜索、外部工具或重复GPU推理。

8项执行器测试通过，覆盖完成重跑、截断、未知传输失败、预算拒绝、原始结果恢复、伪造结果、付费证据删除和真实来源/校准计划生成；另2项Gemini解析器测试通过。不是完整Week8环境回归。

放行计划reports/week8/phase2_gemini_score_release/plan.json，SHA256：3e9f731f159775f3fd3a663b4251e654f86508b827484d7b4deef369ab7a5458。在线结果reports/week8/phase2_gemini_score_original_base；最终以verification.json独立复核为准。

正式输出标记ai_judge_not_human，不冒充两个人类评分员。只有原基座20题完整复核后才报告该模型AI裁判均分；SFT/DPO尚未评分，不能作三模型排序。MATH-05题干仍有断句歧义，报告应保留解释限制。

## 已完成结果

20/20通过独立复核，均分3.9775/5，20次调用、零重试，总用量79117 token。verification.json状态VERIFIED_COMPLETE。SFT/DPO尚待执行，原基座成果可直接复用。
