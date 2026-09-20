## 2026-09-17 选型更新：质量优先，候选升级为Gemini 3.1 Pro Preview

用户明确提出使用更先进模型以减少返工。撤销Gemini 2.5 Pro作为下一轮默认候选，改为gemini-3.1-pro-preview；保持Google AI Studio接入与现有Key配置工具，无需再换账户平台。官方模型页列明支持思考与结构化输出。该型号为Preview，不能声称版本不可变或保证零返工；执行时记录返回模型版本、完整请求和时间，并统一三模型评分协议。

本次改选不是校准通过。此前失败直接涉及输出结构，不能全部归因于模型能力，也不能靠升级型号跳过校准。采用预先冻结的2例兼容性测试、14例完整校准与独立审核，不按成绩反复修改口径。思考预算以评分质量为先、按接口能力设定并记录，不延续为了缩短输出而固定low的假设；具体参数在调用前确定。

官方说明该模型API无免费层，需要账户具备付费API访问条件。尚未开通账单、充值或调用Gemini。Key仍需用户本地配置；原脚本与保存位置保持适用。

来源：https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview
https://ai.google.dev/gemini-api/docs/gemini-3

以下为历史选型记录，不代表当前默认模型。

---

# Week8 替代裁判候选与接入门槛（2026-09-17）

## 当前结论

DeepSeek Flash 的文本JSON、Responses JSON Schema、Beta strict工具路径都未通过完整校准。本轮进一步保持v8请求的所有参数与题目不变，仅将模型改为deepseek-v4-pro（官方当前标注DeepSeek-V4-Pro-0813）。STRICT-BOOL独立探针通过；v9完整校准EQ-BOOL通过，EQ-TEXT返回合法JSON但多了一层arguments包装，违反顶层恰好五维的schema，立即停止。没有解包修复，也没有按分数重试。

本轮3次调用，零重试，输入7432、输出2745、共10177 token。v9 manifest、请求身份、原始响应、保存结果和累计用量已独立复核。8项现有严格工具/裁判加固离线测试通过，另验证Pro解析器拒绝Flash身份、请求仅模型字段变化；不是全套Week8回归。没有启动GPU或修改正式锁。

这说明同服务换模型还未解决我们的输出可靠性问题，不能推出Pro总体能力差或所有调用都会失败。官方模型名为别名，不是不可变版本锁；须记录文档版本与实际响应model。

## 独立服务候选

建议下一候选使用Google Gemini 2.5 Pro（gemini-2.5-pro），而非继续切换同一服务的包装接口。官方模型页列明支持thinking和structured outputs；它是可审计的明确候选，不宣称已经通过我们的评分校准，也不宣称是目前最强模型。Google结构化输出文档也明确：语法符合schema不代表评分语义正确，因此仍须跑原有检查。

Google AI Studio需要独立账户/项目和API凭据，现有DeepSeek Key不能使用。已询问用户是否可使用该服务，尚未创建账户、开启账单或发出API请求。也未搜索其他项目中的私人凭据。用户若已有其他服务，应按其已有接入条件另行选择。

## 确定服务后的实施顺序

1. 用户在https://aistudio.google.com/apikey创建新Key并在本机配置。使用隐式输入的配置工具，避免手写带引号的shell命令；不把Key写进仓库或运行证据。实际配置工具在用户选定服务后提供。
2. 核对模型可用性、API格式、输出schema和token用量定义。实现独立适配器，拒绝截断、缺字段、模型身份不符、重复键、非法分值与空理由；不补括号或自动解包。
3. 冻结请求和预设阈值，先用STRICT-BOOL与EQ-TEXT两个失败类型做兼容性探针，各一次；失败立即停止。模型参数需按该服务能力明确记录，不能把不同提供方的思考参数名称当成相等预算。
4. 探针通过后跑原14例与配对检查，各一次、不自动重试。保持v6评分语义、题目、参考与候选答案、五维权重不变。任何阈值调整须独立说明问题证据，不能为让新模型通过而改答案标准。
5. 校准全通过且独立审核后，再接入正式仅评分入口，保留防重复付费、来源哈希、原始响应持久化、预算和失败锁。此时才重新统一评分原基座20条保存答案，并安排SFT/DPO；不能混合不同裁判协议的成绩。

预计上限为2个探针+14例校准。具体输出/推理预算、每批费用门槛在实现前按API语义冻结；不自动启用搜索、工具执行或批量重试。官方列出的2.5 Pro付费价（短于200k输入）为每百万输入$1.25、输出（含思考）$10；以请求和token双上限控制实际试验，价格执行前复查。是否使用免费层还取决于项目资格、限额与数据政策，不默认它免费或可用。

## 证据与官方来源

- reports/week8/phase2_judge_pro_probe/verification.json
- reports/week8/phase2_judge_calibration_v9/verification.json
- reports/week8/phase2_judge_calibration_v9/turn_summary.json
- https://api-docs.deepseek.com/quick_start/pricing/
- https://ai.google.dev/gemini-api/docs/models/gemini-2.5-pro
- https://ai.google.dev/gemini-api/docs/generate-content/structured-output
- https://ai.google.dev/gemini-api/docs/api-key
- https://ai.google.dev/gemini-api/docs/pricing
