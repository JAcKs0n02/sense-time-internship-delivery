# Week8 custom20裁判建议与运行锁准备

2026-09-15。用户尚未选择裁判，已要求协助确定方案。本文为推荐方案，未将其写成已批准的裁判身份，未调用API或充值，也未放行训练。

## 推荐方案

建议使用DeepSeek官方API，模型名称`deepseek-flash`，base URL为`https://api.deepseek.com`。官方当前说明此名称对应DeepSeek-V4.1-Flash。选用理由是可直接适配现有OpenAI兼容HTTP入口，支持思考模式与JSON输出，且无需重新占用320机部署裁判。是否满足本项目评分要求仍须通过独立校准验证，不能从产品宣传推定裁判准确率。

备选为Kimi官方当前可用的`kimi-k2.6`；本轮未完成其参数与计价核验，暂不作为执行配置。Kimi官方列表已将K2.5等旧型号列为下线，不沿用旧教程的名称。

资料：
- https://api-docs.deepseek.com/ ：接口格式、base URL及当前模型名称映射。
- https://api-docs.deepseek.com/guides/thinking_mode/ ：默认开启思考，支持high强度；思考模式下temperature参数被忽略。
- https://api-docs.deepseek.com/zh-cn/guides/json_mode/ ：JSON模式、格式样例、输出预算与空content限制。
- https://api-docs.deepseek.com/quick_start/pricing/ ：本次搜索索引显示Flash高峰每百万输入（未命中缓存）$0.30、输出$1.20，非高峰减半。直接页面抓取超时，报价仍应在首次调用前核验控制台。
- https://platform.kimi.com/docs/models ：当前与下线模型列表。

DeepSeek官方首页与定价页搜索索引对V4 Pro近期是否路由到Flash的说明不同，故不把`deepseek-v4-pro`名称当作不可变版本依据。云端`deepseek-flash`也不是固定权重快照：须记录实际response.model、时间、可用的服务端指纹及官方映射；如果无法确认精确权重版本，按冻结协议报告这一限制，不声称完全可复现。

## 评分与成本

只评固定custom20：原始基座、最终SFT、最终DPO各20份，共60份回答。C-Eval与CMMLU仍按标准答案程序计分，不发送12928题给裁判。

同一裁判、同一提示和参数评全部模型，请求不包含候选模型名称；沿用五维0–5评分与30/25/20/15/10权重。先用6条独立构造的明显正确/错误、格式与指令遵循案例检查接口与评分方向，不使用正式题目的表现挑裁判。该校准只能发现明显故障，不能证明裁判无偏。

拟用思考high、JSON对象输出，单次总输出预算初设4096 token。现有评分代码只设置temperature=0，未设置思考开关和输出上限，也未按供应商检查finish_reason及返回模型标识；因此不能直接投入调用，需先适配并离线测试。预算不足导致截断或空JSON应报告失败，不能把失败题剔除后计算平均分。

费用是规划估算：若每份输入4000、计费输出4000 token，60份为输入/输出各24万token；按本次可见Flash报价约$0.18–$0.36，另加校准或重试。实际中文token量和思考输出量须读usage计量，此估算不是账单承诺。建议整个裁判验证及正式评分设置¥10预算上限，尚未获得支出授权，不据此充值。

## 仓库已完成的准备

`reports/week8/phase2_release_review/release_review.json`汇总318个去重文件的哈希复核、目标提示与forward证据、20题及rubric身份。38项测试重新通过。汇总时首次使用了不匹配的索引字段，检查脚本报KeyError；改为读取target_runtime.receipt后重跑通过，第一次没有写出通过文件。

后续顺序：确定官方平台账号及本地凭据配置方式；适配裁判请求与响应校验；做少量独立校准并保存实际服务身份；补齐最终运行锁。API密钥只放本地安全配置/环境变量，不发到聊天、不入库。当前320机无需开机，训练仍保持关闭。
