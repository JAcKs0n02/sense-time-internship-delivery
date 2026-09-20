## 2026-09-17 选型更新：质量优先，候选升级为Gemini 3.1 Pro Preview

用户明确提出使用更先进模型以减少返工。撤销Gemini 2.5 Pro作为下一轮默认候选，改为gemini-3.1-pro-preview；保持Google AI Studio接入与现有Key配置工具，无需再换账户平台。官方模型页列明支持思考与结构化输出。该型号为Preview，不能声称版本不可变或保证零返工；执行时记录返回模型版本、完整请求和时间，并统一三模型评分协议。

本次改选不是校准通过。此前失败直接涉及输出结构，不能全部归因于模型能力，也不能靠升级型号跳过校准。采用预先冻结的2例兼容性测试、14例完整校准与独立审核，不按成绩反复修改口径。思考预算以评分质量为先、按接口能力设定并记录，不延续为了缩短输出而固定low的假设；具体参数在调用前确定。

官方说明该模型API无免费层，需要账户具备付费API访问条件。尚未开通账单、充值或调用Gemini。Key仍需用户本地配置；原脚本与保存位置保持适用。

来源：https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview
https://ai.google.dev/gemini-api/docs/gemini-3

以下为历史选型记录，不代表当前默认模型。

---

# Week8 Gemini 本地接入

当前：用户继续推荐的Gemini候选路线。本地配置工具已完成，4项凭据相关测试通过；2026-09-17本地检查确认gemini.key尚不存在，尚未调用Gemini API。正式评分仍未放行。

1. 打开 https://aistudio.google.com/apikey ，登录Google账户，创建用于本项目的新API Key。实际项目资格、可用模型与额度以账户界面为准。暂不因配置步骤自行开通付费账单。
2. 在交互式终端运行：

```sh
python3 "/Users/yifanren/Documents/商汤科技实习/scripts/setup_week8_gemini_key.py"
```

3. 看到提示后只粘贴Key本身，不加引号，按回车。输入不显示是正常现象。看到“Gemini Key 已保存到仓库外的私有文件”才表示保存完成。
4. 可再次检查：

```sh
python3 "/Users/yifanren/Documents/商汤科技实习/scripts/setup_week8_gemini_key.py" --check
```

本地检查仅核对文件权限和非空/无空白格式，不代表API可用或付费额度有效。文件在~/.config/internship-week8/gemini.key，权限0600，不覆盖已有Key，与deepseek.key分开保存。

如果终端原本显示quote>或dquote>，先按Ctrl+C退出未完成命令，再运行上面的完整命令。不要在quote提示下继续粘贴。

配置完成后，进行模型可用性与输出协议检查，然后按docs/week8_judge_backend_selection.md规定的2例探针、14例完整校准推进。新协议的预算与参数需冻结；失败不自动重试。无需启动AutoDL，也无需重新生成20条模型回答。

官方说明：https://ai.google.dev/gemini-api/docs/api-key
