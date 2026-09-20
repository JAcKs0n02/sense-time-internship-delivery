# Week8：OpenCompass 数学文本保真修复

2026-09-15。本次范围是提示模板修复、真实框架全题提示检查及配置保存/重载；不包含模型推理或训练。**本地实际框架提示验收通过**：119 科、12,928 道题全部与独立预期逐题一致，五示例全部保留，最长输入 1,361 token，截断差异 0，配置重载通过，仓库 35 项测试通过。最终运行收据见 `reports/week8/phase2_prompt_fix/runtime-03/verification.json`。

## 修复内容与审核方法

官方 OpenCompass 0.5.3 的 `safe_format` 按字段顺序替换，会将已插入题目中的 `{A}`、`{B}`、`{C}` 当作新的占位符。新增 `scripts/week8_prompt_template.py`，注册专用 `NonRecursivePromptTemplate`：只扫描原模板一次；插入的数学文本保持字面值；已经渲染的五条示例在原始 begin 标记处插入，不再参与待答题的格式化。保留官方对空 SYSTEM 的换行处理。

适配范围明确限定为本周采用的 meta round 模板及独立 begin 示例标记；不支持旧式字符串/按标签模板及 sep_token。安装包源码、原始题目、独立预期、第一阶段归档均未因本次修复而修改。

测试先复现失败，再修复：11 道真实问题记录全部纳入回归；另检查示例二次格式化、输入字典不变、未知占位符、重复占位符、反斜杠、插入文本中的示例标记，以及待答题答案字段清空。原版失败日志 `regression_before.log`，示例插入边界失败日志 `ice_regression_before.log`，最终测试日志 `regression_after.log` 和 `all_tests.log` 均保留。

## 真实框架核验

脚本 `scripts/audit_week8_opencompass_runtime.py` 实际调用 OpenCompass 0.5.3、MMEngine Config.fromfile、注册数据集加载、FixKRetriever、注册模板、GenInferencer、HuggingFacewithChatTemplate。采用 tokenizer_only 模式，不加载权重。

每一道计分题都检查：11 条消息的角色顺序（五组示例加一道用户题）；真实 chat template token 的长度与独立 v2 预期哈希一致；batch_encode_plus 开启截断参数后 token 不变；输入加 32 个输出 token 不超过 2048。配置落盘后重新用 Config.fromfile 加载，并逐科通过注册表构造新模板。测试中没有调用替身模拟框架。

运行尝试按顺序保留，不覆盖失败事实：

1. runtime-01 被过严的“示例必须分离”检查拦截，发现框架实际通过 generate_item 传入示例；补充边界测试后修复。
2. runtime-02 全部题目的提示比较通过，但最终配置保存遇到 Python 类对象无法序列化；此次仍算整体验收未通过。
3. runtime-03 将类型保存为完整可导入类名，再进行全量检查及配置重载；退出码 0，最终状态为 ACTUAL_OC_PROMPT_RUNTIME_PASS_NO_INFERENCE。

运行环境为新增 `/tmp/week8-oc-fix-20260915`，通过 `.pth` 只读复用已验证 CPU 环境的依赖，在新环境安装 OpenCompass 及额外依赖。原 `/tmp/internship-week8-loader-20260915` 未安装或升级包。依赖清单见 `environment.txt`。这次是 macOS CPU 上的真实框架验证，尚不能称为 320 Linux 上修复后的复验。

复现命令（仓库根目录；输出目录必须未存在）：

```bash
/tmp/week8-oc-fix-20260915/bin/python -m unittest discover -s tests
/tmp/week8-oc-fix-20260915/bin/python scripts/audit_week8_opencompass_runtime.py --output reports/week8/phase2_prompt_fix/runtime-new
```

## 后续关卡

生成的 expanded_audit_config.py 是 tokenizer_only 审核配置，不是正式模型评测配置。下一项是将专用模板与固定数据身份纳入正式评测配置，核验真实推理参数及入口行为，再补齐裁判模型准确名称、版本和接口地址。修复后的 320 环境复验与模型 forward/短步 smoke 仍待完成；评测运行锁和训练锁继续关闭。此次没有启动 GPU、训练或付费裁判调用，也没有新模型分数。
