# Week8裁判接入：本地密钥配置

用户已确认DeepSeek裁判方案，并已充值及创建Key。密钥读取及真实接口调用已验证；6条独立校准的预设检查通过，但评分维度一致性仍需整改复核。尚未接入正式评分入口或解除运行锁。详见 week8_judge_calibration.md。

在本机终端运行：

```bash
python3 '/Users/yifanren/Documents/商汤科技实习/scripts/setup_week8_judge_key.py'
```

根据提示粘贴Key并回车，输入不会显示。工具将其保存在`~/.config/internship-week8/deepseek.key`，文件权限0600，位于仓库外；已有文件不覆盖。脚本不打印密钥、不调用API。正常完成后会显示保存成功，告诉助手“已配置”即可，无需发送Key。

独立适配器`week8_deepseek_judge.py`固定官方地址和deepseek-flash请求名称，思考high、JSON输出、max_tokens=4096，无自动重试。校验返回模型标识、finish_reason、非空JSON、五维评分、理由及token用量。它不会在导入时联网，尚未修改正式step3_eval.py的调用路径。

后续须先核验真实服务标识，再做少量独立案例校准；如服务返回名称与预期不同，先记录并查明映射，不自动接受未知模型。校准通过才能纳入运行锁。用户授权范围是此前确认的裁判方案和10元预算，不调用全部benchmark或启动GPU。
