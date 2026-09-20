# 配置来源说明

`config.json` 于 2026-07-17 从 Qwen 官方模型仓库读取：

- 模型：`Qwen/Qwen2.5-7B-Instruct`
- 地址：<https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json>
- 原始文件大小：663 字节
- SHA-256：`7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c`

读取后对官方文件和本地副本分别计算 SHA-256，两者一致。`day3_config_pretty.json` 是使用 `python -m json.tool` 生成的格式化副本，字段和值未改变。

Day 2 已确认实际使用的模型为同一官方模型。2026-07-17 的 Day 3 实际模型验证直接读取 AutoDL 模型目录中的 `config.json`，其 SHA-256 同样为 `7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c`。因此，官方文件、本地副本和服务器实际模型配置三者一致。

参数量同时采用配置公式和实际权重逐模块统计，两种结果完全一致；关键张量形状也与配置推导一致。原始记录见 [`../results/day3_weight_parameter_verification.txt`](../results/day3_weight_parameter_verification.txt)。
