# Day 2：模型下载与原生推理

## 验收结论

| 要求 | 实际结果 | 状态 |
|---|---|---|
| 下载 Qwen2.5-1.5B 或 7B Instruct | 通过 ModelScope 下载 `Qwen/Qwen2.5-7B-Instruct`，15/15 文件完成，退出码为 0 | 通过 |
| 编写 Transformers 原生推理脚本 | 使用 `AutoTokenizer` 和 `AutoModelForCausalLM` 加载本地模型 | 通过 |
| 测试代码生成、逻辑推理、角色扮演 | 三组完整对话及统一 JSONL 均已保存 | 通过 |
| 测试 `apply_chat_template` | 已记录 True/False 两种模板文本、token IDs、tokens 和特殊 token IDs | 通过 |
| 提交模型下载截图、对话日志和推理脚本 | 所有文件均在本目录提供直接入口 | 通过 |

## 核心代码

- [原生推理脚本](source/scripts/inference.py)
- [Chat Template 检查脚本](source/scripts/inspect_chat_template.py)
- [模型完整性检查脚本](source/scripts/model_inventory.py)
- [结果结构验证脚本](source/scripts/validate_results.py)
- [三组实验 Prompt](source/prompts/day2_prompts.json)

推理脚本只加载一次模型，依次处理三组 Prompt；每组消息先经过 `apply_chat_template`，再编码、生成、截取新 token、解码并写入结果文件。

## 三组完整对话

- [代码生成](source/results/day2_code_generation.txt)
- [逻辑推理](source/results/day2_logic_reasoning.txt)
- [角色扮演](source/results/day2_role_play.txt)
- [三组统一 JSONL](source/results/day2_inference_results.jsonl)

## 模型下载与完整性

- [模型下载截图](evidence/00_model_download.jpg)
- [模型清单截图](evidence/02_model_inventory.png)
- [完整下载日志](source/results/day2_model_download.txt)
- [模型文件清单](source/results/day2_model_inventory.txt)

下载结果显示模型名称为 `Qwen/Qwen2.5-7B-Instruct`，共 15 个文件；四个 Safetensors 权重分片均存在，模型配置为 28 层、`hidden_size=3584`。

## Chat Template 结果

- [Chat Template 完整日志](source/results/day2_chat_template.txt)
- [Chat Template 汇总截图](evidence/06_chat_template_summary.png)

`add_generation_prompt=False` 得到 23 个 token，序列结束在 user 回合；设为 `True` 后得到 26 个 token，额外加入 `[151644, 77091, 198]`，对应 `<|im_start|>assistant\n`。这个前缀明确告诉模型从 assistant 回合开始生成。

## 推理结果摘要

| 领域 | 输入 token | 输出 token | 耗时 | 峰值显存 | 复核结论 |
|---|---:|---:|---:|---:|---|
| 代码生成 | 85 | 644 | 15.205s | 14.244GiB | 代码语法通过；模型生成的测试存在错误断言 |
| 逻辑推理 | 93 | 405 | 9.346s | 14.229GiB | 与独立枚举的唯一解一致 |
| 角色扮演 | 66 | 512 | 11.756s | 14.234GiB | 覆盖环境、输出、性能、复现和审查 |

代码回答中的测试失败属于实验结果，原始回答保持不变。它说明代码生成结果必须经过真实执行，不能只根据文本形式判断正确性。

## 截图与报告

- [代码生成截图](evidence/03_code_generation.png)
- [逻辑推理截图](evidence/04_logic_reasoning.png)
- [角色扮演截图](evidence/05_role_play.png)
- [最终验证截图](evidence/07_final_validation.png)
- [Day 2 实验报告](REPORT.md)

推理流程和 Chat Template 原理见 [Day 1 与 Day 2 知识笔记](../notes/day1_day2_knowledge.md)。
