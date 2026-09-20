# 第 1 周最终提交整理设计

## 目标

在不移动、不删除现有仓库文件的前提下，新建一个可直接交给导师的第 1 周提交目录。提交包只保留老师明确要求的报告、代码、日志、Notebook 和截图，不带入归档、失败重试、缓存或内部验证材料。

## 提交目录

目标路径：

`第1周_最终提交_2026-07-24/`

目录内共 19 个文件：

```text
提交说明.md
Day1_环境初始化/
  day1_environment.png
  day1_toolchain.png
Day2_模型下载与原生推理/
  model_download.jpg
  inference.py
  code_generation.txt
  logic_reasoning.txt
  role_play.txt
  chat_template.txt
Day3_架构分析/
  Qwen2.5架构分析报告.md
  config.json
  analyze_params.py
  parameter_counts.csv
Day4_Tokenizer实验/
  tokenizer_experiments.executed.ipynb
Day5_LLaMA-Factory与周报/
  第1周总结报告.md
  qwen25_7b_identity_qlora_run2.yaml
  day5_train_run2.txt
  training_complete.png
  tensorboard_loss.png
```

## 内容处理

- 图片、Python 脚本、配置、CSV、训练日志、模型原始回答和执行版 Notebook 使用原文件副本，不修改内容。
- 新写 `提交说明.md`，按 Day 1–Day 5 列出文件入口，只解释必要的实验事实。
- 根据现有 Day 3 报告重写提交副本，保留配置字段、GQA、RoPE、参数量公式、实际结果和 Llama 3 对比，删除重复的验收陈述和内部校验过程。
- 根据现有 Day 5 周报重写提交副本，保留环境、每天的实验结果、训练配置、问题处理和学习总结，减少模板化标题与重复结论。
- Day 2 的代码生成错误是模型真实输出，不修饰或改写；在提交说明和周报中用一句话说明。
- Day 4 Notebook 的代码、Markdown 单元和输出保持执行时状态，不重新生成。

## 文风

- 面向熟悉大模型的导师，直接写模型、配置、实验方法和结果。
- 使用第一人称实验记录，避免“全面完成”“三重验证”“核心设计哲学”“最终结论”等总结模板。
- 不增加没有做过的实验，不从配置值推断未经测试的性能或上下文能力。
- 保留失败现象、实验局限和统计口径，避免把工程管线通过写成模型能力通过。

## 验证

- 最终目录只能包含上述 19 个文件。
- 对所有未修改副本核对 SHA-256。
- 检查 Python 语法、JSON 与 Notebook 格式。
- 确认执行版 Notebook 无错误输出，代码单元均有执行编号。
- 重新运行 Day 3 参数统计脚本，结果应为 28 层、每层 `233057792`、总参数量 `7615616512`。
- 确认 Day 5 日志包含 `training_exit_code=0`。
- 检查 Markdown 中的相对链接和高频模板化表达。
