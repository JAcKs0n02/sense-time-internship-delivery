# 第 1 周 Day 1–Day 3 阶段提交说明

## 阶段完成情况

| 日期 | 任务 | 结论 | 详细入口 |
|---|---|---|---|
| Day 1 | 环境初始化 | RTX 3090、Python 3.10、PyTorch CUDA 12.1 与四项核心工具链均通过验证 | [Day 1 交付说明](day1/README.md) |
| Day 2 | 模型下载与原生推理 | Qwen2.5-7B-Instruct 下载完整；三类 Prompt 与 Chat Template 均已实际运行 | [Day 2 交付说明](day2/README.md) |
| Day 3 | 架构深度剖析 | 完成配置字段、逐层参数量、GQA、RoPE 与 Llama 3 架构对比 | [Day 3 交付说明](day3/README.md) |

## Day 1 结果

- GPU：NVIDIA GeForce RTX 3090，24576 MiB。
- 模型选型：Qwen2.5-7B-Instruct；后续训练采用 QLoRA。
- Conda 环境：`llm_exp`，Python 3.10.20。
- PyTorch：2.5.1+cu121；`torch.cuda.is_available()` 返回 `True`。
- LLaMA-Factory、vLLM、OpenCompass、LangChain 均完成安装与验证。

证据：

- [环境验收截图](day1/evidence/day1_environment.png)
- [工具链验收截图](day1/evidence/day1_toolchain.png)

## Day 2 结果

- 通过 ModelScope 下载 `Qwen/Qwen2.5-7B-Instruct`，15/15 文件完成。
- 使用 Transformers 的 `AutoTokenizer` 和 `AutoModelForCausalLM` 完成原生推理。
- 保存代码生成、逻辑推理、角色扮演三组完整对话及统一 JSONL。
- 验证 `apply_chat_template` 的生成前缀、token IDs 和特殊 token。
- 代码生成回答中的测试错误按真实实验结果保留，未修改模型原始回答。

核心交付：

- [原生推理脚本](day2/source/scripts/inference.py)
- [三组统一 JSONL](day2/source/results/day2_inference_results.jsonl)
- [代码生成日志](day2/source/results/day2_code_generation.txt)
- [逻辑推理日志](day2/source/results/day2_logic_reasoning.txt)
- [角色扮演日志](day2/source/results/day2_role_play.txt)
- [Chat Template 日志](day2/source/results/day2_chat_template.txt)
- [Day 2 实验报告](day2/REPORT.md)

## Day 3 结果

- `vocab_size=152064`
- `hidden_size=3584`
- `num_hidden_layers=28`
- `num_attention_heads=28`
- `num_key_value_heads=4`
- `head_dim=128`
- `rope_theta=1000000`
- 每 7 个 Query heads 共享一组 K/V。
- 单个 Decoder Layer：`233,057,792` 个参数。
- 总参数量：`7,615,616,512`。
- Transformer 主体参数量：`6,525,621,760`。

核心交付：

- [Qwen2.5 架构分析报告](day3/REPORT.md)
- [官方配置副本](day3/source/config/config.json)
- [参数统计脚本](day3/source/scripts/analyze_params.py)
- [28 层参数量 CSV](day3/source/results/day3_parameter_counts.csv)
- [手工公式交叉核对](day3/source/results/day3_manual_crosscheck.txt)
- [最终验证日志](day3/source/results/day3_final_validation.txt)
- [GPU 环境预检原始日志](day3/source/results/day3_gpu_preflight.txt)
- [实际权重参数验证原始日志](day3/source/results/day3_weight_parameter_verification.txt)
- [实际权重逐层参数 CSV](day3/source/results/day3_weight_parameter_counts.csv)
- [实际模型验证文件校验清单](day3/source/results/day3_gpu_files.sha256)

## 当前验收状态

- CUDA 环境可用。
- Qwen2.5-7B-Instruct 可以正常完成对话生成。
- RoPE 与 GQA 的原理、公式和设计取舍已整理到架构报告。
- Day 3 已在 RTX 3090 上加载实际 BF16 权重，逐层参数统计与公式完全一致，最小前向传播通过。
- Day 1–Day 3 的代码、日志、截图与报告均已保存，可从上述链接逐项复核。
