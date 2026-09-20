# Qwen2.5 Day 2 原生推理实验报告

## 1. 实验信息

- GPU：NVIDIA GeForce RTX 3090，24GB 显存
- 模型：`Qwen/Qwen2.5-7B-Instruct`
- Python：3.10.20
- PyTorch：2.5.1+cu121
- CUDA：12.1
- Transformers：4.50.0
- 权重精度：BF16
- 推理接口：Transformers 原生 API
- 生成设置：`seed=42`、`do_sample=False`

## 2. 任务完成情况

| 实验项 | 结果 | 证据 |
|---|---|---|
| ModelScope 模型下载 | 15/15 完成，退出码 0 | [下载日志](source/results/day2_model_download.txt) |
| 模型完整性检查 | 配置、Tokenizer、索引和四个权重分片齐全 | [模型清单](source/results/day2_model_inventory.txt) |
| Transformers 原生推理 | 三组 Prompt 均产生非空回答 | [统一 JSONL](source/results/day2_inference_results.jsonl) |
| Chat Template 实验 | True/False 文本、IDs、tokens 均已记录 | [模板日志](source/results/day2_chat_template.txt) |
| 推理工程单元测试 | 5 个测试通过 | [最终验证](source/results/day2_completion_check.txt) |

## 3. 模型验证

权重索引引用四个 Safetensors 分片，大小分别为 3,945,441,440、3,864,726,352、3,864,726,424 和 3,556,377,672 字节。离线读取配置得到：

- `model_type=qwen2`
- `num_hidden_layers=28`
- `hidden_size=3584`
- Tokenizer 词表大小 151665

## 4. 三领域推理结果

| 领域 | 输入 token | 输出 token | 耗时（秒） | 峰值显存（GiB） | 结果 |
|---|---:|---:|---:|---:|---|
| 代码生成 | 85 | 644 | 15.205 | 14.244 | 生成 LRUCache 实现和测试 |
| 逻辑推理 | 93 | 405 | 9.346 | 14.229 | A 说谎、B 说真话、C 说谎 |
| 角色扮演 | 66 | 512 | 11.756 | 14.234 | 给出环境与模型验收清单 |

### 4.1 代码生成

模型生成的 LRUCache 实现使用 `OrderedDict`，`get` 和 `put` 的平均时间复杂度均为 O(1)。提取出的代码能够通过 Python 语法检查，但回答中自带的测试存在错误断言，pytest 返回失败。失败记录见 [代码验证日志](source/results/day2_code_validation.txt)。

这里需要区分两类测试：

- `source/tests/test_day2_scripts.py` 验证本次推理与结果处理代码，共 5 个测试通过。
- 模型回答中自带的 pytest 用例用于验证模型生成内容，其中存在逻辑错误。

因此，工程管线通过验证不代表模型生成的代码一定正确，两者必须分别检查。

### 4.2 逻辑推理

独立枚举八种真假组合得到唯一解：A 为说谎者、B 为说真话者、C 为说谎者。模型结论与参考结果一致，参考日志见 [逻辑验证](source/results/day2_logic_reference.txt)。

### 4.3 角色扮演

模型以实习导师身份指出，仅确认模型能输出文字不足以完成验收，还需要检查依赖版本、输出正确性、不同输入、推理性能、可复现文档、安全性和代码审查。

## 5. Chat Template 实验

- `add_generation_prompt=False`：23 个 token，以 user 消息的 `<|im_end|>` 和换行结束。
- `add_generation_prompt=True`：26 个 token，增加 `[151644, 77091, 198]`，对应 `<|im_start|>assistant\n`。
- 特殊 token：`<|endoftext|>=151643`、`<|im_start|>=151644`、`<|im_end|>=151645`。

生成前缀的作用是开启 assistant 回合。如果没有该前缀，输入序列只表示 user 回合已经结束，没有明确提示模型开始生成 assistant 内容。

## 6. 结论

本次实验完成了 Qwen2.5-7B-Instruct 的模型下载、完整性检查、Transformers 原生推理、三类 Prompt 测试和 Chat Template 分析。三组原始输入、生成参数、模板文本、token 数、耗时、显存和回答均有记录。逻辑推理与模板实验通过交叉验证；代码生成实验说明模型输出必须配合真实测试执行进行复核。
