# Day 1 与 Day 2 知识讲解

这份笔记用于理解实验，而不是罗列操作记录。目标是能够独立解释环境为什么这样配置、原生推理如何流动，以及每份证据证明了什么。

## 一、Day 1：环境为什么这样搭建

### 1. GPU 规格如何决定模型选择

模型推理首先要把权重放入显存。7B 模型若使用 BF16，每个参数约占 2 字节，仅权重理论值就约 14GB；实际运行还需要 KV Cache、临时张量和框架开销。因此：

- 8GB 显存更适合 1.5B 模型或量化推理。
- 16GB 可以尝试 7B 量化或较受限的推理。
- 24GB RTX 3090 可以完成 7B BF16 原生推理，并为后续 QLoRA 留出空间。

本次三组推理峰值显存约 14.2GiB，与上述估算一致。

### 2. 为什么训练阶段使用 QLoRA

全参数训练不仅保存模型权重，还需要梯度和优化器状态，显存远高于推理。QLoRA 的核心做法是：

1. 将基座模型权重量化到 4 bit，降低常驻显存。
2. 冻结基座权重。
3. 只训练小规模 LoRA 适配器参数。

因此 24GB GPU 可以承担 7B 模型的小规模实验，而不需要全参数训练所需的多卡资源。

### 3. Conda、CUDA 驱动和 PyTorch 分别负责什么

- Conda 环境隔离 Python 版本和第三方包。
- NVIDIA 驱动负责操作系统与 GPU 硬件通信，`nvidia-smi` 验证的是这一层。
- PyTorch `2.5.1+cu121` 包含 CUDA 12.1 用户态运行库和 GPU 算子。
- `torch.cuda.is_available()` 为 True，说明 PyTorch 能通过驱动发现并使用 GPU。

所以“Conda 创建成功”不等于“CUDA 可用”，`nvidia-smi` 正常也不等于“PyTorch CUDA 构建正确”。两项都必须验证。

### 4. 四项核心工具链的职责

| 工具 | 主要职责 | 本周用途 |
|---|---|---|
| LLaMA-Factory | 微调、LoRA/QLoRA、数据集和训练配置 | Day 5 跑通 identity 数据集训练 |
| vLLM | 高吞吐推理与服务部署 | 后续推理服务实验 |
| OpenCompass | 标准数据集评测和模型能力比较 | 后续模型评测 |
| LangChain | Prompt、模型、检索和工具的应用编排 | 后续应用链路实验 |

Day 1 安装它们的意义是建立整周统一环境，不代表四个工具当天都要完成完整业务实验。

### 5. Day 1 口述示例

> 我使用 RTX 3090 24GB，因此选择 Qwen2.5-7B-Instruct。7B BF16 权重理论占用约 14GB，本次推理峰值也约 14.2GiB；后续训练采用 4-bit QLoRA，避免全参数训练的梯度和优化器状态占满显存。环境使用 Python 3.10 和 PyTorch 2.5.1+cu121，`nvidia-smi`、`torch.cuda.is_available()` 和实际 CUDA 矩阵计算均通过。LLaMA-Factory、vLLM、OpenCompass 和 LangChain 分别对应训练、推理服务、评测和应用编排。

## 二、Day 2：原生推理完整数据流

### 1. 模型下载包含什么

一个可用的本地模型目录不仅需要权重，还需要：

- `config.json`：网络层数、隐藏维度、注意力头等结构配置。
- `tokenizer.json` 和 `tokenizer_config.json`：分词规则与特殊 token。
- `generation_config.json`：默认生成配置。
- `model.safetensors.index.json`：参数名称到权重分片的映射。
- 四个 `model-*.safetensors`：实际模型权重。

模型完整性检查不能只看目录大小，而要读取索引并确认索引引用的所有分片均存在。

### 2. `AutoTokenizer` 与 `AutoModelForCausalLM`

- `AutoTokenizer` 将文字转换为 token IDs，也负责将 token IDs 解码回文字。
- `AutoModelForCausalLM` 根据 `config.json` 自动选择 Qwen2 架构类并加载因果语言模型权重。
- `local_files_only=True` 保证加载来自已经下载的本地目录。

### 3. 一条对话如何变成模型回答

推理脚本中的完整流程是：

1. 从 JSON 读取 system 和 user Prompt。
2. 组成 `messages=[{"role": "system", ...}, {"role": "user", ...}]`。
3. 使用 `apply_chat_template(..., add_generation_prompt=True)` 转成 Qwen 对话格式。
4. Tokenizer 将模板文本编码成 `input_ids` 张量。
5. 张量移动到 GPU。
6. `model.generate()` 自回归生成后续 token。
7. 切掉输入部分，只保留模型新生成的 token。
8. `batch_decode(..., skip_special_tokens=True)` 得到最终回答。
9. 将 Prompt、模板、参数、token 数、耗时、显存和回答写入 JSONL。

“切掉输入部分”非常重要。`generate()` 返回的是“输入 token + 新生成 token”，如果直接解码整个张量，结果会重复包含原 Prompt。

### 4. 生成参数如何理解

本次使用：

- `seed=42`：固定随机种子。
- `do_sample=False`：使用确定性的贪心生成，不进行概率采样。
- `max_new_tokens`：限制最大新生成 token 数，而不是限制总输入长度。
- `use_cache=True`：复用历史注意力键值，减少重复计算。

三类 Prompt 设置不同的 `max_new_tokens`，是因为代码回答通常比逻辑结论和角色回答更长。

## 三、Chat Template 为什么重要

Qwen 使用如下对话边界：

```text
<|im_start|>system
系统消息<|im_end|>
<|im_start|>user
用户消息<|im_end|>
<|im_start|>assistant
```

`add_generation_prompt=True` 会在末尾补上 assistant 回合起始标记。本次实验中：

- False：23 个 token。
- True：26 个 token。
- 新增的三个 token ID 为 `[151644, 77091, 198]`。

它们对应 `<|im_start|>`、`assistant` 和换行。模型在训练时见过这种格式，因此模板是否正确会直接影响模型对角色边界的理解。

### Chat Template 口述示例

> Chat Template 把结构化的 system/user 消息转换成 Qwen 训练时使用的特殊 token 序列。`add_generation_prompt=True` 会追加 `<|im_start|>assistant\n`，明确开启 assistant 回合。本次实验中 token 数从 23 增加到 26，新增 ID 是 151644、77091、198。如果不加生成前缀，序列只表示 user 回合结束，模型没有得到同样明确的 assistant 开始信号。

## 四、如何理解三类 Prompt 的结果

### 代码生成

模型成功生成了 O(1) 的 LRUCache 实现，但回答自带的测试存在错误断言。这个结果说明：

- 语法正确不代表逻辑正确。
- 生成的测试也需要人工检查。
- 代码题必须经过真实执行，而不是只阅读代码块。

### 逻辑推理

独立枚举得到唯一解 `(False, True, False)`，即 A 说谎、B 说真话、C 说谎。模型结论与参考脚本一致，因此不仅有模型回答，还有外部交叉验证。

### 角色扮演

角色 Prompt 主要检验模型是否能遵循“严格导师”的身份和输出要求。回答覆盖环境、正确性、性能、复现、安全和审查，说明角色约束基本生效。

## 五、老师可能追问的问题

### 为什么不用 1.5B？

因为 RTX 3090 有 24GB 显存，实测可以容纳 7B BF16 推理；7B 更符合后续架构分析和 QLoRA 实验目标。

### 为什么不用 vLLM 完成 Day 2？

Day 2 明确要求 Transformers 原生接口。原生接口更容易直接观察 Tokenizer、Chat Template、输入 token 和 `model.generate()` 的完整过程；vLLM 更适合后续高吞吐服务。

### 为什么固定 `do_sample=False`？

为了减少随机性，使三组实验更容易复现和对比。它不代表采样参数永远不重要，只是本次实验选择确定性生成。

### 为什么模型代码失败仍然保留？

实验的目标是记录真实模型能力，而不是修改输出使结果看起来正确。失败本身证明了执行验证的必要性。

### JSONL 和三份 TXT 有什么区别？

JSONL 适合程序读取和批量统计；TXT 适合人工查看。两者保存同一组核心内容，分别服务于机器验证和人工验收。
