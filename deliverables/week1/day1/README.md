# Day 1：环境初始化

## 验收结论

| 老师要求 | 实际结果 | 状态 |
|---|---|---|
| 使用 `nvidia-smi` 查看 GPU 并确定模型选型 | NVIDIA GeForce RTX 3090，24576 MiB；选择 Qwen2.5-7B-Instruct，训练采用 QLoRA | 通过 |
| 创建 `llm_exp` Conda 环境 | Python 3.10.20 | 通过 |
| 安装 CUDA 12.1 对应的 PyTorch | PyTorch 2.5.1+cu121，CUDA build 12.1 | 通过 |
| 安装 LLaMA-Factory、vLLM、OpenCompass、LangChain | 四项工具均已完成安装和验证 | 通过 |
| 验证 CUDA 可用 | `torch.cuda.is_available() == True` | 通过 |
| 提供 `conda list` 和 `nvidia-smi` 环境截图 | 环境验收截图同时展示 GPU、CUDA 和关键 Conda 包 | 通过 |

## 模型选型依据

RTX 3090 提供 24GB 显存，满足 Qwen2.5-7B-Instruct 的 BF16 原生推理需求。本次 Day 2 实测峰值显存约 14.2GiB。后续微调采用 4-bit QLoRA，以减少权重和优化器状态的显存占用。

## 核心环境

| 组件 | 版本或结果 |
|---|---:|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| CUDA Runtime | 12.1 |
| Transformers | 4.50.0 |
| LLaMA-Factory | 0.9.3 |
| vLLM | 0.6.4.post1 |
| OpenCompass | 0.5.3 |
| LangChain | 1.3.13 |
| BF16 | 支持 |
| `pip check` | No broken requirements found |

## 交付证据

- [环境验收截图](evidence/day1_environment.png)：包含 `nvidia-smi`、Python/PyTorch/CUDA、`torch.cuda.is_available()`、关键 `conda list` 包和 `pip check`。
- [工具链验收截图](evidence/day1_toolchain.png)：包含完整核心版本、LLaMA-Factory CLI、CUDA 状态和磁盘检查。

## 我需要能够说明的内容

- 为什么 24GB 显存选择 7B，而不是 1.5B。
- Conda 环境解决的是 Python 与依赖隔离，不等同于 CUDA 驱动。
- PyTorch 的 `+cu121` 表示该构建包含 CUDA 12.1 用户态运行库。
- `torch.cuda.is_available()` 同时依赖 NVIDIA 驱动、PyTorch CUDA 构建和容器对 GPU 的访问。
- LLaMA-Factory、vLLM、OpenCompass、LangChain 分别负责训练、推理服务、评测和应用编排。

完整讲解见 [Day 1 与 Day 2 知识笔记](../notes/day1_day2_knowledge.md)。
