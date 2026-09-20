# 第 7 周：量化部署与交互应用完整执行计划

## 1. 文档用途

本文将老师最新版 [`实习需求.pdf`](../实习需求.pdf) 中 Week 7 的 Day 34–Day 39 要求转换为可执行、可验证、可独立提交的工程流程。

- 设计依据：[Week 7 量化部署设计](superpowers/specs/2026-08-31-week7-quantized-deployment-design.md)
- 测试驱动实施计划：[Week 7 Implementation Plan](superpowers/plans/2026-08-31-week7-quantized-deployment-implementation.md)
- 工程归档：[Week 7 Engineering Archive](../deliverables/week7/README.md)

## 2. 老师要求与当前状态

| Day | 老师要求 | 必交内容 | 当前状态 |
|---|---|---|---|
| 34 | 学习 AWQ/GPTQ；量化 Week 4 DPO 模型 | AWQ 模型 | 本地协议/脚本完成，待 AutoDL 导出与加载验证 |
| 35 | 完成 GPTQ；比较 FP16/AWQ/GPTQ | 对比表和报告 | 数据、配置、基准工具完成，待三模型 GPU 实测 |
| 36 | vLLM 加载量化模型并提供 OpenAI API | 启动脚本、Python 客户端 | 脚本与客户端完成，待真实服务日志 |
| 37 | Gradio 历史、temperature/top-p、流式输出 | `app.py` | 本地行为测试完成，待端到端 UI 验证 |
| 38 | UI 优化；图片上传接入 Week 5 VLM | 演示录屏、优化说明 | 路由与图片门禁完成，待 VLM 服务、截图和录屏 |
| 39 | 本地部署说明和 Week 7 周报 | 部署指南、周报 | 证据绑定版本完成，GPU 结果保持待执行 |

“本地完成”只表示代码、冻结数据、静态配置和单元测试通过，不等于老师的 GPU 验收完成。

## 3. 固定模型血缘

### 文本模型

```text
model_id: reward_corrective_40step_merged
remote_path: /root/autodl-tmp/qwen25-week4/best_model/qwen25-7b-week4-dpo-corrective-merged
manifest_sha256: aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c
weight_bytes: 15231271872
```

若 merged 目录丢失，只能使用 Week 3 最终 SFT merged base 与 Week 4 adapter SHA-256 `d7a932ee4f28c8950db289126381f5d4dd30a037b238852e87ad8e5a241f9e52` 重建；不得回退到其他模型并沿用 Week 4 名称。

### 视觉模型

```text
repository: Qwen/Qwen2-VL-7B-Instruct
revision: eed13092ef92e448dd6875b2a00151bd3f7db0ac
files_digest_sha256: 575d62b8abe01fda09b28a02ff8e6cd04e8df0d15fd03e1642d3a22bbe7eb812
remote_path: /root/autodl-tmp/qwen2-vl-week5/models/Qwen2-VL-7B-Instruct
```

Week 5 LoRA 没有通过质量验收，Week 7 默认使用已验证的基座 VLM；v7-B 只能作为明确标记的实验项。

## 4. 量化路线

LLaMA-Factory 0.9.3 的 `quantization_method` 是加载时字段，其导出路径构造 GPTQ 配置，不能把未量化模型直接导出为 AWQ。因此采用：

- AutoAWQ：AWQ 4-bit、group size 128、zero point、GEMM；
- LLaMA-Factory 0.9.3 + GPTQModel：GPTQ 4-bit；
- 保存老师提示与固定版本不兼容的静态/运行时收据；
- 不把 AutoAWQ 结果描述成 LLaMA-Factory AWQ 导出。

## 5. 冻结评测协议

- 校准集：Week 2 清洗 ShareGPT 的确定性 128 条，SHA-256 `e26e7b96ba8af04b270d2c03c777c4482ddfac7ff081c4da14fce7d4e1a85419`；
- PPL 集：与校准集互斥的 256 条，SHA-256 `36c8f567f1b89e0cf226a88789994ee1d2d5ae799a826bba4674da9a19a165a7`；
- 生成集：10 条 Week 4 安全题、5 条 Week 4 业务题、5 条 Week 3 通用题；
- seed `42`，校准上限 1024 tokens，PPL 窗口 512 tokens；
- 三模型固定相同 tokenizer、模板、输入、生成参数、预热次数和测速轮次。

## 6. 指标定义

| 指标 | 计算口径 |
|---|---|
| 权重体积 | safetensors/bin 普通文件总字节 |
| 静态加载显存 | `torch.cuda.memory_allocated` 与当前进程 `nvidia-smi` 增量交叉检查 |
| 峰值显存 | 固定生成与 PPL 执行过程的进程峰值 |
| tokens/s | 3 次预热后 10 次输出 token 总数 ÷ 总生成时间 |
| 首 token | 离线 1-token 代理指标；Day 36 另记录真实 vLLM streaming TTFT |
| NLL/PPL | 256 条冻结窗口上的 token 加权 NLL 与 `exp(NLL)` |

显存门禁为 `1 - quantized_peak_mib / bf16_peak_mib >= 0.30`。vLLM 的 KV cache 预留不作为主显存口径。

## 7. 服务架构

```text
Gradio
├── 无图片 → Week 4 AWQ/GPTQ 文本 vLLM → week4-dpo-quantized
└── 有图片 → Week 5 Qwen2-VL 基座 vLLM → week5-qwen2-vl-base
```

单张 RTX 3090 默认只运行一个模型服务。两个启动脚本共享 PID 门禁，发现活跃 Week 7 服务即拒绝启动第二个；Gradio 常驻并在目标后端不可达时给出可操作提示。

## 8. 逐日执行顺序

1. Day 34 运行 Week 4 preflight、兼容性收据、AutoAWQ 导出和加载 smoke；
2. Day 35 运行 LLaMA-Factory GPTQ 导出，随后按 BF16、AWQ、GPTQ 顺序各运行一次正式基准；
3. Day 36 选择通过 30% 门禁且综合指标更好的量化模型启动文本 vLLM，验证同步/流式接口；
4. Day 37 启动 Gradio，验证历史、参数透传和增量输出；
5. Day 38 停止文本服务、启动 VLM，上传两张 Week 5 冻结图片并录屏；
6. Day 39 运行 `validate_week7.py --require-remote`，生成最终验收矩阵和教师投影。

## 9. 状态与失败原则

- `未开始`：没有正式证据；
- `本地就绪`：代码/配置/数据/测试通过，但远端收据缺失；
- `进行中`：存在部分远端证据；
- `完成`：当天全部远端门禁与提交验证通过；
- `失败`：有正式运行但指标或功能未通过。

模型缺失、哈希错误、量化退出码非零、PPL 非有限、显存降幅不足、API/stream 失败、图片未进入 VLM 或录屏缺失时均 fail-closed。原始失败证据保留，不手改 PASS。

## 10. 一手参考

- [LLaMA-Factory 模型合并与量化](https://llamafactory.readthedocs.io/en/latest/getting_started/merge_lora.html)
- [vLLM 0.6.4 OpenAI 兼容服务](https://docs.vllm.ai/en/v0.6.4.post1/serving/openai_compatible_server.html)
- [vLLM 0.6.4 量化硬件支持](https://docs.vllm.ai/en/v0.6.4.post1/quantization/supported_hardware.html)
- [vLLM 0.6.4 VLM 支持](https://docs.vllm.ai/en/v0.6.4.post1/models/vlm.html)
- [Gradio ChatInterface](https://www.gradio.app/docs/gradio/chatinterface)

## 11. 当前完成边界

截至 2026-08-31，本地设计、冻结数据、量化配置、基准工具、vLLM 脚本、客户端、Gradio 核心和 fail-closed 验证器已经建立。AutoDL AWQ/GPTQ 权重、真实显存/速度/PPL、服务日志、UI 截图和演示录屏仍需在登录可用的 RTX 3090 实例上执行，未被写成已完成事实。
