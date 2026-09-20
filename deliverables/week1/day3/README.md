# Day 3：Qwen2.5 架构深度剖析

## 完成结论

Day 3 已完成。分析对象为 `Qwen/Qwen2.5-7B-Instruct`。交付内容覆盖真实配置字段分析、28 层参数量统计、实际权重核验、最小 GPU 前向传播、手工公式交叉核对，以及与原始 `Meta-Llama-3-8B` 的架构对比。

核心结果：

- 模型：Qwen2.5-7B-Instruct
- Decoder Layers：28
- Hidden Size：3584
- Attention Heads：28 Q / 4 KV
- Head Dimension：128
- GQA 分组：每 7 个 Query heads 共享一组 K/V
- RoPE Theta：1,000,000
- 总参数量：`7,615,616,512`
- Transformer 主体参数量：`6,525,621,760`

## 老师要求逐项对应

| 老师要求 | 完成情况 | 文件入口 |
|---|---|---|
| 分析 `vocab_size`、`hidden_size`、`num_attention_heads`、`num_key_value_heads`、`rope_theta` | 已完成，同时覆盖层数、FFN、RMSNorm、上下文、bias、weight tying 等字段 | [架构分析报告](REPORT.md#3-configjson-逐字段分析) |
| 编写脚本统计模型各层参数量与总参数量 | 已完成；公式 CSV 与实际权重 CSV 均包含 embedding、28 个 decoder layer、final norm 和 lm head，结果一致 | [统计脚本](source/scripts/analyze_params.py)、[公式统计 CSV](source/results/day3_parameter_counts.csv)、[实际权重统计 CSV](source/results/day3_weight_parameter_counts.csv) |
| 对比 Qwen2.5 与 Llama 3 | 已完成，对象固定为原始 Meta-Llama-3-8B，对比 15 个架构维度 | [架构对比](REPORT.md#7-qwen25-7b-与原始-meta-llama-3-8b-对比) |
| 提交《Qwen2.5 架构分析报告》 | 已完成 | [REPORT.md](REPORT.md) |

## 交付文件

```text
day3/
├── README.md
├── REPORT.md
├── evidence/
│   ├── config_and_parameter_summary.png
│   └── final_validation.png
└── source/
    ├── config/
    │   ├── README.md
    │   ├── config.json
    │   └── day3_config_pretty.json
    ├── results/
    │   ├── day3_final_validation.txt
    │   ├── day3_gpu_files.sha256
    │   ├── day3_gpu_preflight.txt
    │   ├── day3_manual_crosscheck.txt
    │   ├── day3_parameter_counts.csv
    │   ├── day3_parameter_summary.txt
    │   ├── day3_weight_parameter_counts.csv
    │   └── day3_weight_parameter_verification.txt
    ├── scripts/
    │   └── analyze_params.py
    └── tests/
        └── test_analyze_params.py
```

主要阅读顺序：

1. [Qwen2.5 架构分析报告](REPORT.md)
2. [参数统计控制台摘要](source/results/day3_parameter_summary.txt)
3. [28 层参数量 CSV](source/results/day3_parameter_counts.csv)
4. [手工公式交叉核对](source/results/day3_manual_crosscheck.txt)
5. [最终验证日志](source/results/day3_final_validation.txt)
6. [GPU 环境预检原始日志](source/results/day3_gpu_preflight.txt)
7. [实际权重参数验证原始日志](source/results/day3_weight_parameter_verification.txt)
8. [实际权重逐层参数 CSV](source/results/day3_weight_parameter_counts.csv)
9. [实际模型验证文件校验清单](source/results/day3_gpu_files.sha256)

## 配置来源与完整性

`source/config/config.json` 来自 Qwen 官方模型仓库：

<https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json>

官方文件与本地副本的 SHA-256 均为：

```text
7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c
```

AutoDL 实际模型目录中的 `config.json` 也得到同一 SHA-256。具体说明见 [配置来源说明](source/config/README.md)。

## 复现命令

参数统计不加载模型权重，只需要 Python 标准库：

```bash
python3 deliverables/week1/day3/source/scripts/analyze_params.py \
  --config deliverables/week1/day3/source/config/config.json \
  --output deliverables/week1/day3/source/results/day3_parameter_counts.csv
```

运行测试：

```bash
python3 -m unittest discover \
  -s deliverables/week1/day3/source/tests -v
```

验证结果：3 项测试全部通过；CSV 可重复生成；28 个 decoder layer 全部存在；Python 语法、报告必需字段和配置 SHA-256 均通过检查。

## 实际模型与 GPU 运行验证

2026-07-17 在 AutoDL RTX 3090 上使用 PyTorch `2.5.1+cu121`、Transformers `4.50.0` 和 BF16 加载实际四个 `safetensors` 权重分片。验证结果：

- 服务器实际 `config.json` SHA-256 与本地分析副本完全一致。
- 28 个 Decoder Layer 实测均为 `233,057,792` 个参数。
- 实际权重总参数量为 `7,615,616,512`，与公式脚本和手工核对完全一致。
- 第 0 层 Q/K/V/O 与 lm head 的实际张量形状符合报告中的 GQA 结构。
- 单 token 前向传播成功，logits 形状为 `(1, 1, 152064)`，峰值已分配显存约 `14.195 GiB`。

参数量证据由配置公式统计、手工公式核对和实际权重逐模块统计共同构成。三种方法结果一致，GPU 前向传播同时确认模型权重能够正常参与计算。原始日志在复制过程中未修改，可使用 [`day3_gpu_files.sha256`](source/results/day3_gpu_files.sha256) 核对文件完整性。

## 证据图

配置与参数统计：

![配置与参数统计](evidence/config_and_parameter_summary.png)

最终验证：

![最终验证](evidence/final_validation.png)

## 验收结论

- [x] 老师点名的五个配置字段全部解释。
- [x] 28 个 decoder layer 全部出现在 CSV 中。
- [x] 脚本统计与手工公式一致。
- [x] 实际权重逐层统计与公式脚本一致。
- [x] RTX 3090 最小前向传播通过。
- [x] 总参数量为 `7,615,616,512`。
- [x] Qwen2.5 与原始 Llama 3 完成多维对比。
- [x] 报告准确区分 `rope_theta`、配置上下文和实际部署长度。
- [x] RoPE 与 GQA 的原理、收益和取舍均写入报告。
