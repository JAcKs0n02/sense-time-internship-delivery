# Day 11 AutoDL 远端验收证据

本目录保存 2026-08-03 在 AutoDL 新克隆实例
`78fb4ea487-e700080a` 上完成的环境连续性审计和 Day 11 QLoRA smoke
证据。原实例因所在主机无空闲 GPU 无法开机，用户随后克隆到新的 RTX 3090
实例；数据盘复制完成并恢复到与原实例相同的 40.18% 占用后，才开始审计和
训练。

## 新实例连续性结论

| 检查项 | 实际结果 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090，24,576 MiB，compute capability 8.6 |
| 驱动 | 595.58.03 |
| 数据盘 | 100 GB，总占用 41 GB，可用 60 GB |
| Python | 3.10.20 |
| PyTorch / CUDA | 2.5.1+cu121 / 12.1，`cuda_available=True` |
| Transformers | 4.50.0 |
| bitsandbytes | 0.43.3 |
| LLaMA-Factory | 0.9.3 |
| Week 2 数据 | 4,999 行，SHA-256 `600c01cb1d2ca41bb1fbe7fca3973336568062ead29ee0e6b6033733a86b2743` |
| Day 8 正式训练 | `global_step=max_steps=3747`，adapter 路径存在 |
| Day 9 合并模型 | `valid=true`，4 个权重分片共 15,231,271,872 bytes |

Day 8 adapter 和 Day 9 四个合并权重分片的 SHA-256 均与本地 Week 2
归档逐项相同。Week 1 顶层工程目录、Week 2 Day 6/7 交付、Day 8 正式
训练与 smoke、Day 9 合并和评测目录也均存在。

## Day 11 smoke 结果

| 指标 | 实际结果 |
|---|---:|
| 状态 | `completed` |
| 退出码 | 0 |
| 样本数 | 8 |
| Epoch | 1 |
| Optimizer steps | 2 / 2 |
| Train loss | 1.8717711567878723 |
| Train runtime | 5.3726 秒 |
| Samples / second | 1.489 |
| Steps / second | 0.372 |
| Adapter 大小 | 80,792,096 bytes |
| Adapter SHA-256 | `62cc225bfe327cee9398c9e61bb277c480ed395544e3c2b59a132a28f03f6fe9` |
| 远端路径 | `/root/autodl-tmp/qwen25-week3/smoke/day11-qlora-smoke` |

Smoke 的作用仅是验证数据注册、模型加载、4-bit QLoRA 初始化、训练、保存
adapter 和日志归档链路。它不属于九次正式实验，也不进入超参数比较结果。

## 证据说明

- `audit/new_instance_audit.txt`：GPU、磁盘、软件版本、关键路径、数据哈希和大小。
- `audit/new_instance_weight_sha256.txt`：Day 8 adapter 与 Day 9 四个合并权重分片哈希。
- `audit/prior_artifacts_inventory.txt`：Week 1/2 目录和完成标记盘点。
- `logs/day11_transfer.sha256`：远端 smoke 配置、脚本和数据副本哈希。
- `logs/day11_smoke_preflight.txt`：训练启动前环境、GPU、磁盘与输入哈希。
- `logs/day11_smoke.log`：LLaMA-Factory 完整训练日志。
- `logs/day11_smoke_status.json`：开始/结束时间、退出码和输出文件列表。
- `logs/day11_smoke_summary.json`：本次 smoke 的机器可读核心指标。
- `logs/day11_smoke_adapter.sha256`：新生成 adapter 的 SHA-256。
- `logs/day11_smoke_inventory.txt`：远端输出文件及字节数。
- `smoke/day11-qlora-smoke/`：不含模型权重的训练状态、指标和 adapter 配置。
- `audit/day11_smoke_evidence.tgz`：上述远端文本证据的原始下载包。
- `day11_smoke_evidence.sha256`：服务器生成的证据包校验值；本地下载后复核一致。

为避免把约 80 MB adapter 和约 162 MB optimizer checkpoint 提交到 Git，
仓库只保存哈希、文件清单、配置、训练状态和日志；完整 smoke 输出保留在上述
AutoDL 路径。
