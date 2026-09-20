# Day 5：LLaMA-Factory 初探与周报

## 验收结论

Day 5 已使用 AutoDL RTX 3090 和既有 `llm_exp` 环境完成。LLaMA-Factory 0.9.3 对 Qwen2.5-7B-Instruct 的官方 identity 数据执行了两轮 4-bit QLoRA SFT；两轮 `llamafactory-cli train` 均以退出码 0 完成，TensorBoard 同时记录两条 loss 曲线。固定身份评测从训练前 0/8 提升到 Run 1 的 6/8，再提升到 Run 2 的 7/8。

| 老师要求 | 实际结果 | 状态 |
|---|---|---|
| 使用 `llamafactory-cli` 跑通 identity 数据集 | 91 条官方 identity 样本，Qwen2.5-7B-Instruct 4-bit QLoRA；Run 1 和 Run 2 均正常退出 | 通过 |
| 训练管线无报错完成 | 正式成功日志包含完整 step、loss、learning rate、epoch、耗时与 `training_exit_code=0` | 通过 |
| 可视化训练日志 | TensorBoard Scalars 同时显示 Run 1、Run 2 的 `train/loss` 曲线与 step | 通过 |
| 撰写第 1 周总结报告 | 覆盖 Day 1–Day 5、问题排查、设计哲学、验收映射和复现入口 | 通过 |
| 在流程跑通基础上尽量取得更好结果 | 仅把 epoch 从 3 调到 5，固定身份评测由 6/8 提升到 7/8 | 通过 |

## 核心结果

| 项目 | Run 1 | Run 2（最终选择） |
|---|---:|---:|
| 配置 epoch | 3 | 5 |
| Trainer 实际记录 epoch | 2.8791 | 4.7912 |
| 优化步数 | 66 | 110 |
| 训练耗时 | 96.5597s | 160.9298s |
| 汇总 train loss | 1.4312 | 1.0749 |
| 最后一步 loss | 0.5694 | 0.3365 |
| 固定身份评测 | 6/8 | 7/8 |
| 训练退出码 | 0 | 0 |

LLaMA-Factory 以 91 条样本、batch size 1、gradient accumulation 4 计算优化步数。因为每个 epoch 的最后不足 4 个 micro-batches 不形成额外优化步，Trainer 最终显示的实际 epoch 小于配置整数；这不是训练提前中断。两份日志均包含计划步数并以退出码 0 结束。

Run 2 的训练变量只把 `num_train_epochs` 从 3.0 改为 5.0。`output_dir` 和 `run_name` 的变化仅用于隔离产物；模型、数据、量化、LoRA、学习率、batch、模板、随机种子和评测条件均保持不变。

## 训练配置

- 基座模型：`Qwen/Qwen2.5-7B-Instruct`
- 方法：SFT + 4-bit QLoRA
- 量化：bitsandbytes NF4、double quantization、BF16 compute
- LoRA：rank 8、alpha 16、dropout 0.05、target `all`
- 数据：91 条 identity 样本，`cutoff_len=512`
- batch：每卡 1，梯度累积 4，有效 batch 4
- 优化：AdamW 默认实现、学习率 `1e-4`、cosine、warmup 0.1、weight decay 0.01
- 可复现：`seed=42`，每 step 记录 TensorBoard
- 可训练参数：20,185,088 / 7,635,801,600，约 0.2643%

配置入口：

- [Run 1 配置](configs/qwen25_7b_identity_qlora.yaml)
- [Run 2 配置](configs/qwen25_7b_identity_qlora_run2.yaml)
- [结构化指标](source/results/day5_training_metrics.json)

## identity 数据来源与处理

数据来自 LLaMA-Factory 官方 Hugging Face 数据集 `llamafactory/demo_data`，固定 revision 为 `999e7a11dadd6ce929180d7f3d61d4ca7e761db8`，与本次 LLaMA-Factory 0.9.3 使用时间相匹配。官方模板共 91 条；只替换其文档化占位符：

- `{{name}}` → `Qwen2.5-Intern`
- `{{author}}` → `Week 1 Internship Project`

模板 SHA-256 为 `475cca...efc8`，解析后数据 SHA-256 为 `57e0d9...f3c3`。目标名称仅是本次训练实验标签，不代表 Qwen 官方产品名称。

数据入口：

- [官方模板快照](source/data/identity.template.json)
- [占位符解析后的训练数据](source/data/identity.json)
- [本地数据注册](source/data/dataset_info.json)
- [数据准备脚本](source/scripts/prepare_identity_data.py)
- [来源与哈希记录](source/results/day5_data_preparation.txt)

AutoDL 直接连接官方数据源时出现网络停滞，因此没有伪造下载成功记录，也没有重写数据；改为把本地校验过的同一固定 Git revision 原样上传到数据盘。首次网络失败保留在 [下载尝试日志](source/results/day5_data_download_attempt.txt)。

## 训练前后固定评测

三轮评测使用相同的 8 个 Prompt、system message、Qwen Chat Template，以及 `do_sample=False`、`max_new_tokens=128`、`use_cache=True`。训练前与训练后都用 BF16 加载基座，以保证评测口径一致；4-bit 量化用于训练。

- [固定 Prompt](source/prompts/identity_prompts.json)
- [训练前原始 JSONL](source/results/identity_before_training.jsonl)
- [Run 1 原始 JSONL](source/results/identity_after_run1.jsonl)
- [Run 2 原始 JSONL](source/results/identity_after_run2.jsonl)
- [逐题评测报告](source/results/identity_evaluation.md)
- [三轮完整文本汇总](source/results/day5_identity_comparison_summary.txt)

Run 2 唯一严格失败项是 `conflict_chatgpt`：回答给出了正确创建者，也没有接受错误的 ChatGPT/OpenAI 身份，但遗漏精确名称 `Qwen2.5-Intern`，因此按预设规则记为失败。原始回答未修改。

## 训练日志与 TensorBoard

- [Run 1 完整训练日志](source/results/day5_train.txt)
- [Run 2 完整训练日志](source/results/day5_train_run2.txt)
- [Run 1 adapter 文件清单](source/results/day5_adapter_inventory.txt)
- [Run 2 adapter 文件清单](source/results/day5_adapter_inventory_run2.txt)
- [TensorBoard 页面文本记录](source/results/day5_tensorboard_page_text.txt)
- [训练完成截图](evidence/training_complete.png)
- [TensorBoard loss 截图](evidence/tensorboard_loss.png)

adapter 权重、checkpoint、TensorBoard event 文件、模型权重和 Conda 环境没有复制到 Git 仓库；清单和截图证明产物存在，原始训练日志提供可审计数值。

## 问题排查记录

正式成功结果没有删除之前的失败事实：

1. bitsandbytes 0.43.1 不满足当前 Transformers/Accelerate 4-bit 加载路径要求，首次训练在进入 step 前失败。[失败日志](source/results/day5_train_failed_bnb0431.txt) 已保留。
2. 只把 bitsandbytes 升级到 0.43.3；PyTorch、Transformers、Accelerate 和 LLaMA-Factory 保持不变，`pip check` 通过。[升级日志](source/results/day5_bitsandbytes_upgrade.txt)
3. TensorBoard 在服务器 `localhost:6006` 返回 HTTP 200，但该机房的外部 HTTP 服务映射不可用，因此在服务器内部用无头 Chromium 打开真实 TensorBoard 页面并截图。[页面捕获日志](source/results/day5_tensorboard_capture.txt)

这些失败日志用于说明根因和解决过程，不影响两份正式训练日志的退出码 0。

## 周报与复现

- [《第 1 周：环境与大模型导论总结报告》](REPORT.md)
- [Day 5 只读验证脚本](source/scripts/validate_day5.py)
- [Day 5 测试](source/tests/test_day5_scripts.py)
- [最终验证日志](source/results/day5_final_validation.txt)
- [交付文件 SHA-256](source/results/day5_files.sha256)

本地只读验证：

```bash
python3 -m unittest discover \
  -s deliverables/week1/day5/source/tests -v

python3 deliverables/week1/day5/source/scripts/validate_day5.py \
  --root deliverables/week1/day5 \
  --output deliverables/week1/day5/source/results/day5_final_validation.json
```

## 能力边界

本实验验证的是 LLaMA-Factory 的数据注册、4-bit QLoRA、adapter 保存、TensorBoard 记录和加载 adapter 的完整管线。7/8 只代表固定 identity 问题上的结果，不说明模型的通用推理、代码生成、事实性或安全能力整体提升。
