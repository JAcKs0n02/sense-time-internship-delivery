# 第8周第一阶段任务5：真实LLaMA-Factory加载验收

完成日期：2026-09-15。最终状态：**TASK5_REAL_LF_CPU_LOADER_PASS_NOT_DATA_READY**。使用LLaMA-Factory真实get_train_args、load_tokenizer、get_template_and_fix_tokenizer和get_dataset路径完成SFT数据预处理，未构造Trainer、未调用模型权重加载或训练。

## 实测结果

| 检查 | train | validation |
|---|---:|---:|
| 源条数 / 框架实际条数 | 1422 / 1422 | 158 / 158 |
| 助手回答轮数 | 1441 | 158 |
| 受监督token | 85596 | 8702 |
| 多轮对话 | 13 | 0 |
| 最大序列长度 | 2048 | 1949 |
| 独立token/labels/mask比较不一致 | 0 | 0 |

共1580条、1599个助手回答、94298个受监督token。实际input_ids与冻结tokenizer完整Chat Template逐token一致，长度也与任务3记录一致。没有过滤丢行、二次截断、无有效监督或历史回答误屏蔽。每个助手回答均包含受监督的`<|im_end|>`（ID 151645），system、user及assistant角色头部均为-100；attention_mask全部为1。模板中的助手结束换行也保留在对应响应区间中。

labels是与input_ids对齐的未移位标签，这是框架返回的正确形态；因果语言模型的损失移位属于后续模型forward，本次没有执行该步骤，不能把此结果当作模型损失验证。

## 实际环境与配置

独立环境：`/tmp/internship-week8-loader-20260915/`，macOS arm64，Python 3.11.14，CPU。

| 包 | 实际版本 |
|---|---|
| LLaMA-Factory | 0.9.3 |
| Transformers | 4.50.0 |
| PyTorch | 2.5.1 |
| datasets | 3.2.0 |
| PEFT / Accelerate | 0.15.2 / 1.7.0 |
| TRL / tokenizers | 0.9.6 / 0.21.1 |

pip check通过。直接依赖见[CPU加载环境配置](../configs/requirements-loader-cpu.txt)，全量解析版本见[environment_freeze.txt](../reports/week8/phase1_task5/environment_freeze.txt)。这是本次CPU加载环境记录，不是Linux CUDA训练环境已验证的声明。

以Week3 epoch-e5配置为基础，绑定本周ShareGPT登记、train与eval_dataset、seed/data_seed=42、cutoff_len=2048、train_on_prompt=false、mask_history=false、packing=false、val_size=0，移除旧max_samples限制。为CPU验收关闭BF16/量化，设use_cpu=true、单进程预处理、隔离缓存和不写训练输出目录。完整有效配置与覆盖清单均落盘。do_train=true仅用于选择与SFT训练一致的预处理分支，脚本并不实例化Trainer或执行训练。

## 三重比对

1. 独立渲染器从原始system/user/assistant正文按冻结ChatML规则构建预期token和助手监督区间，不调用LLaMA-Factory的模板编码或processor标签函数；再与冻结的完整Chat Template及框架实际数组比对。
2. 将同一批交付Alpaca文件登记history、system并运行真实Alpaca加载器，1580条的input_ids、labels和attention_mask均与ShareGPT完全一致。不是仅比较两份JSON文本。
3. 对保存的数组再次独立核对长度、ID、split/row_index、所有助手区间和EOS、非助手屏蔽，并与归档Linux原1580条训练输入的实际数组逐sample_id对照，1580条全部匹配。后者只是历史行为一致性证据，不替代在目标GPU机器上的新鲜运行。

另通过一个独立合成样本检查显式system与两轮回答；合成样本不混入正式数据。代表性可读视图覆盖全部13条多轮和长度边界，保留完整渲染文本及每轮被监督的文本。

## 配置身份问题及修正

首轮只暂存五份tokenizer文件，没有模型config.json。AutoProcessor调用AutoConfig时，从目录名attempt中的“mpt”猜出MptConfig。首轮Qwen tokenizer及标签比对虽通过，但配置来源存在歧义，因此该轮已标记SUPERSEDED，不能作为最终验收。

已将Day3归档的原始Qwen配置一起绑定，其SHA-256为`7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c`，与此前基座配置记录一致；运行时显式断言model_type=qwen2，禁止依赖路径猜测。只复制配置与冻结tokenizer，不复制adapter_config，不加载权重。

修正后在新目录attempt-02重跑，日志正确识别Qwen2Config，不再出现MptConfig；前后两轮token行文件字节完全相同。新增回归测试保证暂存目录即使带mpt字样也有明确Qwen配置。最终加载日志仍有CPU未启用混合精度的建议提示，符合本次不训练的CPU覆盖设置。

## 交付与复现

最终证据目录：[attempt-02](../logs/week8-real-loader-20260915/attempt-02/)。首轮保留供审计，附superseded.json说明。

- [最终加载收据](../reports/week8/phase1_task5/verification.json)和[独立落盘核验](../reports/week8/phase1_task5/independent_verification.json)。
- [有效SFT配置](../reports/week8/phase1_task5/effective_sft_config.yaml)、[CPU覆盖项](../reports/week8/phase1_task5/audit_overrides.json)、[环境记录](../reports/week8/phase1_task5/environment.json)。
- 原目录train_token_rows.jsonl、validation_token_rows.jsonl：全量input_ids、labels、attention_mask、sample_id和助手区间。
- [代表样本监督视图](../reports/week8/phase1_task5/representative_spans.json)、[显式system合成检查](../reports/week8/phase1_task5/synthetic_receipt.json)。
- [加载日志](../reports/week8/phase1_task5/loader.log)、[测试记录](../reports/week8/phase1_task5/tests.txt)。27项回归测试通过，其中6项直接覆盖监督错误和配置身份；真实1580条检查独立于这些小测试。

在匹配依赖的Python环境，用不存在的新输出目录执行：

```bash
python scripts/check_week8_loader.py \
  --data-dir logs/week8-protocol-data-20260914/run-a \
  --output-dir logs/your-new-loader-audit
```

脚本先校验任务3产物与任务4内容复核绑定的哈希，再加载，完成后复核源文件未变。不会通过命令行偷偷更换seed、长度或训练数据。当前默认数据及协议均未更改，故无需重新划分或重新抽查。

## 尚未完成

任务6的最终输入/输出/配置归档和数据发布尚未完成；benchmark本地快照、裁判身份及对应污染复检仍未锁定。正式训练前还需Linux GPU软件环境及完整基座权重身份预检。本次通过的是数据加载与监督，不是CUDA可运行、模型质量改善或训练完成。
