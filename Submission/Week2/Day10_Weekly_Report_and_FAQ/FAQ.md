# Week 2 FAQ：问题复盘与排查记录

本文只把有本地证据支撑的情况写成“本周实际问题”。没有真实发生的
CUDA OOM 和正式训练数据解析异常单独放入预防性检查，避免把演练场景写成
实验事实。

## 1. 记录口径

- **实际问题**：本周真实遇到，并能链接到原始数据、配置或结果文件。
- **预防性检查**：本周未发生，但老师点名要求掌握的诊断与处理流程。
- **现象**是直接观察到的结果；**根因**是产生该现象的底层原因。
- `Traceback` 是 Python 异常调用栈，应优先保留首个异常位置和最后一行错误。
- `SHA-256` 是文件内容摘要，可验证本地与 AutoDL 文件是否逐字节一致。

## 2. 本周实际问题

### FAQ 1：为什么 COIG-PC 下载前需要额外登录？

| 项目 | 记录 |
|---|---|
| 状态 | 实际遇到的访问条件 |
| 阶段 | Day 6 数据收集 |
| 现象 | 未接受数据集条款或未配置只读凭据时，无法读取 COIG-PC 文件 |
| 影响 | 2,000 条 COIG-PC 样本无法进入统一抽样流程 |
| 根因 | `BAAI/COIG-PC` 是 gated dataset，仓库授权与普通公开数据集不同 |
| 修复 | 在 Hugging Face 页面接受条款，创建 read token，并在下载环境登录；凭据不写入仓库 |
| 验证 | 固定 revision 的 Parquet 文件成功读取，上游 253,558 条中确定性抽取 2,000 条；Day 6 总体验证 `valid: true` |
| 预防 | 新环境运行前先检查页面授权和 token 权限；日志、配置和提交中不得包含 token |
| 风险 | 数据提供方仍可能调整访问条款，因此复现时要重新检查数据卡 |

证据：

- [Day 6 数据源与访问说明](../../../deliverables/week2/day6/README.md#2-数据源与固定版本)
- [采集元数据](../../../deliverables/week2/day6/source/manifests/collection_metadata.json)
- [Day 6 最终验证](../../../deliverables/week2/day6/source/results/day6_validation.json)

### FAQ 2：ShareGPT 原始对话为什么不能全部原样进入 SFT？

| 项目 | 记录 |
|---|---|
| 状态 | 实际发生的数据格式异常 |
| 阶段 | Day 6 格式转换 |
| 现象 | 抽取的 1,000 条 ShareGPT 数据中有 10 条在若干完整问答后，以没有 assistant 回答的末尾 human 消息结束 |
| 期望 schema | `human → gpt` 严格交替，每一个监督目标都必须有 assistant 内容 |
| 实际 schema | 已有完整轮次之后多出一个孤立的末尾 human 消息 |
| 影响 | 若把该末尾消息当成训练轮次，就不存在可计算 SFT loss 的 assistant 目标；若静默忽略，又会损失可审计性 |
| 根因 | 上游 ShareGPT 对话是用户日志风格数据，最后一次提问不一定已经得到回答 |
| 修复 | 原始 `raw/` 记录保持不变；仅在训练可见格式中删除无法配对的最后一条 human 消息；此前所有完整轮次保留；在 provenance 中记录 `dropped_trailing_unanswered_human` |
| 验证 | 两种格式各 5,000 条且逐行对齐，provenance 为 5,000 个唯一 ID，独立验证 `valid: true` |
| 预防 | 转换器必须检查角色类型、严格交替、最少一组完整问答以及最后一条消息角色；任何自动调整都要写入 provenance |
| 风险 | 不能把相同策略扩大到角色错序、非字符串内容或没有任何完整问答的记录；这些情况必须直接失败 |

受影响的上游 `source_index`：

```text
175, 1744, 3971, 6532, 13286,
19638, 24316, 25479, 26488, 28093
```

最小结构示例：

```json
{
  "conversations": [
    {"from": "human", "value": "问题一"},
    {"from": "gpt", "value": "回答一"},
    {"from": "human", "value": "尚未得到回答的问题二"}
  ]
}
```

修复后的训练可见结构保留第一组完整问答，并在血缘记录中写明调整原因。

证据：

- [Day 6 格式转换说明](../../../deliverables/week2/day6/README.md#4-格式转换规则)
- [原始 ShareGPT 抽样](../../../deliverables/week2/day6/source/data/raw/sharegpt_zh_1k.jsonl)，其中原始索引 175 位于文件第 5 行
- [逐样本 provenance](../../../deliverables/week2/day6/source/data/interim/day6_provenance.jsonl)，其中该样本位于第 4005 行
- [转换测试](../../../deliverables/week2/day6/source/tests/test_day6_pipeline.py)
- [最终验证](../../../deliverables/week2/day6/source/results/day6_validation.json)

### FAQ 3：老师要求复制 `qwen_lora.yaml`，为什么仓库中保存的是另一个官方示例？

| 项目 | 记录 |
|---|---|
| 状态 | 实际发生的版本/文件名不一致 |
| 阶段 | Day 8 配置准备 |
| 现象 | LLaMA-Factory 0.9.3 中没有字面名为 `qwen_lora.yaml` 的纯文本 LoRA SFT 示例 |
| 影响 | 不能假装复制了不存在的文件，也不能直接照抄其他版本的不兼容参数 |
| 根因 | 老师使用的是示例类别名称，而当前固定版本的官方示例目录采用 `llama3_lora_sft.yaml` 等名称 |
| 修复 | 搜索本地固定版本，原样归档同版本官方文本 LoRA SFT 示例，再单独生成 Qwen2.5 配置；两者不混写 |
| 验证 | 官方副本 SHA-256 为 `ecd8efc4cdf29d52d8a0be25028443cc813f6d82db957a90f124d7dee5095fba`；正式配置 45 个活动参数均有注释，LLaMA-Factory 参数解析退出码为 0 |
| 预防 | 先确认 LLaMA-Factory 版本和实际示例树，再确定配置基线；不要从最新版文档复制参数到旧版运行环境 |
| 风险 | 升级框架后参数名、模板名和导出方式仍需重新验证 |

证据：

- [Day 8 官方配置来源说明](../../../deliverables/week2/day8/README.md#official-configuration-source)
- [同版本官方示例副本](../../../deliverables/week2/day8/configs/qwen_lora.official.yaml)
- [逐参数注释配置](../../../deliverables/week2/day8/configs/qwen25_7b_week2_lora_annotated.yaml)
- [配置验证结果](../../../deliverables/week2/day8/results/config_validation_local.json)

### FAQ 4：为什么 LoRA 合并成功，但合并模型没有优于基座模型？

| 项目 | 记录 |
|---|---|
| 状态 | 实际发生的验收未通过 |
| 阶段 | Day 9 推理评测 |
| 现象 | 盲评总分：基座模型 25/30，合并模型 22/30；预定判定规则返回 `merged_model_better: false` |
| 影响 | Week 2 验收标准 4 不能标记为通过 |
| 排除项 | 不是 LoRA 合并失败：导出退出码为 0，合并目录有四个完整权重分片且没有 adapter 权重；两个模型都能 BF16 CUDA 加载并回答三题 |
| 根因判断 | 现有证据只支持“本次小样本评测未提升”。可能因素包括训练数据目标过宽、缺少专门的约束遵循样本，以及三题评测方差较大；不能仅凭当前实验确定唯一根因 |
| 处理 | 如实保留盲评结果，不调换标签、不挑选更有利的问题，也不把 loss 下降解释成通用能力提升 |
| 验证 | 三题在推理前冻结；与 4,999 条训练 prompt 的最大 Jaccard 为 0.146789、0.116959、0.104478，均低于 0.80；评分后才解盲 |
| 后续 | 建立独立验证集，增加格式约束和指令遵循数据，进行单变量训练对照，并扩大评测题量 |
| 风险 | 三题只能提供小规模受控证据，不能代表模型全部能力 |

证据：

- [Day 9 对比说明](../../../deliverables/week2/day9/README.md#three-question-evaluation)
- [完整对比结果](../../../deliverables/week2/day9/source/results/base_vs_merged_comparison.json)
- [盲评评分](../../../deliverables/week2/day9/source/results/evaluation_scores.json)
- [新问题检查](../../../deliverables/week2/day9/source/results/evaluation_novelty_check.json)
- [合并模型验证](../../../deliverables/week2/day9/source/results/merged_model_validation.json)

## 3. 本周未发生：CUDA OOM 预防性排查

**状态：本周正式 smoke test、SFT、导出和推理均未发生 CUDA OOM。**
Day 8 正式训练退出码为 0，3,747 条 loss 均为有限值，错误模式计数为 0。
因此下述内容是排查手册，不是本周故障记录。

`OOM` 是 Out Of Memory。CUDA OOM 表示某次 GPU 内存分配无法满足，可能发生在
模型加载、前向计算、反向传播、optimizer step、导出或推理阶段。

### 排查顺序

1. 保存完整 Traceback，并标记最后一个成功 step；不要只截最后一行。
2. 用 `nvidia-smi` 检查剩余显存和其他 GPU 进程，区分容量不足与残留进程。
3. 记录当前 `quantization_bit`、精度、micro batch、梯度累积、
   `cutoff_len`、最长样本和 gradient checkpointing。
4. 判断故障阶段；加载阶段和反向阶段的解决方案不同。
5. 每次只改变一个变量，再用同一最小样本复现。

### 建议修复优先级

1. 清理无关 GPU 进程，但不在训练中“清空 optimizer state”。
2. 保持 4-bit QLoRA、`per_device_train_batch_size: 1` 和
   `gradient_checkpointing: true`。
3. 如果 micro batch 大于 1，先减小它，再通过
   `gradient_accumulation_steps` 维持有效 batch。
4. 降低一次推理的 `max_new_tokens`，或在评测时顺序加载模型。
5. 只有前述方法仍失败时才降低 `cutoff_len`；一旦从 2048 改小，必须重新运行
   Day 7 截断和长度统计，不能只改训练 YAML。

本周避免 OOM 的配置证据：

- [Day 8 正式配置](../../../deliverables/week2/day8/configs/qwen25_7b_week2_lora_annotated.yaml)
- [SFT 最终摘要](../../../deliverables/week2/day8/results/first_sft_summary.json)

## 4. 本周未发生：正式训练数据解析异常预防性排查

**状态：Day 8 正式训练没有出现数据解析异常。** Day 7 清洗后数据为 4,999 条，
结构验证通过，LLaMA-Factory 参数解析和正式训练均成功。下述内容用于以后排查。

### Alpaca 格式最小要求

```json
{"instruction": "必填字符串", "input": "可为空字符串", "output": "必填字符串"}
```

常见问题包括缺少 `output`、字段为 `null`/数组、空回答，以及
`dataset_info.json` 把列名映射到了不存在的字段。

### ShareGPT 格式最小要求

```json
{
  "conversations": [
    {"from": "human", "value": "问题"},
    {"from": "gpt", "value": "回答"}
  ]
}
```

常见问题包括空 conversations、角色标签不被模板识别、消息错序，以及最后没有
assistant 监督目标。

### 排查与修复

1. 保存失败行号、`source`、`source_id` 和最小原始记录。
2. 对照 `dataset_info.json` 的列映射，不先修改数据内容。
3. 运行独立 schema 校验，区分 JSON 语法错误、结构错误和空值。
4. 用实际训练 tokenizer 和 chat template 渲染一条记录，检查角色 token 和
   assistant label mask。
5. 修复转换器而不是手改最终 JSONL；添加回归测试后全量重跑。
6. 核对输出计数、SHA-256 和 2048-token 上限，再启动训练。

证据：

- [Day 7 最终验证](../../../deliverables/week2/day7/source/results/day7_validation.json)
- [Day 8 数据注册](../../../deliverables/week2/day8/source/data/dataset_info.json)
- [Day 8 配置验证](../../../deliverables/week2/day8/results/config_validation_local.json)

## 5. 关键结论

- 本周实际发现并修复了 ShareGPT 未配对末轮这一数据格式问题。
- 本周正式训练没有 CUDA OOM，也没有正式数据解析异常。
- 配置准备阶段发现了老师示例名与固定框架版本不一致的问题，并保留官方基线。
- LoRA 导出和加载成功不等于模型质量一定提升；本次三题盲评没有达到质量验收。

