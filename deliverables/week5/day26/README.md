# Day 26：200 条图文数据与冻结视觉塔 LoRA

## 状态

**实验流程已完成，效果总门槛未通过。** 已完成 200 条唯一图文指令、70 张真实来源图片、LLaMA-Factory 多模态 LoRA、视觉塔/投影层冻结审计、单步 smoke、6 组正式训练候选、开发集选模和两次独立最终盲评。最终 v6 纠偏候选通过胜题数与幻觉率门槛，但均分仅提升 `+0.0125`，未达到预注册的 `+0.50`，因此不能写成“明显提升 PASS”。

执行依据：[第 5 周完整执行计划](../../../docs/week5_execution_plan.md) · [Week 5 工程索引](../README.md) · [Day 22 基座规范](../day22/README.md) · [Day 25 污染排除规范](../day25/README.md)

## 老师要求

1. 构造 200 条图文指令数据；
2. 使用 LLaMA-Factory 对 VLM 做 LoRA 微调；
3. 冻结 ViT 视觉编码器，只训练语言模型中的低秩适配器；
4. 交付微调后的 VLM 与微调日志；
5. Week 5 最终验收要求微调后有明确效果提升。

本项目在老师的 200 条训练数据之外，另建 20 条开发记录与 20 条最终记录。开发集只用于选模；最终集不计入 200 条训练数量，也不用于调参。两组均与训练集图片 SHA 零重叠。

## 当日交付

- 正好 200 条通过验证的图文训练数据及数据 manifest；
- 额外 20 条开发记录、20 条最终评测及仅本地保存的私有盲评映射；
- 训练/评测图片 SHA-256 零重叠和 Day 25 污染检查；
- 锁定 revision 的 LLaMA-Factory 数据 schema smoke；
- 生效的多模态 SFT LoRA 配置，包含 `freeze_vision_tower: true`；
- 单步 smoke、正式训练命令、原始日志、loss 和显存记录；
- LoRA adapter 清单、大小、SHA-256 和加载 smoke；
- Base/LoRA 各 20 条原始回答、盲评评分和三项预注册门槛结论；
- 失败 run 摘要与纠偏记录。

## 实际执行结果

### 数据与真实来源

- 使用 70 张真实、可追溯、revision 固定的开源图片，来源于 scikit-image、PaddleOCR、Pix2Text 和 OmniParser；
- 图片分为 50 张训练、10 张开发、10 张最终，70 个 SHA-256 全部唯一；
- 由 50 张训练图片构造正好 200 条唯一训练指令，每类 40 条；
- v6 仍保留 200 条唯一教师数据，只对 direct/correction 做确定性重复采样，实际训练采样 300 条；
- `training_input_manifest_v6.json` 固定 grounded 200 条、实际 weighted 300 条和逐文件 SHA，并由最终验证器交叉核对训练配置、完整 Day22 基座 manifest、评测配置/样本、LoRA scale 与 adapter lineage；
- 训练/开发/最终图片 SHA 零重叠，并与 Day 25 图像、Prompt 零污染；
- 三个 split 使用同一组通用任务指令模板：开发/最终各 20 条 Prompt 均能在训练集中找到同模板，但目标答案精确重叠为 0；因此本评测衡量的是新图片上的受控任务泛化，不是新指令措辞泛化；
- 所有来源页、原始下载 URL、作者、许可、revision、尺寸和哈希保存在 manifest。
- 来源元数据按对应开源仓库/项目记录并固定 revision；它用于技术追溯，不等同于逐图片的法律授权意见。

### 训练与冻结验证

最终候选为 `formal_lora_v6_grounded/checkpoint-84`：

| 项目 | 结果 |
|---|---:|
| 框架 | LLaMA-Factory 0.9.3 (`ca75f1e…`) |
| 基座 | Qwen2-VL-7B-Instruct (`eed1309…`) |
| LoRA target | q/k/v/o + gate/up/down，仅语言模型 |
| rank / LR / epoch | 16 / 2e-5 / 1.5 |
| optimizer steps | 113 |
| train loss | 2.188334 |
| eval loss | 1.943 |
| 训练耗时 | 289.9694 秒 |
| 可训练参数 | 40,370,176 / 8,331,745,792 |
| adapter tensor | 392 |
| adapter 大小 | 161,533,192 bytes |
| adapter SHA-256 | `fb36ad9452cdd1444b06f368c6c39a5d70c973769ccb886c4c811d88f15aa423` |
| 视觉塔/投影层 | trainer 冻结日志、可训练参数审计及 adapter 键检查均 PASS；基座参考张量在加载 adapter 前后 SHA 一致 |
| 正式训练 exit | 0 |

首次 smoke 包装命令因远端没有 `/usr/bin/time` 在训练前退出 127；改用可用计时方式后，单步 smoke 的 loss 为约 2.7655，adapter 可加载、视觉参数未变化，退出码为 0。该失败尝试保留，不影响正式训练。

### 候选选择与盲评

v1–v5 分别暴露过拟合、LoRA 容量不足或类别不平衡问题。v6 将目标改写为“可见事实—证据—不确定性边界”，并只在固定开发集选择 `checkpoint-84` 和 LoRA scale `0.85`。

首轮 v2 最终盲评：Base 3.9175、LoRA 3.8750、LoRA 8/20 胜、幻觉率 65%→45%，总门槛 FAIL。

纠偏 v6 最终盲评在评分文件哈希冻结后揭盲：

| 指标 | Base | LoRA | 检查 |
|---|---:|---:|---|
| 加权均分 | 3.7825 | 3.7950 | `+0.0125 < +0.50`，FAIL |
| 胜题数 | — | 12/20 | PASS |
| 幻觉率 | 65% | 55% | PASS |

评分前冻结的匿名评分文件 SHA-256 为 `2d3ea02814cedb9ce3b15ddb79b41c7df7f3bc4a029673047a779c3422a5a846`。原 v2 失败结果、v6 匿名 packet、评分理由和揭盲结果在工程区保留；教师投影只保留身份明确的复核结果与预揭盲哈希，不包含匿名 packet、匿名评分、盲化种子或私有 A/B 映射。

## 专业术语

| 术语 | 解释 |
|---|---|
| SFT | Supervised Fine-Tuning，用“输入图像+指令→目标回答”监督模型学习 |
| LoRA | Low-Rank Adaptation，在目标线性层增加少量低秩可训练参数，降低训练显存和模型增量体积 |
| Adapter | LoRA 训练产生的增量权重；加载时必须与正确的基座模型组合 |
| Rank | LoRA 低秩矩阵的维度；越大参数越多，但不保证效果更好 |
| 视觉塔 / ViT | 将图像编码为视觉特征的视觉 Transformer；老师要求保持冻结 |
| 冻结 | 参数不更新；既要检查 `requires_grad`/可训练参数名，也要验证训练前后视觉参数未改变 |
| LLM adapter | 插入语言模型线性层的 LoRA 参数；本项目不能把视觉模块列为可训练目标 |
| Multimodal ShareGPT | LLaMA-Factory 支持的一类多模态对话数据格式，消息与图片路径分开注册 |
| Schema smoke | 用 1–4 条真实数据经 dataset parser 和 processor 完整加载，验证字段确实兼容锁定版本 |
| 隔离评测 | 20 条从未进入训练、示例、调参或选模的固定样例 |
| 数据污染 | 训练与评测共享图片、问题、答案或近似改写，导致评测不能代表泛化 |
| 盲评 | 隐藏 Base/LoRA 身份后评分，减少知道模型身份带来的偏差 |
| Model lineage | 基座 revision、数据哈希、配置、代码 revision、训练 run 和 adapter 哈希构成的血缘 |
| Train loss | 训练目标在数据上的优化值；下降说明优化进行，但不能单独证明回答质量提升 |

## 固定输入

### 基座与训练框架

| 字段 | 固定规则 |
|---|---|
| base model | Day 22 冻结的 `Qwen/Qwen2-VL-7B-Instruct` |
| base revision | 必须与 Day 22 model manifest 完全一致 |
| framework | LLaMA-Factory，执行时固定不可变 Git revision |
| stage | 多模态 SFT |
| finetuning | LoRA |
| vision freeze | `freeze_vision_tower: true` |
| seed | `42` |
| training count | `200` |
| held-out count | `20` |

LLaMA-Factory 的其他 YAML 字段、LoRA target 命名和多模态数据映射必须以执行时锁定 revision 的真实 parser/config 为准，先做静态检查和 schema smoke，不能照搬其他版本未经验证的字段。

### 数据任务分布

200 条正式训练数据需覆盖：产品/场景描述、OCR 转写、表格提取与摘要、UI 理解/操作说明、公式转写与解释。每类保留明确 `task_type`，同时平衡短答案、结构化答案和解释性答案，避免 200 条全部来自同一模板。

额外开发/最终记录覆盖相同能力类别，使用与训练集不同的图片 SHA 和目标回答；同类任务复用受控的通用问题模板，并在 manifest 中披露。Day 25 固定十题的原文、近似改写和图片不得进入训练集。

### 数据记录合同

内部标准记录至少包含：

```text
id,image_relative_path,image_sha256,task_type,user_instruction,
target_answer,source,license,split,construction_method
```

转换为 LLaMA-Factory 格式后，消息角色、图片列表和 `<image>` 等占位标记必须与锁定版本的官方多模态 parser 完全一致；内部标准记录保留以便审计。

## 详细实施步骤

### 1. 建立训练和评测素材池

只使用自建、自有或许可明确的图片，去除真实客户、账号、定位、密钥和个人信息。先分配图片到 `train` 或 `heldout`，再生成指令，避免同图跨集合。保存来源、许可、尺寸、字节和 SHA-256。

### 2. 构造 200 条训练记录

围绕五类核心能力生成正好 200 条图文指令。每条目标回答必须与图片一致、可读、非空，并在必要时保留结构。记录构造方式和审核理由，不用乱码、空答案或重复模板凑数量。

### 3. 构造额外 20 条隔离评测

在训练前冻结开发与最终图片及其目标答案。评测图片 SHA 与全部训练图片零重叠，目标答案规范化后精确重叠为 0；同类任务沿用受控的通用 Prompt 模板，因此不把本实验表述为“新指令措辞”评测。冻结后不根据 Base/LoRA 回答换题，也不将最终评测回答用于新增训练数据。

### 4. 执行数据质量和污染门禁

检查：总数 200/20、ID 唯一、图片存在可解码、SHA 一致、角色合法、Prompt/回答非空、许可完整、重复/近重复、图片跨 split、Day 23/Day 25 评测污染和路径逃逸。任何失败记录必须修正或在运行模型前替换并重新冻结 manifest。

### 5. 转换并运行真实 schema smoke

根据锁定 LLaMA-Factory revision 的官方多模态示例生成数据和 `dataset_info.json` 注册。先用 1–4 条样例通过 LLaMA-Factory dataset parser，再调用 Qwen2-VL processor 生成张量，确认图片实际被读取、视觉 token 非空、labels 与消息角色合理。

### 6. 配置 LoRA 与冻结视觉塔

生效 YAML 至少明确多模态 SFT、LoRA、`freeze_vision_tower: true`、基座路径/revision、数据集、输出目录、seed、精度、batch、gradient accumulation、学习率、epoch、保存/日志策略和图像像素范围。

启动前打印全部可训练参数名和数量：任何视觉塔/visual 模块参数可训练都必须停止；语言模型 LoRA 参数应非空。对非 LoRA 基座参数确认 `requires_grad=False`。

### 7. 运行单步 GPU smoke

使用少量真实多模态样例运行一个 optimizer step，记录前向、反向、loss、峰值显存、return code 和 adapter 文件。核对 trainer 冻结日志、可训练参数名以及 adapter 中不存在视觉/投影键；另比较基座在加载 adapter 前后的视觉参考张量 SHA，确认 adapter 不携带视觉增量。该证据不冒充训练进程内的逐步张量快照。

### 8. 启动正式训练

只有 schema 与单步 smoke 通过后才运行 200 条正式训练。保存实际生效配置、启动命令、stdout/stderr、return code、每步或固定间隔 loss、学习率、grad norm、epoch、吞吐和峰值显存。输出目录带唯一 run ID，不能覆盖失败 run。

### 9. 验证 adapter 与模型加载

训练自然结束后检查指标有限、adapter 文件非空、配置指向正确基座。生成逐文件相对路径、大小和 SHA；在新的进程中离线加载 Base+adapter，运行一条不属于训练/评测的 smoke。若另行合并模型，adapter 与 merged 分别保留 lineage。

### 10. 运行 Base/LoRA 隔离评测

Base 与 LoRA 使用相同 20 条 held-out、相同图片文件和相同确定性生成参数。评测脚本在模型加载前逐文件核对 Day22 基座 manifest，并把基座 repository/revision/文件集合摘要、manifest SHA、评测配置 SHA、样本 SHA、adapter config/权重 SHA 和 LoRA scale 写入不可变 run spec；续跑时任一字段变化都会失败。保存两模型共 40 条原始回答；随机生成 A/B 身份映射后评分。匿名材料与盲化种子只留工程区，教师提交仅保留身份明确的复核结果和预揭盲评分哈希。

评分维度和权重固定为：

| 维度 | 权重 |
|---|---:|
| 视觉事实正确性 | 35% |
| 指令完成度 | 25% |
| 完整性 | 15% |
| 有用性 | 15% |
| 格式 | 10% |

每维使用 1–5 分，先计算每条加权分，再计算模型均分、逐题胜负和视觉幻觉率。

### 11. 判断明显提升

预注册三项门槛为：

```text
mean_gain = lora_mean - base_mean >= 0.50 / 5
lora_wins >= 12 / 20
lora_hallucination_rate <= base_hallucination_rate
```

三项必须同时满足才标记“明显提升 PASS”。若未通过，保留本 run 摘要，优先排查训练目标质量、数据污染、学习率、LoRA rank、epoch 和视觉 token 上限；20 条 held-out 保持冻结，纠偏模型对全部 20 条重跑。

### 12. 更新状态并关闭资源

训练、adapter、冻结验证和效果门槛全部有证据后才更新状态。复制并校验必要的小型日志/结果后关闭 AutoDL；模型大文件留在持久化目录，老师材料不包含关机证据。

## 实际文件结构

关键文件如下（模型权重留在 AutoDL 持久化目录）：

```text
deliverables/week5/day26/
├── README.md
├── configs/
│   ├── dataset_info.json
│   ├── qwen2vl_lora_smoke.yaml
│   ├── qwen2vl_lora_candidate_v6_grounded.yaml
│   ├── evaluation_config.json
│   └── evaluation_config_v6_final.json
└── source/
    ├── data/
    │   ├── week5_vlm_train.json
    │   ├── week5_vlm_dev.json
    │   ├── week5_vlm_final.json
    │   ├── data_manifest.json
    │   └── split_manifest.json
    ├── results/
    │   ├── day26_data_validation.json
    │   ├── trainable_parameters_summary.json
    │   ├── smoke_training_summary.json
    │   ├── training_summary_v6.json
    │   ├── adapter_manifest_v6_compact.json
    │   ├── base_lora_paired_final_v6_remediation.json
    │   ├── blind_scores_final_v6_remediation.json
    │   ├── comparison_summary_v6_remediation.json
    │   ├── CANDIDATE_SELECTION.md
    │   ├── EFFECT_COMPARISON.md
    │   └── day26_validation.json
    ├── scripts/
    │   ├── build_vlm_dataset.py
    │   ├── validate_vlm_dataset.py
    │   ├── verify_trainable_parameters.py
    │   ├── verify_adapter.py
    │   ├── run_base_lora_eval.py
    │   ├── score_base_lora.py
    │   └── validate_day26.py
    └── tests/
        └── test_day26_pipeline.py
```

完整图片集、adapter、merged 模型、checkpoint、optimizer state 和缓存根据体积与许可留在 AutoDL，不直接进入 Git。

最终目录可直接复核；验证器以 `FAIL` 退出码保留未通过质量门槛的语义，而不是把它误报成脚本故障：

```bash
python source/scripts/validate_day26.py \
  --root . \
  --output /tmp/day26_validation.json
```

## 完成门禁

- [x] 正式训练数据正好 200 条，开发/最终评测各 20 条；
- [x] 训练/评测图片 SHA-256 零重叠，Day 25 题集无污染；
- [x] ID、图片、schema、非空、许可、重复和路径检查通过；
- [x] 锁定版本 LLaMA-Factory parser 与真实 Qwen2-VL processor schema smoke 通过；
- [x] 生效配置包含 `freeze_vision_tower: true`；
- [x] 可训练参数只包含预期语言模型 LoRA，trainer 报告冻结视觉塔/投影层，adapter 无视觉或投影键；
- [x] 单步 smoke 与正式训练 return code 为 0，指标无 NaN/Inf；
- [x] Adapter 文件非空、清单与 SHA 可重算，独立离线加载通过；
- [x] 200/300 训练输入、训练配置、Day22 基座 manifest、评测配置/样本、LoRA scale、adapter 配置/权重及评分配置 SHA 交叉一致；
- [x] Base/LoRA 各 20 条原始回答齐全，生成参数一致；
- [x] 盲评维度、权重、逐题理由和身份映射完整；
- [ ] 平均提升至少 `0.50/5`（实际 `+0.0125`）；
- [x] LoRA 至少在 `12/20` 条上胜出；
- [x] LoRA 视觉幻觉率不高于 Base；
- [ ] 三项质量条件同时满足（当前总门槛 FAIL）。

任一硬门槛没有证据时，Day 26 保持未完成。

## 故障处理

- **LLaMA-Factory 不识别字段**：读取锁定 revision 的官方多模态示例和 parser，先修正 1–4 条 schema smoke，再重新转换全量；不猜测旧版本字段。
- **图片没有进入 processor**：检查图片列表、消息中的图像占位标记和相对根目录；视觉 token 为空时禁止训练。
- **视觉塔仍可训练**：停止运行，检查 `freeze_vision_tower: true`、LoRA target 和 trainable parameter names；修正后从新 run 开始。
- **CUDA OOM**：先降低图像像素上限、micro batch，增加 gradient accumulation，并使用 bf16/4-bit 可兼容路线；修改后重新做单步 smoke。
- **Loss 为 NaN/Inf**：保留失败 run，检查坏图、超长样例、精度、学习率和梯度；不能删除日志后重启同目录。
- **训练完成但质量未提升**：保留 adapter 和结果；20 条评测不变，针对数据/超参数创建新 run 并对全部 held-out 重跑。
- **LoRA 幻觉率变高**：即使均分提升也判定三项门槛失败；检查训练数据是否鼓励猜测或缺少纠错/不确定性样例。

## 下一日交接

Day 27 接收完整基座 lineage、200/20/20 数据 manifest、生效配置、训练日志、adapter 清单、Base/LoRA 原始回答、盲评汇总和失败门槛。后续若继续优化，应新增未曝光最终集，并为新 split 增加不同措辞的 Prompt bank；不能围绕当前最终集继续调参。匿名 A/B 材料、盲化种子、私有映射、模型大权重和 AutoDL 运维证据不进入教师提交。
