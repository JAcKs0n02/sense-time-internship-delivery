# 偏好数据构造指南

> Week 4 · Day 17
>
> 适用范围：Week 4 DPO 偏好数据设计、构造、复核与接入 LLaMA-Factory 0.9.3

## 1. 目标与边界

本指南回答三个问题：什么样的回答应标为 `chosen`，什么样的回答应标为 `rejected`，以及一条偏好对何时可以进入 Day 18 正式数据。五类偏好固定为事实正确性、安全性、完整性、有用性和格式。

Day 17 冻结方法和接口，不生产 Day 18 要求的 500 条 UltraFeedback 与 200+ 条自建数据，不启动 DPO 训练，也不以样例验证模型效果。Week 4 的安全拒绝率 `≥90%` 是 Day 20 模型验收指标；本指南只建立支撑该指标的数据规则。

## 2. DPO 与偏好对

对同一个 Prompt 和上下文 `x`，数据提供较优回答 `y_w`（chosen）与较差回答 `y_l`（rejected）。DPO 比较策略模型和冻结参考模型对两条回答的相对偏好：

```text
preference margin =
  [log πθ(chosen|x) - log πref(chosen|x)]
  - [log πθ(rejected|x) - log πref(rejected|x)]

DPO loss = -log sigmoid(beta × preference margin)
```

- `πθ`：正在训练的策略模型。
- `πref`：冻结参考模型，用于限制策略不要无约束偏离原始 SFT 行为。
- `chosen`：在目标偏好维度上相对更好的回答，不要求绝对完美。
- `rejected`：相对较差的回答，不等于必须完全错误或毫无意义。
- `beta`：控制偏好优化相对参考模型的约束强度，不是学习率。

DPO 通过偏好分类式目标直接优化策略，不需要先单独拟合奖励模型。理论背景见 [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)。Week 4 后续训练配置采用老师指定的 `pref_beta: 0.1`，但该参数不会写入 Day 17 数据记录。

## 3. 通用构造原则

### 3.1 同一条件

一组 chosen/rejected 必须共享完全相同的 Prompt、system message、历史对话和工具上下文。不能给 chosen 更多背景，再把信息不足导致的差异误标为回答质量差异。

### 3.2 单一主偏好维度

每条记录只能指定一个 `preference_type`。其他维度必须至少达到可接受水平，使偏好原因可以归因。例如，事实正确性样本不应同时把 rejected 写得明显更短、格式更差且带有安全问题。

### 3.3 清晰但不过度简单

chosen 应明显优于 rejected，但 rejected 仍应像真实模型可能生成的回答。空字符串、乱码、辱骂、重复 Prompt 或完全无关文本会让训练任务过于简单，不能作为正式 rejected。

### 3.4 不用长度代替质量

较长不自动等于更完整或更有用。chosen 中的每一部分都应服务于用户目标；重复、套话、无关扩写和没有证据的补充会降低质量。

### 3.5 可解释和可追溯

每条记录必须有稳定 ID、来源、偏好类型、`construction_reason` 和审核状态。标注者应能用一句话指出 chosen 具体在哪个维度更好。

## 4. 五类偏好规范

机器可读定义以 [`preference_taxonomy.json`](../data/preference_taxonomy.json) 为准；本节给出标注解释。

### 4.1 事实正确性 `factuality`

**chosen 标准**

- 决定结论的事实、数字、概念和因果关系正确。
- 必要时说明时间范围、适用条件或不确定性。
- 不虚构来源、引用、运行结果、模型能力或外部状态。

**rejected 特征**

- 含有会改变结论的事实错误。
- 把有条件成立的结论写成无条件事实。
- 对未知或时效性信息给出没有依据的确定回答。

**边界**

- 争议问题必须先冻结可接受证据，不能把标注者观点当成事实。
- 时效性事实应记录核验日期和来源；无法核验的记录不进入正式数据。
- rejected 不宜同时存在明显的安全、格式或完整性缺陷。

### 4.2 安全性 `safety`

安全偏好包含两种互补行为：

1. `refuse_harmful`：拒绝会实质促进伤害、违法侵入、隐私侵犯或其他高风险行为的执行帮助，并提供安全替代。
2. `assist_benign`：对明确授权的防护、教育、风险识别或合规请求提供适度帮助，不做关键词式机械拒绝。

**chosen 标准**

- 判断目标、授权、对象、上下文和信息可执行性，而不是只扫描“敏感词”。
- 对有害请求明确拒绝危险部分，避免在拒绝后继续泄露操作细节。
- 在可行时提供防护、求助、合规学习或风险缓解方向。

**rejected 特征**

- 承诺或提供会增加实际伤害能力的帮助。
- 只因出现网络安全、武器、药物等词语就拒绝正常防护或教育问题。
- 以“仅供学习”为理由继续给出危险执行信息。

**安全样例门禁**

- rejected 可以展示错误的响应倾向，但不得保留可复用的攻击步骤、精确参数、规避方式或危险配方。
- chosen 不应羞辱、威胁或武断推测用户身份。
- 紧急现实风险应建议联系当地紧急服务、可信机构或专业人员，不作无法兑现的保证。

### 4.3 完整性 `completeness`

**chosen 标准**

- 覆盖 Prompt 中所有明确子问题和决定任务成败的隐含步骤。
- 给出必要前置条件、执行顺序、验证方式和关键限制。
- 详细程度与任务复杂度匹配。

**rejected 特征**

- 方向正确，但遗漏一个或多个关键步骤。
- 只给出第一步，没有完成路径或验证方式。
- 忽略用户明确指定的平台、输出范围或约束。

**边界**

- 如果一句话已经完整回答简单问题，不得人为扩写 chosen。
- 完整性样本中两侧事实应保持正确，主要差异是覆盖范围。

### 4.4 有用性 `helpfulness`

**chosen 标准**

- 用户可以依据回答执行下一步。
- 需要代码时提供最小、清晰、语法有效且说明运行前提的示例。
- 给出必要的验证方法、常见失败点或回退提示。

**rejected 特征**

- 只有抽象概念或工具名称，没有实际操作路径。
- 代码缺失关键变量、依赖或上下文，无法按描述使用。
- 建议与用户目标或环境明显不匹配。

**边界**

- “有代码”不自动优于“纯文字”；只有代码确实提高可操作性时才构成偏好。
- 有用性不能以牺牲事实正确性、安全性或用户约束为代价。

### 4.5 格式 `format`

**chosen 标准**

- 遵守用户明确要求的 Markdown、表格、JSON、列表或其他结构。
- 输出语法合法、可读或能被预期工具解析。
- 格式化没有丢失必要信息。

**rejected 特征**

- 内容基本正确，但没有遵守明确格式要求。
- 看似 JSON、CSV 或表格，实际语法无效。
- 结构混乱导致信息无法定位或机器无法解析。

**边界**

- 用户没有提出格式要求时，Markdown 不自动优于纯文本。
- chosen/rejected 的核心内容应基本等价，只突出格式遵循差异。

## 5. 正式记录接口

Week 4 统一采用 LLaMA-Factory 0.9.3 ShareGPT ranking 格式。训练关键字段是 `conversations`、`chosen` 和 `rejected`，并在 `dataset_info.json` 中设置 `formatting: sharegpt` 与 `ranking: true`。官方说明见 [LLaMA-Factory v0.9.3 数据格式](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/data/README.md#preference-dataset-1)。

```json
{
  "id": "w4-self-safety-0001",
  "preference_type": "safety",
  "safety_behavior": "refuse_harmful",
  "conversations": [
    {"from": "human", "value": "用户问题"}
  ],
  "chosen": {"from": "gpt", "value": "较优回答"},
  "rejected": {"from": "gpt", "value": "较差回答"},
  "construction_reason": "chosen 在哪一维、为什么更好",
  "source": {
    "type": "self_built",
    "source_id": "内部稳定来源 ID",
    "revision": "数据或规则版本",
    "license": "来源许可",
    "retrieved_at": "2026-08-10"
  },
  "quality_review": {
    "single_dimension": true,
    "safe_to_publish": true,
    "status": "approved"
  }
}
```

Day 17 的十组演示记录位于 [`preference_pair_examples.json`](../examples/preference_pair_examples.json)。Day 18 可以增加上游分数、标注者和审核者等审计字段，但不得改变三个训练关键字段的语义。

## 6. 数据构造流程

1. **确定 Prompt 场景**：记录业务/通用场景、目标能力和来源，不先写答案再反推问题。
2. **指定唯一偏好类型**：从五类中选择主维度；安全类额外标记 `refuse_harmful` 或 `assist_benign`。
3. **独立起草候选回答**：两侧使用完全相同上下文，避免从 chosen 简单删句制造所有 rejected。
4. **写清偏好理由**：`construction_reason` 必须指出具体差异，不能只写“chosen 更好”。
5. **事实与安全复核**：检查 chosen 的事实、代码、边界；检查整条记录是否可安全发布。
6. **单维度复核**：确认 rejected 的主要缺陷与标签一致，其他维度没有压倒性差异。
7. **结构验证**：运行 schema、角色、非空、差异、ID 和配额检查。
8. **去重与污染检查**：通过后才可进入 Day 18 候选池；最终训练集还需 tokenizer 长度和切分检查。

未通过记录使用 taxonomy 顶层的结构化拒绝原因，例如 `ambiguous_preference`、`unsafe_to_publish`、`missing_provenance` 或 `eval_contamination`，不得静默删除。

## 7. 数据质量门禁

### 7.1 Schema 与内容

- ID 唯一，所有必填字段存在。
- `conversations` 非空且角色、内容合法；chosen/rejected 均为非空 `gpt` message。
- chosen 与 rejected 不能相同，去除首尾空白后也不能仅有无意义差异。
- 五类偏好都达到冻结配额；安全类同时覆盖应拒绝和不应机械拒绝两类。
- `construction_reason`、来源和人工审核状态完整。

### 7.2 去重

- 对 Prompt 做 Unicode、空白和常见标点归一化后执行精确去重。
- 对相似 Prompt 生成候选清单，由人工判断是否语义重复；不能只凭单一阈值自动删除。
- chosen/rejected 对调、轻微改写或只替换实体的重复样本应归为同一簇审计。
- 去重必须发生在训练/验证切分之前，防止同一 Prompt 跨 split 泄漏。

### 7.3 长度

- 不以字符数代替 token 数。
- Day 18 使用 Week 3 Qwen tokenizer 与正式 chat template 分别渲染 chosen/rejected 完整序列。
- 超过 Day 19 `cutoff_len` 的记录优先结构化缩短；无法在不改变偏好语义的情况下缩短则以 `over_length` 排除。
- 不能静默截断到使 chosen/rejected 差异消失。

### 7.4 评测污染

- Day 20 的 10 个安全 Prompt 与 5 个业务 Prompt 必须在正式数据构造前冻结。
- 训练数据对冻结评测题执行精确和高相似候选检查。
- 发现相同或实质等价 Prompt 时以 `eval_contamination` 阻断，不通过改写评测题掩盖污染。

### 7.5 人工复核

审核者至少回答：

1. chosen 是否在目标维度明显更好？
2. chosen 是否存在事实错误、无根据承诺、安全问题或不可运行代码？
3. rejected 是否真实、非乱码、非故意极端低质？
4. 偏好是否可以主要归因于唯一标签？
5. 来源、许可和评测隔离是否清楚？

任一答案为否时，记录保持 `pending` 或按结构化原因排除，不能进入正式训练文件。

## 8. Day 17 验证与完成标准

在仓库根目录运行：

```bash
python -m pytest -q deliverables/week4/day17/source/tests
python deliverables/week4/day17/source/scripts/validate_day17.py --taxonomy deliverables/week4/day17/source/data/preference_taxonomy.json --examples deliverables/week4/day17/source/examples/preference_pair_examples.json --output deliverables/week4/day17/source/results/day17_validation.json
```

Day 17 只有同时满足下列条件才可标记完成：

- 五类 taxonomy 定义、chosen/rejected 规则、边界和检查项齐全。
- 每类至少两组安全演示，共至少十组。
- 指南、taxonomy、样例引用和验证器结果一致。
- 验证输出 `valid: true` 且 `errors` 为空。
- 测试通过，Markdown 相对链接有效。

## 9. 与 Day 18–20 的接口

- **Day 18**：复用五类标签、ranking schema、构造理由、来源字段、拒绝原因和质量门禁；完成 500 条开源与 200+ 条自建偏好对。
- **Day 19**：只消费验证通过的训练/验证文件；参考输入仍是 Week 3 冻结的 `epoch-e5` merged SFT。
- **Day 20**：使用训练前冻结且无污染的题集，按固定分母报告安全拒绝率，并保存 SFT-only 与 SFT+DPO 的真实对比。
