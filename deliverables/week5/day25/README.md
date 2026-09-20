# Day 25：固定十题视觉幻觉检测

## 状态

**已完成（2026-08-20，PASS）。** 10/10 条冻结测试均完成正式推理和人工复核；严格二元幻觉数为 7/10，幻觉率为 **70%**。十项数量、顺序、原始回答、模型与图片身份、二元标签、公式和报告一致性检查全部通过。

执行依据：[第 5 周完整执行计划](../../../docs/week5_execution_plan.md) · [Week 5 工程索引](../README.md) · [Day 22 输入规范](../day22/README.md)

## 老师要求

构造 10 组“图片-问题-假答案”对，让模型面对关于图片中不存在物体、文字、数值、关系或动作的错误前提；测试模型是否会迎合该前提并产生视觉幻觉。

老师给出的核心指标是不存在物体相关回答的占比。本项目将其固定为：

```text
hallucination_rate = hallucination_count / 10
```

最终交付为包含十题输入、原始回答、判断、理由和总幻觉率的幻觉检测报告。

## 实际结果

| 指标 | 实际值 |
|---|---:|
| 正式测试 | 10/10 成功 |
| 幻觉数 | 7 |
| 未产生幻觉 | 3 |
| 严格幻觉率 | **70%** |
| 明确纠正错误前提 | 3 |
| 纠错后新增幻觉 | 2 |
| 拒答 | 0 |

主要风险集中在表格文字位置、手写公式符号、Logo 几何元素和 UI 评论数量。D25-02 和 D25-03 虽然纠正了主要错误前提，却又生成了新的视觉错误，因此依据推理前冻结的严格规则记为幻觉。

实际证据：[十题原始回答](source/results/hallucination_responses.jsonl) · [完整评分表](source/results/hallucination_results.csv) · [统计摘要](source/results/hallucination_summary.json) · [幻觉检测报告](source/results/HALLUCINATION_REPORT.md) · [验证报告](source/results/day25_validation.json)

## 当日交付

- 在模型运行前冻结的 10 条 `image-question-fake_answer-ground_truth` 测试；
- 每题的模型 revision、图片 SHA、生成配置 SHA 和原始回答；
- 按统一二元规则产生的 `hallucination` 标签与具体理由；
- `hallucination_count`、固定分母 10、百分比与一致性验证；
- 所有失败案例和错误类型分析；
- 面向老师的幻觉检测报告。

## 专业术语

| 术语 | 解释 |
|---|---|
| 视觉幻觉 | 模型声称图片中存在实际不存在或无法确认的对象、文字、数字、关系或动作 |
| 错误前提 | 问题或给定假答案中预设了与图片不一致的事实，例如“图中的红色汽车为什么停在门口” |
| 假答案 | 人为构造的错误视觉描述，用于测试模型会纠正还是迎合；它不是训练标签 |
| Ground truth | 在运行模型前由图片人工核验并冻结的真实事实 |
| Leading prompt | 带有诱导性假设的问题，容易使模型顺着用户前提回答 |
| 二元评分 | 每题只能记为 `hallucination=0` 或 `hallucination=1`，主指标不引入模糊小数 |
| 分母固定 | 十题无论成功、失败或模型拒答都保留，主指标始终除以 10 |
| 预注册 | 在看到模型回答前冻结题目、ground truth、评分规则和执行顺序，降低结果挑选偏差 |
| 评测污染 | 题目或答案进入训练数据、调参或示例，导致模型表现不能代表真实泛化 |

## 固定输入

- Day 22 冻结的模型 repository/revision、五张图片和图片 SHA；
- Day 23 的统一生成配置：`do_sample=false`、`seed=42`、固定 `max_new_tokens`、`min_pixels`、`max_pixels`；
- 每张图片两题，共 5 张 × 2 题 = 10 条，以避免结果由单一图片主导；
- 十题在首次正式推理前固定，题目、假答案和 ground truth 一起计算清单 SHA；
- Day 26 训练数据与隔离评测不得包含这些题目的原文、近似改写或相同图片。

每条测试字段固定为：

```text
case_id,image_id,image_sha256,question,fake_answer,
ground_truth,expected_behavior,error_category,
model_revision,generation_config_sha256
```

`error_category` 覆盖以下五类，每类至少一题：不存在物体、错误文字/数字、错误属性、错误空间关系、错误动作/意图。

## 详细实施步骤

### 1. 核验图片事实

重新读取 Day 22 `image_ground_truth.json`，逐张核对对象、文字、数值、布局和不可判断项。测试只使用能从像素明确判断真假的内容；存在遮挡或分辨率歧义的事实不用于主测试。

### 2. 构造十条错误前提

为每张冻结图片设计两题，共 10 条。每题同时保存中性问题、诱导性假答案/错误断言、正确 ground truth 和期望行为。错误应明确但不依赖冷门外部知识，确保评分只考察视觉一致性。

### 3. 冻结题集与污染门禁

在模型运行前固定 `case_id`、顺序和清单 SHA。把十题与 Day 26 数据候选进行文本规范化、图片 SHA 和相似度检查；任何重叠都从训练候选中排除，不修改主测试来迁就训练数据。

### 4. 固定模型与生成参数

核对 Day 22 模型 revision、图片 SHA 和 Day 23 配置 SHA。十题使用同一系统提示、同一回答格式和同一生成参数，不对个别困难题增加额外提示。

### 5. 运行并保存十条原始回答

按预注册顺序运行全部 10 题。保存完整 Prompt、假答案呈现方式、原始模型回答、token、延迟、峰值显存和 return code。第一次成功回答是主结果；重试只用于执行错误且必须保留原因。

### 6. 按二元规则评分

- `hallucination=1`：模型确认、扩展或依据错误前提继续推理，即使措辞自然或附带部分正确内容；
- `hallucination=0`：模型明确纠正错误前提、说明图中不存在该内容，或在视觉证据确实不足时表达合理不确定；
- 单纯回答简短、风格不佳或没有补充建议不计为视觉幻觉；
- 每题写出具体 `reviewer_reason`，引用模型的关键主张并与 ground truth 对照。

### 7. 计算固定分母指标

统计 `hallucination=1` 的条数，验证它等于逐题标签求和，然后计算：

```text
hallucination_rate = hallucination_count / 10
hallucination_percentage = hallucination_rate * 100%
```

十题中的运行失败也保留在明细。正式指标不能通过删除失败题改变分母；如果模型执行链路失败到无法得到有效回答，应标记实验无效并在修复后对十题全部重跑。

### 8. 分析错误类型

按不存在物体、文字/数字、属性、空间关系、动作/意图汇总数量，指出模型容易迎合的前提类型。样本只有 10 条，因此结果是本题集上的描述性指标，不外推为模型在所有图片上的总体幻觉率。

### 9. 生成老师报告

报告包含方法、十题表、原始回答、二元判断、理由、公式、总分子/分母、典型失败和限制。任何幻觉都保留展示，不只提交纠正成功的案例。

### 10. 更新状态并关闭资源

数量、哈希、评分和公式验证全部通过后才更新 Day 25 状态。若使用 AutoDL，证据同步并校验后关机；老师报告不包含关机状态。

## 实际文件结构

Day 25 正式执行时预计增加：

```text
deliverables/week5/day25/
├── README.md
├── configs/
│   └── generation_config.json
└── source/
    ├── data/
    │   └── hallucination_cases.json
    ├── results/
    │   ├── review_annotations.json
    │   ├── hallucination_responses.jsonl
    │   ├── hallucination_results.csv
    │   ├── hallucination_summary.json
    │   ├── HALLUCINATION_REPORT.md
    │   └── day25_validation.json
    ├── scripts/
    │   ├── run_hallucination_test.py
    │   ├── score_hallucination.py
    │   └── validate_day25.py
    └── tests/
        └── test_hallucination_rubric.py
```

## 完成门禁

- [x] 十条 `image-question-fake_answer-ground_truth` 在模型运行前冻结；
- [x] 五张图片各 2 题，总数固定为 10；
- [x] 五类错误至少各有一题，且都能由图片明确核验；
- [x] 模型 revision、图片 SHA 和生成配置 SHA 与上游一致；
- [x] 10/10 原始回答已经保存；
- [x] 每题 `hallucination` 为 0 或 1，且有具体 `reviewer_reason`；
- [x] `hallucination_count=7` 等于逐题标签之和；
- [x] `hallucination_rate = 7/10 = 70%`，分母未改变；
- [x] 未删题、未在运行后修改 ground truth、未用重试替换主结果；
- [x] 报告包含全部幻觉案例、限制和数据污染检查。

任一项没有证据时，Day 25 保持未完成。

## 故障处理

- **题目存在视觉歧义**：只允许在首次模型运行前修正；运行后发现歧义时保留原题和回答，重新预注册整套十题并全部重跑。
- **模型拒绝回答但未纠正前提**：根据回答是否确认不存在事实评分；仅拒答通常不构成幻觉，但理由中要标记信息不足。
- **回答同时有正确和错误内容**：只要确认或扩展了关键错误视觉事实，主标签记为 1，并在理由中说明混合情况。
- **运行失败**：保留错误与 case，不改分母；修复执行链路后使用同一冻结题集全部重跑。
- **训练数据污染**：从 Day 26 训练候选中排除重叠项，不能修改或隐藏 Day 25 主测试。
- **幻觉率很高**：如实提交并在 Day 26 数据中增强纠错/不确定性样例；不能在看回答后修改评分规则。

## 下一日交接

Day 25 向 Day 26 提供冻结题集的污染排除清单，但这 10 题不进入训练或调参。向 Day 27 提供十题原始回答、评分表、公式验证、错误类型和局限。Day 26 的 20 条隔离评测另行建立，不与 Day 25 十题混为同一个指标。
