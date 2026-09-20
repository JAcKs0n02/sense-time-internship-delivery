# Day 18 Submission: Preference Dataset

本目录对应老师 Day 18 的“收集开源偏好数据、自建偏好对并完成清洗合并”要求。最终数据 710 条，超过周验收最低 300 条，也满足当天 500 条 UltraFeedback 与 200+ 条自建数据的明细要求。

## Final Result

| Item | Result |
|---|---:|
| UltraFeedback | 500 |
| Self-built pairs | 210 |
| Total | 710 |
| DPO train | 639 |
| DPO validation | 71 |
| Factuality / Safety / Completeness / Helpfulness / Format | 42 each |
| Business / General self-built scenes | 105 / 105 |
| Exact prompt duplicates | 0 |
| Frozen-evaluation exact / high-similarity contamination | 0 / 0 |
| Records over 2048 tokens | 0 |
| LLaMA-Factory 0.9.3 schema smoke | PASS |
| Combined validation | `valid: true` |

## Required Deliverable

- [`Merged_Preference_Dataset.json`](Merged_Preference_Dataset.json)：710 条完整偏好数据，ShareGPT ranking 格式。
- [`Dataset_Statistics.json`](Dataset_Statistics.json)：来源、类别、场景和 split 统计及最终文件哈希。
- [`Data_Validation.json`](Data_Validation.json)：本地结构、远端哈希、真实 tokenizer 和 LLaMA-Factory 的联合验收。
- [`UltraFeedback_Provenance.json`](UltraFeedback_Provenance.json)：上游固定 revision、许可、原始文件哈希、排除计数和确定性抽样规则。

## Training Inputs for Day 19

- [`DPO_Train.json`](DPO_Train.json)：639 条；SHA-256 `88c6a176eda7df50f5ffb0e01e78610c99178e71e14518d82b6c0bd20a50c320`。
- [`DPO_Validation.json`](DPO_Validation.json)：71 条；SHA-256 `5626226d8d0aff49cec6119de63720692cb3f3f27091bf94332524cd83a53372`。
- [`Dataset_Info.json`](Dataset_Info.json)：LLaMA-Factory ranking 数据注册。

初始 710 条数据完成老师 Day 18 交付后，为修正最终模型的 Rewards 方向，额外加入 160 条中文安全偏好并重新做确定性切分；最终选中模型实际使用以下 870 条纠偏输入：

- [`Corrective_DPO_Train.json`](Corrective_DPO_Train.json)：783 条；SHA-256 `99f90060ba8b0904e561e14296722e46e1aff55f8f61ab16a0c173a720e5cfcc`。
- [`Corrective_DPO_Validation.json`](Corrective_DPO_Validation.json)：87 条；SHA-256 `8af26c924daee6a47daf61f44ee0d17b082d6ec55cd52e7bdbc761a7b4f84e60`。
- [`Corrective_Dataset_Info.json`](Corrective_Dataset_Info.json)：纠偏 ranking 数据注册。
- [`Corrective_Data_Validation.json`](Corrective_Data_Validation.json)：870/783/87/35 数量与老师题 exact/near 污染均为 0 的验证结果。

这里将“老师要求的 Day 18 原始 710 条交付”和“最终纠偏模型实际使用的 870 条输入”同时保留，避免模型 lineage 与提交数据不一致。

## Audit Evidence

- [`Preference_Split.csv`](Preference_Split.csv)：710 条逐 ID 的 train/validation 归属。
- [`Self_Built_Review.csv`](Self_Built_Review.csv)：210 条自建数据逐条复核记录；`reviewer_kind=codex_review`，不冒充人工双标。
- [`Length_Statistics.json`](Length_Statistics.json)：真实 Week 3 Qwen tokenizer 统计，超长数量为 0。
- [`LLaMAFactory_Schema_Smoke.json`](LLaMAFactory_Schema_Smoke.json)：LLaMA-Factory 0.9.3 成功加载 8 条 ranking 样本并构造双侧 labels。

## Notes

- UltraFeedback 使用 `allenai/ultrafeedback_binarized_cleaned` 的固定 revision `f304ce59671d155273c861746581faa354d8a9af`，许可为 MIT。
- 外部偏好统一标记 `mixed_external`，因为上游分数可能同时反映多个质量维度。
- 有 322 对自建 Prompt 因受控模板产生高相似候选；已逐对审计为相同模板下的不同主题，未自动删除，也没有待审候选。
- Day 18 只验证数据与训练入口可用，不表示 DPO 模型已经训练或质量已经提升。

本教师提交目录已经包含完整数据、切分、统计、来源、逐条自建复核、长度统计和 schema smoke 结果，可独立审阅。
