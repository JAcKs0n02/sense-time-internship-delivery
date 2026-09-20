# Day 18：开源偏好数据与自建偏好对

Day 18 已完成。最终数据共 710 条：固定 revision 的 UltraFeedback 500 条，自建偏好对 210 条；按 `seed=42` 分层切分为训练集 639 条、验证集 71 条。完整结构、本地语义门禁、真实 Qwen tokenizer 长度统计和 LLaMA-Factory 0.9.3 ranking schema smoke 均通过，AutoDL 实例已确认关机。

## 需求映射

| 老师要求 | 实际完成 | 证据 |
|---|---:|---|
| 收集约 500 条 UltraFeedback | 500 | [`ultrafeedback_manifest.json`](source/manifests/ultrafeedback_manifest.json) |
| 自建不少于 200 条偏好对 | 210 | [`self_built_preferences.json`](source/data/raw/self_built_preferences.json) |
| 覆盖事实、安全、完整、有用、格式 | 每类 42 | [`preference_stats.json`](source/results/preference_stats.json) |
| 合并为 JSON 偏好数据 | 710 | [`week4_preferences_full.json`](source/data/final/week4_preferences_full.json) |
| 可直接供 DPO 使用 | ShareGPT ranking；639/71 split | [`dataset_info_week4.json`](source/manifests/dataset_info_week4.json) |
| 清洗、去重、长度与格式检查 | 全部门禁通过 | [`day18_combined_validation.json`](source/results/day18_combined_validation.json) |

## 数据构造结果

### UltraFeedback

- 数据集：`allenai/ultrafeedback_binarized_cleaned`。
- revision：`f304ce59671d155273c861746581faa354d8a9af`。
- split：`train_prefs`；许可：MIT。
- 上游 parquet：60,829 条、219,203,601 bytes，SHA-256 为 `c2414c7870cf9db881ae21008dd7b72ba40d40807a5511021e7ae853e577543c`。
- 对角色、上下文、分数、空值和 chosen/rejected 差异进行门禁后，按 `SHA256("42:ultrafeedback:<prompt_id>")` 升序确定性选择。
- 内容复核额外排除 3 条：1 条不适合公开、1 条偏好关系含糊、1 条真实 tokenizer 序列超过 2048；随后按冻结排序补入下一条合格记录，最终仍为 500 条。
- 第二次独立目录复跑得到相同文件 SHA-256：`03fbbbf93475b363d43b167d3db642f551cae2823506590f77d3b7fcb46ced38`。

UltraFeedback 的单条偏好可能同时受到多个质量因素影响，因此统一标记为 `mixed_external`，不虚构成单一人工标注维度。

### 自建偏好对

- 共 210 条，事实正确性、安全性、完整性、有用性和格式各 42 条。
- 每类业务场景 21 条、通用场景 21 条；全体业务/通用各 105 条。
- 安全类包含 21 条应拒绝的有害请求，以及 21 条不应机械拒绝的良性安全请求。
- 每条保存 `construction_reason` 和逐条 review 结果。
- `reviewer_kind` 为 `codex_review`，表示本轮由 Codex 按冻结规则复核，不冒充真实人工双标。

## 清洗、去重与污染检查

- 710 个 ID 全部唯一，chosen/rejected 非空且不同，ShareGPT 角色顺序合法。
- Prompt 精确重复组为 0。
- 保守相似审计产生 322 对候选，均来自自建受控模板；逐对复核后保留为“模板相同、主题不同”，没有遗留 `review_required`。
- 冻结的 10 个安全题和 5 个业务题在训练数据中的精确污染为 0，高相似污染为 0。
- 分层切分后 train 与 validation ID 无交叉，联合集合与完整 710 条一致。

冻结题集只用于 Day 20 评测，不进入训练数据。

## 真实 tokenizer 与 LLaMA-Factory 验证

AutoDL 实例 `78fb4ea487-e700080a` 使用 Week 3 冻结模型：

```text
/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged
```

真实 Qwen chat template 对 710 条完整统计：

| 序列 | 最大值 | P95 | P99 |
|---|---:|---:|---:|
| chosen 完整序列 | 1738 | 955 | 1365 |
| rejected 完整序列 | 1670 | 935 | 1403 |

正式 `cutoff_len=2048` 下超长记录为 0。LLaMA-Factory 0.9.3 用 `template=qwen`、逻辑 `stage=dpo` 加载 8 条 smoke 样本，成功产生：

- `chosen_input_ids` / `rejected_input_ids`；
- 两侧 `attention_mask`；
- 两侧 `labels`。

远端 train/validation SHA-256 与仓库最终文件完全一致。实例最后在 AutoDL 控制台显示“已关机”，见 [`autodl_shutdown_evidence.json`](source/results/autodl_shutdown_evidence.json)。

## Day 19 唯一输入

| 用途 | 文件 | 数量 | SHA-256 |
|---|---|---:|---|
| DPO train | [`week4_dpo_train.json`](source/data/final/week4_dpo_train.json) | 639 | `88c6a176eda7df50f5ffb0e01e78610c99178e71e14518d82b6c0bd20a50c320` |
| DPO validation | [`week4_dpo_validation.json`](source/data/final/week4_dpo_validation.json) | 71 | `5626226d8d0aff49cec6119de63720692cb3f3f27091bf94332524cd83a53372` |

Day 19 不得重新抽样或临时换数据；如文件哈希变化，必须重新执行 Day 18 全部门禁和远端 schema smoke。

## 复现与验证

```bash
pytest -q deliverables/week4/day18/source/tests

python deliverables/week4/day18/source/scripts/validate_week4_dataset.py \
  --source-root deliverables/week4/day18/source
```

主要脚本：

- [`collect_ultrafeedback.py`](source/scripts/collect_ultrafeedback.py)：固定 revision 收集、过滤和确定性抽样。
- [`build_self_built_preferences.py`](source/scripts/build_self_built_preferences.py)：构造 210 条配额冻结的自建数据。
- [`build_week4_dataset.py`](source/scripts/build_week4_dataset.py)：合并、去重、污染检查和分层切分。
- [`remote_validate_day18.py`](source/scripts/remote_validate_day18.py)：真实 tokenizer 与 LLaMA-Factory 远端验收。
- [`validate_week4_dataset.py`](source/scripts/validate_week4_dataset.py)：联合验证本地、远端、哈希和关机证据。

## 边界与限制

- UltraFeedback 的上游评分代表外部模型评审结果，不等同于本项目人工重新标注。
- 322 对相似候选反映自建模板复用；已按主题逐对保留，但 Day 19 训练后仍不得把模板规律误解为真实分布覆盖。
- Day 18 只证明数据与加载管线可用，不证明 DPO 会改善模型；效果必须由 Day 20 冻结题集对比决定。
