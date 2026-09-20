# Week 2 Day 7：清洗 Pipeline 开发

Day 7 已完成。`clean_pipeline.py` 已在 AutoDL 的 `llm_exp` 环境中使用
Qwen2.5-7B-Instruct 的真实 tokenizer 对 Day 6 的 5000 条语义样本执行完整
清洗。最终保留 4999 条，并同时生成一一对应的标准 Alpaca 和 ShareGPT
数据；所有最终样本套用 Qwen chat template 后均不超过 2048 tokens。

正式验证结果为 `valid: true`，20 项测试全部通过，完整数据第二次复跑的
12 项语义产物逐字节一致。

## 1. 老师要求与完成情况

| 老师要求 | 实际结果 | 证据 |
|---|---|---|
| 去除 HTML 标签 | 72 条样本发生 HTML/entity 清理 | `source/results/cleaning_stats.json`、`cleaning_audit.jsonl` |
| 去除不可见控制字符 | 27 条样本发生控制字符清理 | `source/results/cleaning_stats.json`、`cleaning_audit.jsonl` |
| 过长截断（>2048 tokens） | 276 条超过上限，276 条完成结构化截断；最终最大长度 2048 | `source/results/cleaning_stats.json`、`day7_validation.json` |
| 空值过滤 | 初始和清洗后空值均为 0；规则及异常路径已由测试覆盖 | `source/tests/test_clean_pipeline.py` |
| 模糊去重 | 实现 64-bit SimHash；真实候选经 Jaccard 防误删校准后无确认的模糊重复 | `simhash_calibration.csv`、`duplicate_pairs.csv` |
| 清洗前后数量变化 | 5000 → 4999，删除 1 条精确重复 | `cleaning_stats.json`、`cleaning_counts.png` |
| 长度分布图 | 已生成清洗前、清洗后截断前、最终三阶段图 | `evidence/length_distribution.png` |
| 清洗后数据集 ≥1500 条 | 4999 条，Alpaca 与 ShareGPT 各 4999 条 | `data/`、`day7_validation.json` |

## 2. 实际统计

### 2.1 数量变化

| 阶段或原因 | 数量 |
|---|---:|
| 原始输入 | 5000 |
| 结构错误删除 | 0 |
| 初次空值删除 | 0 |
| HTML/entity 修改 | 72 |
| 控制字符修改 | 27 |
| 空白规范化修改 | 1127 |
| 清洗后二次空值删除 | 0 |
| 超过 2048 tokens | 276 |
| 完成截断 | 276 |
| 精确重复删除 | 1 |
| 模糊重复删除 | 0 |
| 最终保留 | 4999 |

HTML、控制字符、空白和截断是“修改计数”，不是删除计数。因此数量守恒关系为：

```text
5000 - 0 - 0 - 0 - 1 - 0 = 4999
```

### 2.2 Qwen chat-template token 长度

| 阶段 | min | P25 | median | mean | P75 | P90 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 清洗前 | 40 | 89 | 175 | 539.761 | 337 | 1114.1 | 2301.55 | 6754.2 | 29343 |
| 清洗后、截断前 | 40 | 89 | 175 | 537.817 | 335 | 1114.1 | 2288.15 | 6754.2 | 29343 |
| 最终保留 | 40 | 89 | 175 | 365.411 | 335 | 1096.6 | 1716 | 2048 | 2048 |

这里的长度不是字符数。脚本使用与 Day 8 训练相同的 Qwen2.5 tokenizer 和
`apply_chat_template(..., add_generation_prompt=False)` 计算模型实际可见长度。

## 3. Pipeline 顺序

正式顺序固定为：

1. 逐行解析 JSONL 并校验 ShareGPT 结构、字段类型和角色顺序；
2. 初次空值过滤；
3. 用标准库 `HTMLParser` 去除已知 HTML、script/style 和注释，并解码 entity；
4. 删除不应进入训练文本的 C0/C1、零宽和格式控制字符；
5. 规范化换行、Tab、行尾空白和过多空行；
6. 清洗后再次检查空值；
7. 用 Qwen chat template 统计长度，并对超过 2048 tokens 的记录结构化截断；
8. 对 canonical conversation 的 SHA-256 做精确去重；
9. 用字符 3-gram 的 64-bit SimHash 召回近重复候选，再用角色级 Jaccard 门槛确认；
10. 从同一批保留记录生成 Alpaca 和 ShareGPT 两种格式并执行最终验证。

多轮截断先移除最旧的完整 user-assistant 对，始终保留最后一个监督目标。
如果最后一对本身仍超长，再在消息 token 边界内缩短内容，并重新套用 chat
template 验证，直至长度不超过 2048。

## 4. SimHash 校准与“模糊删除为 0”

本实现没有用全量两两比较，也没有仅凭低 Hamming distance 直接删除：

- 规范文本生成字符 3-gram；
- 每个特征使用 SHA-256 派生稳定 64-bit 哈希；
- 加权投票生成 SimHash；
- 使用 4 个 16-bit band 召回候选；
- 正式 Hamming distance 阈值为 `3`；
- 整体字符 3-gram Jaccard 至少为 `0.85`；
- user 和 assistant 两侧 Jaccard 的较小值至少为 `0.90`。

距离 0–6 共导出 77 对真实候选。最高整体 Jaccard 为 `0.971034`，但最高
双角色最小 Jaccard 仅为 `0.811321`，所以 77 对都判为
`review_nonduplicate`。其中包括：

- SimHash 相同但语义完全不同的 ShareGPT 碰撞；
- 共享同一法律案情、但问题和答案不同的 COIG 样本。

这两类真实误判都已固化为回归测试。`duplicate_pairs.csv` 中唯一删除记录的
原因是 `exact`；`dedup_clusters.jsonl` 为空，表示正式数据没有确认的模糊重复
簇。没有为了制造删除数量而降低阈值。

## 5. 文件结构

```text
deliverables/week2/day7/
├── README.md
├── clean_pipeline.py
├── data/
│   ├── week2_clean_alpaca.jsonl
│   └── week2_clean_sharegpt.jsonl
├── evidence/
│   ├── length_distribution.png
│   └── cleaning_counts.png
└── source/
    ├── results/
    │   ├── cleaning_stats.json
    │   ├── cleaning_stats.csv
    │   ├── cleaning_audit.jsonl
    │   ├── cleaned_provenance.jsonl
    │   ├── simhash_calibration.csv
    │   ├── duplicate_pairs.csv
    │   ├── dedup_clusters.jsonl
    │   ├── day7_clean_pipeline.txt
    │   ├── day7_pytest.txt
    │   ├── determinism_check.txt
    │   ├── day7_validation.json
    │   └── day7_files.sha256
    └── tests/
        └── test_clean_pipeline.py
```

主要入口：

- [`clean_pipeline.py`](clean_pipeline.py)：可独立运行的正式 Pipeline。
- [`week2_clean_alpaca.jsonl`](data/week2_clean_alpaca.jsonl)：Day 8 可注册的
  Alpaca 训练候选数据。
- [`week2_clean_sharegpt.jsonl`](data/week2_clean_sharegpt.jsonl)：与上面
  同一批 4999 条样本的 ShareGPT 等价表示，不能与 Alpaca 版叠加训练。
- [`cleaning_audit.jsonl`](source/results/cleaning_audit.jsonl)：5000 条输入的
  逐条清洗、长度、截断和保留状态审计。
- [`cleaned_provenance.jsonl`](source/results/cleaned_provenance.jsonl)：4999 条
  最终样本回溯 Day 6 来源的血缘映射。
- [`day7_validation.json`](source/results/day7_validation.json)：正式完整数据
  验证结果。

## 6. AutoDL 正式复现命令

以下命令在 AutoDL 的 `/root/autodl-tmp/qwen25-week2` 执行，只加载 tokenizer，
不加载 7B 模型权重：

```bash
MODEL_DIR=/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct \
conda run -n llm_exp python deliverables/week2/day7/clean_pipeline.py \
  --input deliverables/week2/day6/source/data/formatted/week2_5k_sharegpt.jsonl \
  --provenance deliverables/week2/day6/source/data/interim/day6_provenance.jsonl \
  --output-alpaca deliverables/week2/day7/data/week2_clean_alpaca.jsonl \
  --output-sharegpt deliverables/week2/day7/data/week2_clean_sharegpt.jsonl \
  --tokenizer "$MODEL_DIR" \
  --max-tokens 2048 \
  --simhash-threshold 3 \
  --min-jaccard 0.85 \
  --min-role-jaccard 0.90 \
  --seed 42 \
  --results-dir deliverables/week2/day7/source/results \
  --evidence-dir deliverables/week2/day7/evidence \
  2>&1 | tee deliverables/week2/day7/source/results/day7_clean_pipeline.txt
```

测试命令：

```bash
MODEL_DIR=/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct \
conda run -n llm_exp pytest -q \
  deliverables/week2/day7/source/tests/test_clean_pipeline.py
```

## 7. 最终核对

- `clean_pipeline.py --help` 可直接运行。
- HTML、entity、script/style、控制字符、空值和错误结构均有测试。
- 276 条超长记录全部完成结构化截断。
- 最终 4999 条均不超过 2048 Qwen chat-template tokens。
- SimHash、banding、Hamming distance、Jaccard 防误删和真实碰撞均有测试。
- Alpaca 与 ShareGPT 两份输出各 4999 条且 schema 正确。
- 统计数量守恒，两个 matplotlib 图表已人工检查。
- 正式验证为 `valid: true`、`errors: []`。
- AutoDL 最终测试为 20/20 通过。
- 第二次完整运行的 12 项语义产物全部逐字节一致。
