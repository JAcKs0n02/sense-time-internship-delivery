# Day 7 数据清洗 Pipeline 设计

## 1. 目标与边界

Day 7 只完成老师要求的数据清洗，不启动模型训练。输入是 Day 6 固定 revision、
固定 seed 抽取并转换得到的 5,000 个语义样本；输出是可供 Day 8 SFT 使用的清洗
数据、独立运行脚本、数量与长度统计以及 matplotlib 图表。

必须满足：

- `clean_pipeline.py` 能通过 CLI 独立运行；
- 去除 HTML 标签和不应进入训练文本的不可见控制字符；
- 过滤空值和错误结构；
- 使用 Qwen2.5 训练 tokenizer 实测，最终样本不超过 2,048 tokens；
- 实现跨三个来源的 64-bit SimHash 模糊去重；
- 清洗后语义样本不少于 1,500 条；
- 提交清洗前后数据量、长度统计和图表；
- Alpaca 与 ShareGPT 两种输出由同一批保留样本生成，语义上保持一一对应。

Day 7 不重新下载数据集、不覆盖 Day 6 原始归档、不加载 7B 模型权重，也不把
Alpaca 和 ShareGPT 两份等价表示重复计入训练数据。

## 2. 执行环境和数据流

清洗、测试、token 统计、去重和绘图全部在 AutoDL 数据盘执行。AutoDL 已存在：

- `/root/autodl-tmp/conda/envs/llm_exp`
- Transformers 4.50.0
- Matplotlib 3.10.9
- `/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct`

本地仓库只用于版本控制、最终归档和提交。Day 6 已提交数据先同步到 AutoDL，
远端按清单复核 SHA-256、来源数量和总数后才开始 Day 7。

正式数据流：

```text
Day 6 ShareGPT + provenance
          |
          v
canonical conversations（唯一语义样本）
          |
          v
校验 -> 清洗 -> token 截断 -> 精确去重 -> SimHash 去重
          |
          +------------------+
          |                  |
          v                  v
clean Alpaca           clean ShareGPT
```

清洗只执行一次。双格式输出不能各自独立去重，否则可能保留不同样本。

## 3. 固定清洗顺序

1. 解析 JSON/JSONL 并校验顶层结构。
2. 校验字段类型、角色名称和 user-assistant 配对。
3. 初次空值过滤。
4. HTML 标签移除和 HTML entity 解码。
5. 不可见控制字符清理。
6. 空白规范化。
7. 清洗后二次空值过滤。
8. 使用 Qwen2.5 tokenizer 统计模板化长度并执行结构化截断。
9. 使用 canonical text 的 SHA-256 执行精确去重。
10. 使用 64-bit SimHash 执行模糊去重。
11. 生成两种格式并执行最终 schema、长度、数量和确定性验证。

每条输入都写入 `cleaning_audit.jsonl`，记录样本 ID、来源、是否保留、过滤或
去重原因、修改类型、三个阶段的 token 长度及截断细节。阶段统计必须守恒；
HTML、控制字符和截断属于修改计数，不计为删除。

## 4. HTML、Unicode 与空白策略

HTML 使用 Python 标准库 `html.parser.HTMLParser`，不使用宽泛正则：

- 已知行内标签只去除标签本身；
- `br`、段落、标题、列表、表格单元等块边界转换为换行；
- `script` 和 `style` 标签及其内容删除；
- HTML entity 解码；
- 未识别的尖括号内容按普通文本保留，避免破坏代码和比较表达式；
- 最终空白规范化避免标签两侧文本粘连。

Unicode 规则：

- 先把 CRLF/CR 统一为 LF；
- 删除除 LF、Tab 外不应进入文本的 C0/C1 控制字符；
- 删除 U+200B、U+200C、BOM、word joiner 和双向格式控制符；
- 保留 U+200D，防止破坏组合 emoji；
- 保留中文、emoji、组合字符、数学符号和正常 Unicode；
- Tab 展开为固定空格，保留代码缩进；
- 去除行尾空白、首尾空行并压缩过多连续空行，不折叠行内正常空格。

空值规则：

- Alpaca 的 `instruction` 和 `output` 必须非空，`input` 可以是空字符串；
- ShareGPT 至少包含一个非空、顺序正确的 user-assistant 对；
- `null`、错误类型、空列表、只有标签或控制字符的伪非空样本被分类拒绝；
- 不把错误类型静默转换为字符串。

## 5. 2,048-token 长度和截断

使用：

```python
AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
)
```

长度包含训练时实际存在的角色、模板和特殊 token。记录清洗前、清洗后截断前
和最终长度。

结构化截断规则：

1. 多轮样本从最旧的完整 user-assistant 对开始移除；
2. 始终保留最后一个有效 user-assistant 对；
3. 最小结构仍超长时，只在消息内容的 token ID 边界截断；
4. 优先缩短当前最长内容，同时保证 user 和 assistant 内容均非空；
5. 解码后重新套用 chat template，通过二分搜索收敛到上限；
6. 最终重新编码并断言长度不超过 2,048；
7. 审计中记录被移除轮次、处理字段、原长度、最终长度和删除 token 数。

该口径与 Day 8 的 `cutoff_len: 2048` 保持一致。

## 6. 精确去重与 SimHash

用于跨格式、跨来源比较的 canonical text 是清洗后的全部对话，使用明确的
`user:` 和 `assistant:` 角色前缀串联；provenance 不参与相似度。

精确去重先对 canonical text 计算 SHA-256。

模糊去重自行实现：

- 特征为规范文本的字符 3-gram；
- 每个特征使用 SHA-256 的前 64 bit 作为稳定哈希；
- 特征频次参与 64 个 bit 位的加权投票；
- 生成 64-bit SimHash；
- 切分为 4 个 16-bit band，只有共享 band 的记录才成为候选；
- 计算候选的 Hamming distance；
- 用整体字符 3-gram Jaccard 以及 user/assistant 角色级 Jaccard 作为确认门槛，
  避免 SimHash 碰撞和“共享长上下文、但任务或答案不同”的误删；
- 很短文本跳过模糊去重，避免“好的”“谢谢”等误杀；
- 正式阈值从距离 0–6 的真实边界候选及构造扰动样本中校准；
- 校准优先保证 precision，阈值和判定证据写入
  `simhash_calibration.csv`；
- 只有 Hamming distance 和三项 Jaccard 门槛同时满足时，才使用 union-find
  形成确定性重复簇。

SimHash 负责候选召回，Jaccard 负责精确确认，两者共同组成一个模糊去重
判定器，而不是两套独立执行并分别删数据的规则。

重复簇代表样本的排序优先级为：未截断、结构完整、保留信息更多、来源名称和
原始索引。输出 `duplicate_pairs.csv` 和 `dedup_clusters.jsonl`，能够从每个
删除样本追溯到保留样本和 Hamming distance。

## 7. 测试策略

严格使用 RED-GREEN-REFACTOR：

- 先编写一个行为测试并运行，确认因为目标功能缺失而失败；
- 只写使该测试通过的最小实现；
- 运行测试确认通过后再重构；
- 对下一个行为重复此过程。

测试至少覆盖：

- HTML 标签、entity、块边界及 script/style；
- 普通尖括号代码不被误删；
- C0/C1、零宽字符、emoji、组合字符和代码缩进；
- Alpaca 三字段空值和错误类型；
- ShareGPT 空消息、角色错序、多轮对话；
- 模板化长度恰好 2,048、2,049 和远大于上限；
- 多轮删除和单轮 token 边界截断；
- 精确重复；
- 轻微空白、标点和措辞变化的近重复；
- 主题相似但答案不同的非重复；
- 同一输入两次执行产生相同输出和哈希。

完整 5,000 条另执行集成验证，不以单元测试代替真实数据验收。

## 8. 统计、图表和交付

`cleaning_stats.json` 与 `cleaning_stats.csv` 保存原始数、结构错误、两次空值、
HTML 修改、控制字符修改、超长和截断、精确重复、模糊重复及最终数。三个长度
阶段分别保存 min、P25、median、mean、P75、P90、P95、P99、max。

Matplotlib 生成：

- `length_distribution.png`：同一坐标展示清洗前、清洗后截断前和最终 token
  长度分布，并标出 2,048；
- `cleaning_counts.png`：展示各阶段剩余数量和各原因删除数量。

正式交付目录：

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
    │   ├── day7_validation.json
    │   └── day7_files.sha256
    └── tests/
        └── test_clean_pipeline.py
```

完成条件是脚本帮助和错误退出可用、全部测试通过、5,000 条统计守恒、最终两种
格式一一对应、每条不超过 2,048 tokens、最终数不少于 1,500、两次完整运行哈希
一致、图表可读，且所有 AutoDL 产物同步回仓库并提交。
