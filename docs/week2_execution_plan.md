# Qwen2.5 实习第 2 周执行计划

## 1. 文档用途与执行依据

这份文档是第 2 周任务的长期执行入口，用于回答四个问题：当天要做什么、为什么这样做、完成后保留什么、验收时怎样证明结果真实有效。

执行时遵循以下优先级：

1. 老师发布的原始任务与验收标准具有最高优先级。
2. 本计划把老师要求拆成可执行、可验证、可复现的步骤，但不把计划中的预期写成已经完成的事实。
3. 每天开始前再次阅读当天的“老师原始要求”“完成标准”和“最终检查”。
4. 每天完成后，在对应的 `deliverables/week2/dayN/README.md` 中记录实际结果和文件入口。
5. 如果本计划与老师后续通知冲突，以老师最新通知为准，并同步更新本计划。
6. 原始数据、模型输出、训练日志和失败日志必须保留真实内容，不因结果不理想而修改。

Day 6–Day 10 已按本计划完成。数据、清洗、训练、TensorBoard、LoRA 合并、
三问评测、FAQ 和周报均有正式交付；五项验收中第 1、2、3、5 项通过，
第 4 项因三题盲评基座 25/30、合并模型 22/30 而未通过。实际交付见
[`deliverables/week2/`](../deliverables/week2/)和
[`Submission/Week2/`](../Submission/Week2/README.md)。

## 2. 本周目标与最终验收

本周前半段完成多来源指令数据的收集、格式统一、清洗、截断、模糊去重和统计；后半段使用 LLaMA-Factory 完成首次正式 SFT、TensorBoard 监控、LoRA 合并、对话测试和周报复盘。

老师给出的五项最终验收必须逐条满足：

1. `clean_pipeline.py` 可以通过命令行独立运行。
2. 清洗后训练集不少于 1500 条，且数据格式正确。
3. SFT 训练 loss 整体稳定下降，不出现 NaN、Inf 或异常发散。
4. 合并后模型在固定的新问题上，对话质量优于同一基座模型。
5. 提交《第 2 周：数据与 SFT 入门报告》及本周 FAQ。

“训练成功”和“模型质量提升”是两项不同结论：

- 训练成功由命令退出码、完整 step、有限 loss、adapter 文件和 TensorBoard event 共同证明。
- 模型质量提升必须用未进入训练集的固定问题，在相同模板和生成参数下比较基座模型与合并模型，不能只用训练 loss 代替。

## 3. 已确定的实验基线

第 2 周沿用 Week 1 已经实际验证的环境和模型，不无故重装核心依赖或重复下载基座模型。

| 项目 | 当前事实或执行口径 |
|---|---|
| 计算设备 | NVIDIA GeForce RTX 3090，24GB 显存 |
| 数据盘 | `/root/autodl-tmp`，容量 100GB |
| 远端项目根目录 | `/root/autodl-tmp/qwen25-week1` |
| 第 2 周远端目录 | `/root/autodl-tmp/qwen25-week2` |
| Conda 环境 | `llm_exp`，Python 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| CUDA Runtime | 12.1 |
| Transformers | 4.50.0 |
| LLaMA-Factory | 0.9.3 |
| 基座模型 | `Qwen/Qwen2.5-7B-Instruct` |
| 本地模型目录 | `/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct` |
| Tokenizer | 与基座模型同目录的 Qwen2.5 Tokenizer |
| 训练方式 | SFT + LoRA；是否继续使用 4-bit 量化以实际 `qwen_lora.yaml`、显存预检和版本兼容性为准 |
| 随机种子 | 默认固定为 `42`，用于抽样、数据划分和训练 |
| 最大样本长度 | 2048 tokens；以实际 Qwen2.5 Tokenizer 统计，不按字符数估算 |

Day 6 开始时需要重新记录 GPU、磁盘、Python 和关键包版本。如果实际环境与上表不同，后续报告采用当天日志中的实际值，并说明变化。

## 4. 当前进度

| 日期 | 老师要求 | 状态 | 计划交付入口 |
|---|---|---|---|
| Day 6 | 数据收集与格式统一 | 已完成 | [`deliverables/week2/day6/`](../deliverables/week2/day6/README.md) |
| Day 7 | 清洗 Pipeline、模糊去重与统计图 | 已完成 | [`deliverables/week2/day7/`](../deliverables/week2/day7/README.md) |
| Day 8 | 逐行理解配置并启动首次 SFT | 已完成 | [`deliverables/week2/day8/`](../deliverables/week2/day8/README.md) |
| Day 9 | TensorBoard、LoRA 合并与三问测试 | 已完成 | [`deliverables/week2/day9/`](../deliverables/week2/day9/README.md) |
| Day 10 | 报错复盘、FAQ 与第 2 周报告 | 已完成 | [`deliverables/week2/day10/`](../deliverables/week2/day10/README.md) |

只有在原始日志、脚本、配置和交付文件通过当天检查后，才能把状态改为“已完成”。

## 5. 每天通用的执行规则

### 5.1 开始前

- 激活 Week 1 已验证的环境，不另建同功能环境：

```bash
conda activate llm_exp
cd /root/autodl-tmp/qwen25-week1
```

- 使用任务专用变量避免多处手写路径：

```bash
export WORKDIR=/root/autodl-tmp/qwen25-week1
export WEEK2_DIR="$WORKDIR/week2"
export MODEL_DIR="$WORKDIR/week1/models/Qwen2.5-7B-Instruct"
```

- 创建目录时按 raw、interim、processed、configs、scripts、logs、plots、saves、merged 和 reports 分开，避免覆盖原始数据：

```bash
mkdir -p \
  "$WEEK2_DIR/data/raw" \
  "$WEEK2_DIR/data/interim" \
  "$WEEK2_DIR/data/processed" \
  "$WEEK2_DIR/configs" \
  "$WEEK2_DIR/scripts" \
  "$WEEK2_DIR/logs" \
  "$WEEK2_DIR/plots" \
  "$WEEK2_DIR/saves" \
  "$WEEK2_DIR/merged" \
  "$WEEK2_DIR/reports"
```

- 每天保存 `pwd`、磁盘余量、Git revision、Python 和关键依赖版本。
- 涉及 GPU 的 Day 8、Day 9 先运行 `nvidia-smi` 和 `torch.cuda.is_available()`。
- 不把老师要求的 2k、2k、1k 理解为“大约”；抽样结果必须分别等于 2000、2000、1000，除非上游有效样本不足，并在清单中如实说明。

### 5.2 执行中

- 关键命令使用 `tee` 保存完整控制台输出，并通过 `set -o pipefail` 保留真实退出状态。
- 新生成的文件不得覆盖上游原始文件；所有转换和清洗结果写入新目录。
- 每条样本保留 `source` 和 `source_id` 或等价的可追溯字段；给 LLaMA-Factory 使用时，如果需要只保留标准字段，则另存一份 provenance 映射表。
- 抽样、去重、划分和训练全部固定随机种子。
- 任何过滤、截断或去重都要记录原因和数量，不能只保留一个最终总数。
- 失败先保存日志，再进行单变量修复；重试使用新日志，不覆盖第一次失败。
- 截图只作为可视化证据，JSON、CSV、脚本、配置和文本日志才是可复核主体。

### 5.3 结束前

- 运行当天的格式验证、数量核对、哈希清单和链接检查。
- 检查文件中没有密码、Token、私钥、云平台余额或个人信息。
- 不把基座权重、完整合并模型、checkpoint、缓存、Conda 环境或 TensorBoard event 提交到 Git。
- 大体积产物保留在服务器，并在仓库中提交目录清单、大小、哈希、生成命令和必要截图。
- 未经明确确认，不执行 Git 提交或推送。

## 6. Day 6：数据收集与格式统一

### 6.1 老师原始要求

1. 下载 Alpaca-GPT4-zh，取 2000 条。
2. 下载 COIG-PC，取 2000 条。
3. 下载 ShareGPT-zh，取 1000 条。
4. 统一转换为标准 Alpaca 格式：`instruction`、`input`、`output`。
5. 统一转换为 ShareGPT 格式：`conversations`。
6. 提交原始数据归档清单。

### 6.2 当日完成标准

- 三个数据源分别完成来源确认、许可检查、revision 固定和原始字段审计。
- 三个抽样文件的记录数严格为 2000、2000、1000，总计 5000 条。
- 抽样方法可复现：记录原始 split、原始索引、随机种子和抽样脚本版本。
- 同一批 5000 条语义样本分别输出 Alpaca 版和 ShareGPT 版，不把两个等价表示拼接成 10000 条训练数据。
- Alpaca 版每条只含合法的 `instruction`、`input`、`output` 主字段；其中 `instruction` 和 `output` 为非空字符串，`input` 可以是空字符串。
- ShareGPT 版每条含 `conversations` 列表，并至少有一组顺序正确的 human/user 与 gpt/assistant 消息。
- 单轮数据无损转换；多轮 ShareGPT 原始样本必须明确转换策略，不能静默丢掉后续轮次。
- 原始数据归档清单包括数据集名称、官方来源、revision、split、许可、下载时间、上游条数、抽样条数、抽样规则、文件路径、文件大小和 SHA-256。

数据源的准确仓库、split、字段和许可必须在 Day 6 实际打开上游数据卡后确认。仅凭数据集简称猜测来源，不满足归档要求。

### 6.3 数据分层与不可变原则

目录职责固定如下：

```text
week2/data/
├── raw/                  # 下载或导出的原始子集，只读保存
├── interim/              # 字段映射后的中间文件和 provenance
└── processed/            # Day 7 清洗后的训练候选数据
```

规则：

- `raw/` 文件一经归档不再修改。
- 字段转换只写入 `interim/`。
- 清洗、截断和去重只写入 `processed/`。
- 每个派生文件都能通过 `source`、`source_id` 或独立映射表回到原始记录。
- 原始样本与最终训练样本的哈希分别记录，避免把清洗结果误称为原始数据。

### 6.4 下载与抽样口径

先为每个数据源记录：

1. 官方数据卡或仓库 URL。
2. 精确 revision、commit 或下载文件哈希。
3. 使用的 split 和该 split 的总条数。
4. 原始字段名、消息角色标记和数据类型。
5. 许可与使用限制。
6. 是否有流式下载、脚本执行或远程代码需求。

抽样流程：

1. 读取固定 revision 的目标 split。
2. 给每条记录附加不可变的原始索引。
3. 只做保证 JSON 可序列化所需的最小处理，不在 Day 6 清洗内容。
4. 使用固定 `seed=42` 的确定性抽样。
5. 按原始索引排序后写出，保证多次执行顺序一致。
6. 校验条数、唯一原始索引和 SHA-256。

如果数据源本身少于目标数量，立即停止并保留证据，不用其他来源静默补齐；需要先向老师或用户确认替代口径。

### 6.5 标准 Alpaca 格式

目标结构：

```json
{
  "instruction": "用户希望模型完成的任务",
  "input": "完成任务所需的补充上下文；没有时为空字符串",
  "output": "期望的助手回答"
}
```

转换规则：

- 原始 prompt、question 或首轮用户消息映射到 `instruction`。
- 明确独立存在的上下文映射到 `input`；不存在时写 `""`，不写 `null`。
- assistant、answer 或 response 映射到 `output`。
- 不把 assistant 回答拼进 `instruction`。
- 不把数据来源文本混进模型可见字段；来源信息放 provenance。
- 多轮 ShareGPT 样本转换到 Alpaca 时，默认把最后一个待回答用户消息之前的有效对话序列序列化为 `input`，最后一个用户消息作为 `instruction`，对应助手消息作为 `output`。该策略需用固定模板并记录，不能只取首轮后丢弃其余轮次。

### 6.6 标准 ShareGPT 格式

目标结构采用 LLaMA-Factory 0.9.3 能通过 `dataset_info.json` 明确注册的角色标签。常见表示如下，最终标签以当前版本实际示例为准：

```json
{
  "conversations": [
    {"from": "human", "value": "用户消息"},
    {"from": "gpt", "value": "助手回答"}
  ]
}
```

转换规则：

- Alpaca 的 `instruction` 与非空 `input` 按固定模板组合成 human 消息。
- `output` 作为紧随其后的 gpt 消息。
- 原始 ShareGPT 多轮样本保留所有有效、顺序正确且能形成 user-assistant 配对的轮次。
- 角色映射必须显式记录，例如 human/user → 用户，gpt/assistant → 助手。
- system 消息是否保留以及怎样注册，必须与实际 LLaMA-Factory 版本的数据格式一致。
- conversations 不能为空，角色不能连续错位，最后一个训练目标必须是 assistant 回答。

### 6.7 转换脚本与验证

Day 6 应编写独立的下载/抽样脚本和格式转换脚本，避免在 Notebook 中手工复制数据。建议入口：

```text
week2/scripts/
├── collect_day6.py
├── convert_formats.py
└── validate_day6.py
```

验证至少覆盖：

- 三个原始子集计数为 2000、2000、1000。
- Alpaca 和 ShareGPT 两个转换版本各有 5000 条。
- 所有 Alpaca 样本都有三个标准字段，字符串类型正确。
- 所有 ShareGPT 样本的 conversations 非空、角色合法、顺序合法。
- 两个表示通过稳定 `sample_id` 一一对应。
- JSON 或 JSONL 能从头解析到结尾。
- 同样输入、revision 和 seed 重跑时，输出哈希一致。

### 6.8 原始数据归档清单

`raw_data_manifest.csv` 至少包含：

| 字段 | 含义 |
|---|---|
| `dataset_name` | 数据集标准名称 |
| `source_url` | 官方数据卡或仓库 |
| `revision` | commit、tag 或数据 revision |
| `split` | 实际使用的 split |
| `license` | 数据卡声明的许可 |
| `downloaded_at` | 含时区的下载时间 |
| `upstream_count` | 上游 split 总条数 |
| `sample_count` | 本次抽样条数 |
| `sampling_method` | 抽样算法与 seed |
| `raw_file` | 本地原始子集路径 |
| `file_size_bytes` | 文件大小 |
| `sha256` | 文件校验值 |
| `notes` | 字段差异、限制或异常 |

另保存一份人类可读的 `raw_data_manifest.md`，解释三种数据的字段映射和任何未决问题。

### 6.9 预计用时与 GPU 需求

预计 3–5 小时：来源和许可核对约 0.5–1 小时，下载与确定性抽样约 1–1.5 小时，双格式转换约 1–1.5 小时，验证和归档清单约 0.5–1 小时。

Day 6 不需要 GPU。使用 CPU、网络和数据盘即可，不加载基座模型权重。

### 6.10 本地交付目录

Day 6 实际交付目录为：

```text
deliverables/week2/day6/
├── README.md
└── source/
    ├── data/
    │   ├── raw/
    │   │   ├── alpaca_gpt4_zh_2k.jsonl
    │   │   ├── coig_pc_2k.jsonl
    │   │   └── sharegpt_zh_1k.jsonl
    │   ├── formatted/
    │   │   ├── week2_5k_alpaca.jsonl
    │   │   └── week2_5k_sharegpt.jsonl
    │   └── interim/
    │       └── day6_provenance.jsonl
    ├── manifests/
    │   ├── collection_metadata.json
    │   ├── raw_data_manifest.csv
    │   └── raw_data_manifest.md
    ├── results/
    │   ├── day6_collection.txt
    │   ├── day6_validation.json
    │   └── day6_files.sha256
    ├── scripts/
        ├── collect_day6.py
        ├── convert_formats.py
        └── validate_day6.py
    └── tests/
        └── test_day6_pipeline.py
```

如果原始子集文件因许可或体积不适合提交 Git，仓库只保存清单、脚本、哈希和可公开的必要样例，并在 README 中写明服务器归档路径。

### 6.11 Day 6 最终检查

- [x] Alpaca-GPT4-zh 精确抽样 2000 条。
- [x] COIG-PC 精确抽样 2000 条。
- [x] ShareGPT-zh 精确抽样 1000 条。
- [x] 三个数据源均记录官方来源、revision、split、许可和哈希。
- [x] 5000 条样本全部转换为标准 Alpaca 格式。
- [x] 同一批 5000 条样本全部转换为标准 ShareGPT 格式。
- [x] 两种表示可以通过稳定 ID 一一对应，没有被误算为两倍数据。
- [x] 多轮数据转换策略已写明且通过样例核对。
- [x] 原始数据归档清单字段完整。
- [x] 尚未执行 Day 7 清洗，也未把 Day 6 原始数据标成清洗后训练集。

## 7. Day 7：清洗 Pipeline 开发

### 7.1 老师原始要求

1. 编写 `clean_pipeline.py`。
2. 去除 HTML 标签。
3. 去除不可见控制字符。
4. 对超过 2048 tokens 的样本执行截断。
5. 过滤空值。
6. 使用 SimHash 或 Jaccard 相似度实现模糊去重。
7. 统计清洗前后的数据量变化。
8. 使用 matplotlib 绘制长度分布图。
9. 提交清洗脚本、清洗后数据集和统计图表；清洗后数据集不少于 1500 条。

### 7.2 当日完成标准

- `clean_pipeline.py` 有明确 CLI，可以对输入文件运行，不依赖 Notebook 状态或手工修改源码。
- 支持标准 Alpaca 和 ShareGPT 两种输入格式，或通过明确参数指定格式。
- 清洗顺序固定，并为每一步记录输入数、删除数、修改数和输出数。
- HTML 清理不把相邻词错误粘连；实体解码策略明确。
- 只删除不应进入训练文本的控制字符；保留合法的换行、Tab 处理策略和正常 Unicode。
- token 长度使用本地 Qwen2.5 Tokenizer 实测；任何清洗后样本均不超过 2048 tokens。
- 模糊去重跨三个来源执行，并生成“保留样本—重复样本—相似度证据”映射。
- 清洗后样本数不少于 1500；如果低于 1500，不通过降低清洗标准掩盖问题，而是检查阈值、输入数据和错误分类。
- 统计文件中的各阶段计数能够守恒。
- 图表标题、坐标、单位、样本数和截断点清晰可读。

### 7.3 清洗顺序

固定顺序如下：

1. 解析 JSON/JSONL 和结构校验。
2. 字段类型规范化。
3. 空值过滤。
4. HTML 标签移除与 HTML entity 处理。
5. 不可见控制字符清理。
6. 空白规范化。
7. 清洗后再次空值过滤。
8. tokenizer 长度统计与超过 2048 tokens 的结构化截断。
9. 精确去重。
10. 模糊去重。
11. 最终格式、长度和数量验证。

顺序不能随意交换。例如先清理 HTML 再判空，可以发现只包含标签的伪非空样本；先做精确去重再做 SimHash，可以减少模糊比较量。

### 7.4 HTML 与控制字符处理

HTML 处理要求：

- 使用 HTML parser 或经过测试的解析库，不只依赖一个贪婪正则。
- `<br>`、`</p>`、`</li>` 等块级边界转换为空格或换行，防止文本粘连。
- script 和 style 内容不作为自然语言保留。
- 解码常见 HTML entity 后再规范化空白。
- 保留代码中的比较符号和普通尖括号文本，避免误删非 HTML 内容。

不可见字符处理要求：

- 删除 C0/C1 中不应进入文本的控制字符。
- 明确处理零宽空格、零宽非连接符、BOM 等格式字符。
- 换行和 Tab 先按字段策略转换，再统一空白。
- 不删除正常中文、emoji、组合字符和合法数学符号。
- 每类规则都要有单元测试，防止用过宽的 Unicode 正则破坏内容。

### 7.5 空值过滤

Alpaca 格式：

- `instruction` 清洗后必须非空。
- `output` 清洗后必须非空。
- `input` 可以为空字符串。
- `null`、空白字符串、空列表、错误类型不能被静默转换成有意义文本。

ShareGPT 格式：

- `conversations` 必须是非空列表。
- 消息内容清洗后不能为空。
- 至少存在一个 user-human 与 assistant-gpt 配对。
- 角色顺序错误、缺少回答或只有 system 消息的样本应被拒绝并记录原因。

### 7.6 2048-token 统计与截断

- 使用 `AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)` 加载与训练基座一致的 Tokenizer。
- 长度按模型实际可见的样本内容统计，明确是否包含模板和特殊 token。
- 清洗报告同时保存清洗前 token 长度、清洗后截断前长度和最终长度。
- 截断在字段或消息边界上执行，不直接截断 JSON 字符串。
- 截断后必须重新解码并验证；不能把多字节字符或 token ID 切坏。
- 优先保留任务指令和至少一个完整回答目标；多轮数据从最旧上下文开始裁剪，保留最后一个有效 user-assistant 训练对。
- 如果最小有效结构本身仍超过 2048 tokens，再对最长内容字段做 tokenizer 级截断。
- 最终重新编码，断言长度 `<= 2048`。
- 记录每条被截断样本的原长度、最终长度、被处理字段和截断 token 数。

训练配置中的 `cutoff_len` 也要设为 2048。清洗端限制和训练端限制需要一致，避免训练时再次静默截断。

### 7.7 精确去重与 SimHash 模糊去重

本周选择 64-bit SimHash，原因是 5000 条数据规模下实现清楚、速度足够，并能保存可解释的 Hamming distance。正式实现由 SimHash 负责候选召回，再用整体和角色级字符 3-gram Jaccard 作为确认门槛；这不是两套独立去重流程，而是一个优先保证 precision 的两阶段判定器。

用于去重的规范文本：

- Alpaca：规范化后的 `instruction + input + output`。
- ShareGPT：按固定角色前缀串联全部有效消息内容。
- 为了跨格式和跨来源去重，最终统一投影为“用户上下文 + 助手回答”的 canonical text。
- provenance 字段不进入相似度计算。

流程：

1. 对 canonical text 计算 SHA-256，先移除精确重复。
2. 对中文和混合文本生成字符 n-gram 特征，并计算 64-bit SimHash。
3. 使用分桶减少不必要比较。
4. 用固定 Hamming distance 阈值召回近重复候选；阈值在正式运行前用真实候选和构造扰动样本校准。
5. 用整体 Jaccard 与 user/assistant 角色级 Jaccard 确认语义双方均高度相似，避免共享长上下文但任务或答案不同的误删。
6. 对边界样本保存文本对、距离、Jaccard 分数和最终判定。
7. 重复簇中按确定性规则保留一条，默认优先保留结构完整、无截断、回答非空且原始索引更早的样本。
8. 输出 `duplicate_pairs.csv` 和 `dedup_clusters.jsonl`。

阈值不能只为保证“清洗后不少于 1500 条”而调整。阈值选择、校准样本和误判观察必须写入报告。

### 7.8 `clean_pipeline.py` CLI

正式接口：

```bash
conda run -n llm_exp python deliverables/week2/day7/clean_pipeline.py \
  --input deliverables/week2/day6/source/data/formatted/week2_5k_sharegpt.jsonl \
  --provenance deliverables/week2/day6/source/data/interim/day6_provenance.jsonl \
  --output-alpaca deliverables/week2/day7/data/week2_clean_alpaca.jsonl \
  --output-sharegpt deliverables/week2/day7/data/week2_clean_sharegpt.jsonl \
  --tokenizer /root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct \
  --max-tokens 2048 \
  --simhash-threshold 3 \
  --min-jaccard 0.85 \
  --min-role-jaccard 0.90 \
  --seed 42 \
  --results-dir deliverables/week2/day7/source/results \
  --evidence-dir deliverables/week2/day7/evidence \
  2>&1 | tee deliverables/week2/day7/source/results/day7_clean_pipeline.txt
```

脚本至少应支持：

- `--input`
- `--provenance`
- `--output-alpaca`
- `--output-sharegpt`
- `--tokenizer`
- `--max-tokens`
- `--simhash-threshold`
- `--min-jaccard`
- `--min-role-jaccard`
- `--seed`
- `--results-dir`
- `--evidence-dir`

输入错误时返回非零退出码并给出明确原因；成功时打印输入数、输出数、各阶段变化和产物路径。

### 7.9 统计与图表

`cleaning_stats.json` 和 `cleaning_stats.csv` 至少记录：

- 原始总数。
- JSON/结构错误数。
- 初次空值数。
- HTML 被修改样本数。
- 控制字符被修改样本数。
- 清洗后二次空值数。
- 超长样本数。
- 被截断样本数。
- 精确重复删除数。
- 模糊重复删除数。
- 最终样本数。
- 每阶段剩余数。
- token 长度的 min、P25、median、mean、P75、P90、P95、P99、max。

使用 matplotlib 至少生成：

1. `length_distribution.png`：同一可比坐标下展示清洗前、清洗后截断前、最终数据的 token 长度分布，并标出 2048。
2. `cleaning_counts.png`：展示各阶段剩余样本数或各原因删除数。

原始数值同时保存为 CSV/JSON，不能只提交图片。

### 7.10 测试要求

单元测试至少覆盖：

- HTML 标签和实体。
- script/style 内容。
- 控制字符、零宽字符、正常 emoji 和组合字符。
- Alpaca 三字段空值。
- ShareGPT 空消息、错序角色和多轮对话。
- 恰好 2048、2049 和远大于 2048 tokens。
- 截断后仍能解码且长度不超过上限。
- 精确重复。
- 轻微标点、空白或措辞变化的近重复。
- 主题相似但答案不同的非重复，防止误杀。
- 固定输入两次运行得到相同输出。

另对完整 5000 条执行集成验证，确保各阶段计数守恒。

### 7.11 预计用时与 GPU 需求

预计 5–8 小时：清洗规则和测试约 2–3 小时，tokenizer 截断约 1–1.5 小时，SimHash 与阈值校准约 1.5–2 小时，完整运行、统计和图表约 1–1.5 小时。

Day 7 只加载 Tokenizer，不加载模型权重，使用 CPU 即可。

### 7.12 本地交付目录

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
    │   ├── duplicate_pairs.csv
    │   ├── dedup_clusters.jsonl
    │   ├── day7_clean_pipeline.txt
    │   ├── day7_validation.json
    │   └── day7_files.sha256
    └── tests/
        └── test_clean_pipeline.py
```

### 7.13 Day 7 最终检查

- [x] `clean_pipeline.py --help` 可直接运行。
- [x] HTML 标签和不可见控制字符按测试规则清理：实际修改 72 条和 27 条。
- [x] 空值和错误结构均被过滤并分类计数：真实数据均为 0，异常路径由测试覆盖。
- [x] 所有最终样本使用 Qwen2.5 Tokenizer 统计后均不超过 2048 tokens：最大值 2048。
- [x] SimHash 跨来源模糊去重完成，阈值、Jaccard 门槛和候选证据已保存。
- [x] 清洗后训练候选集不少于 1500 条：实际 4999 条。
- [x] 清洗前后数量变化能够守恒：5000 减 1 条精确重复等于 4999。
- [x] token 长度原始统计和 matplotlib 图表均已保存。
- [x] Alpaca 和 ShareGPT 两种清洗结果格式均通过验证，各 4999 条。
- [x] 完整数据集重跑结果确定且可复现：12 项语义产物逐字节一致。

## 8. Day 8：配置详解与首次训练

### 8.1 老师原始要求

1. 复制 LLaMA-Factory 的 `qwen_lora.yaml`。
2. 逐行注释所有参数。
3. 重点理解并解释 `model_name_or_path`、`template`、`finetuning_type`、`lora_rank`、`lora_alpha` 和 `learning_rate`。
4. 使用清洗后的数据启动首次 SFT 训练。
5. 提交带详尽注释的 YAML 配置文件。

### 8.2 当日完成标准

- `qwen_lora.yaml` 来自当前 LLaMA-Factory 0.9.3 实际安装包或同 revision 官方仓库，保留来源路径和 SHA-256。
- 原始示例副本保持不改；另建带注释的训练配置。
- 配置中的每一个有效参数上方都有解释，不只注释老师点名的六个参数。
- 每条注释至少回答：参数控制什么、当前值为什么适合本实验、调大或调小的主要影响。
- YAML 能被解析，字段名与当前 LLaMA-Factory 版本兼容。
- 清洗数据通过 `dataset_info.json` 正确注册；预处理能够打印真实样本数。
- 训练使用清洗后的一种标准表示，不把 Alpaca 与 ShareGPT 的等价副本同时加入训练。
- 首次 SFT 自然启动并产生真实 step、loss 和 TensorBoard event。
- 完整训练若跨到 Day 9 结束，Day 8 仍需保留启动日志、配置哈希和输出目录。

### 8.3 先复制、再理解、后修改

执行顺序：

1. 定位当前版本实际 `qwen_lora.yaml`。
2. 保存原始文件为 `qwen_lora.official.yaml`。
3. 记录 LLaMA-Factory 版本、文件来源和 SHA-256。
4. 从上到下阅读每一行，按模型、方法、数据、输出、训练、精度六组整理。
5. 创建 `qwen25_7b_week2_lora_annotated.yaml`，逐项添加中文注释。
6. 只修改本实验必须变化的路径、数据集名、输出目录、长度和显存相关值。
7. 对比原始与修改版本，确保没有无解释地删除字段。
8. 用 YAML parser 和 `llamafactory-cli train --help` 或当前版本预检验证。

不能直接从网络文章复制一份不同版本配置，也不能只让命令跑通而不解释参数之间的关系。

### 8.4 六个重点参数

#### `model_name_or_path`

- 指向基座模型目录或模型仓库 ID。
- 本实验使用 Week 1 已下载并验证的 Qwen2.5-7B-Instruct 本地目录。
- 它决定模型架构、权重和默认配置；路径错误会在加载阶段失败。
- LoRA 只训练增量参数，导出时仍需同一个基座模型。

#### `template`

- 决定 instruction、input、system、assistant 等字段怎样被拼成模型看到的 token 序列。
- Qwen2.5-Instruct 必须使用与当前 LLaMA-Factory 版本匹配的 Qwen 模板。
- 模板错误会导致角色 token、生成前缀和标签掩码不匹配，即使训练能运行也可能学坏。
- 配置注释必须写出当前值来自哪个官方示例，并用一条样本实际预览模板结果。

#### `finetuning_type`

- 决定全参数、LoRA 或其他参数高效微调方式。
- 当前任务选择 `lora`，只更新低秩 adapter，降低显存和存储需求。
- 它与 `lora_rank`、`lora_alpha`、`lora_target` 和 adapter 输出文件共同作用。
- 不能把 `stage: sft` 与 `finetuning_type: lora` 混为一谈：前者是训练阶段，后者是参数更新方式。

#### `lora_rank`

- 低秩矩阵的秩，控制 adapter 容量和可训练参数量。
- rank 较大通常有更强拟合能力，但增加显存、计算量和过拟合风险。
- 当前值应沿用官方示例或 Week 1 已验证基线，除非显存或结果证据要求调整。
- 报告需给出当前 rank 和由训练日志统计的可训练参数量。

#### `lora_alpha`

- LoRA 更新的缩放系数，与 rank 一起决定增量更新的有效尺度。
- 常见实现中的缩放与 `alpha / rank` 或具体 PEFT 变体有关，必须按当前实现说明。
- alpha 不是学习率，也不等于 adapter 参数量。
- 当前选择要与 rank 配套解释，不能孤立写“越大越好”。

#### `learning_rate`

- 控制优化器每步更新幅度。
- LoRA SFT 常能使用比全参数训练更高的学习率，但具体值仍受 batch、数据规模、epoch 和 scheduler 影响。
- 过高可能导致 loss 振荡、发散或灾难性偏移；过低可能收敛慢或欠拟合。
- 当前值必须来自官方示例、Week 1 经验和本次有效 batch 的综合判断，并在训练曲线中验证。

### 8.5 其余参数逐行注释要求

配置中出现的所有字段都要解释，包括但不限于：

- 模型：`trust_remote_code`、量化相关字段。
- 阶段：`stage`、`do_train`。
- LoRA：`lora_target`、`lora_dropout`。
- 数据：`dataset`、`dataset_dir`、`cutoff_len`、`max_samples`、`overwrite_cache`、`preprocessing_num_workers`。
- 输出：`output_dir`、`logging_steps`、`save_steps`、`save_total_limit`、`plot_loss`、`overwrite_output_dir`、`report_to`。
- batch：`per_device_train_batch_size`、`gradient_accumulation_steps`。
- 优化：`num_train_epochs`、scheduler、warmup、weight decay（如果存在）。
- 精度和性能：`bf16`、`fp16`、gradient checkpointing（如果存在）。
- 复现：`seed`。

如果实际 `qwen_lora.yaml` 有其他字段，也必须逐行补充，不能以此清单为上限。

### 8.6 数据注册与训练输入

正式训练选择清洗后的 Alpaca 版作为唯一语义训练集，ShareGPT 版用于格式交付和交叉验证，不把两份等价数据一起训练。

`dataset_info.json` 注册项至少明确：

- 文件名。
- formatting 类型。
- instruction、input、output 的列映射。
- 是否有 system 或 history。

启动前随机抽查至少 20 条，并用 LLaMA-Factory 预处理日志确认：

- 样本数等于 Day 7 最终统计。
- 模板渲染后角色正确。
- label 只覆盖预期 assistant 目标。
- 长度不超过 2048。
- 没有把 provenance 字段拼进训练文本。

### 8.7 训练前预检

```bash
test -f "$MODEL_DIR/config.json"
test -f "$WEEK2_DIR/data/processed/week2_clean_alpaca.jsonl"
test -f "$WEEK2_DIR/data/dataset_info.json"
test -f "$WEEK2_DIR/configs/qwen25_7b_week2_lora_annotated.yaml"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"
llamafactory-cli version
```

还要检查：

- 数据集最终条数不少于 1500。
- YAML 可解析且没有重复 key。
- 配置中的路径全部存在。
- `cutoff_len=2048` 与 Day 7 一致。
- `logging_steps=1`，以满足 Day 9 每步 loss 记录要求。
- `report_to` 包含 TensorBoard，`plot_loss=true`。
- GPU 没有其他进程占用大量显存。
- 输出目录不存在或明确允许覆盖。
- 磁盘同时容纳 adapter、checkpoint 和 Day 9 合并模型。
- 配置、数据和测试问题的 SHA-256 已保存。

### 8.8 首次 SFT 命令

```bash
cd "$WORKDIR"
set -o pipefail
llamafactory-cli train \
  "$WEEK2_DIR/configs/qwen25_7b_week2_lora_annotated.yaml" \
  2>&1 | tee "$WEEK2_DIR/logs/day8_sft_train.txt"
status=${PIPESTATUS[0]}
echo "training_exit_code=$status" | tee -a "$WEEK2_DIR/logs/day8_sft_train.txt"
test "$status" -eq 0
```

如果训练时间延续到 Day 9，不人为中止；Day 8 README 记录开始时间、当前 step、输出目录和监控入口。

### 8.9 OOM 处理顺序

遇到 CUDA OOM 时一次只改一个变量，并保留每次配置和日志：

1. 检查是否有其他 GPU 进程。
2. 确认实际使用 LoRA 和预期量化方式，而非误加载全参数训练。
3. 保持 `per_device_train_batch_size=1`。
4. 启用当前版本支持的 gradient checkpointing。
5. 调整 gradient accumulation 以维持可接受的有效 batch。
6. 如果仍 OOM，再降低 `cutoff_len`；此时必须重新运行 Day 7 长度与截断口径，并在报告中解释不再使用 2048 的原因。

不能通过减少总样本数解决单步峰值显存，也不能把失败日志覆盖成一次看似成功的运行。

### 8.10 预计用时与 GPU 需求

预计 4–8 小时：定位和逐行理解官方配置约 1.5–2.5 小时，数据注册与预检约 1–1.5 小时，首次 SFT 视样本长度和 epoch 约 1–4 小时。

配置阅读和数据注册可在 CPU 环境完成；正式训练必须使用 RTX 3090。

### 8.11 本地交付目录

```text
deliverables/week2/day8/
├── README.md
├── configs/
│   ├── qwen_lora.official.yaml
│   └── qwen25_7b_week2_lora_annotated.yaml
└── source/
    ├── data/
    │   └── dataset_info.json
    └── results/
        ├── qwen_lora_source.txt
        ├── config_diff.txt
        ├── config_validation.txt
        ├── day8_preflight.txt
        ├── day8_sft_train.txt
        └── day8_files.sha256
```

### 8.12 Day 8 最终检查

- [x] 官方 LoRA SFT YAML 的来源、版本和哈希已保存；当前 0.9.3 无字面同名纯文本 `qwen_lora.yaml`，使用同版本官方 `llama3_lora_sft.yaml` 原样归档。
- [x] 原始示例副本未被修改，SHA-256 为 `ecd8efc4cdf29d52d8a0be25028443cc813f6d82db957a90f124d7dee5095fba`。
- [x] 训练配置中 45 个有效参数均有“作用、当前值、调整影响”中文注释。
- [x] 六个重点参数的作用、当前值和相互关系均已解释。
- [x] YAML 与 LLaMA-Factory 0.9.3 参数解析器均通过。
- [x] 清洗数据正确注册，只训练 4,999 条 Alpaca 表示。
- [x] 实际训练样本不少于 1500：实际 4,999 条。
- [x] 首次 SFT 自然完成，产生 3,747 步 loss、TensorBoard event 和静态 loss 图。
- [x] 训练退出码为 0；首末 100 步均值为 1.6075/1.0615，整体斜率为负。
- [x] 最终 LoRA adapter 为 80,792,096 字节，rank 8 / alpha 16 与配置一致。

## 9. Day 9：训练监控与合并

### 9.1 老师原始要求

1. 开启 TensorBoard 监控 loss 曲线。
2. 记录每一步 loss 值。
3. 训练完成后，使用 `llamafactory-cli export` 将 LoRA 权重合并到基座模型。
4. 加载合并后的模型。
5. 使用 3 个新问题测试微调效果。
6. 提交 loss 曲线截图、合并模型文件夹和测试对话记录。

### 9.2 当日完成标准

- 训练自然结束且退出码为 0。
- 每个 logging step 都有 step、epoch、loss、learning rate，`logging_steps=1`。
- TensorBoard Scalars 显示完整训练 run 的 loss 曲线、横轴 step、run 名称和数据点。
- loss 为有限值，整体趋势稳定下降；不要求每一步严格单调。
- adapter 配置和权重存在，训练参数与 Day 8 注释配置一致。
- `llamafactory-cli export` 自然结束且退出码为 0。
- 合并目录包含可加载的模型配置、Tokenizer 和完整权重分片。
- 对合并目录执行文件清单、总大小和 SHA-256。
- 三个测试问题在训练前固定，确认不在训练数据中。
- 基座模型和合并模型使用同一问题、模板和生成参数，保留完整原始回答。
- “优于基座”由预先定义的评分维度支持，不凭主观挑选单个好回答。

### 9.3 每步 loss 记录

训练配置必须设置 `logging_steps: 1`。训练后从实际 Trainer 日志或 `trainer_state.json` 导出：

```text
step,epoch,loss,learning_rate,grad_norm
```

保存为 `train_loss_by_step.csv`。如果某字段在当前版本不存在，保留空值并在 README 说明，不能编造。

同时计算：

- 第一个有效 loss。
- 最后一个有效 loss。
- 最小和最大 loss。
- 前 10% steps 的平均 loss。
- 后 10% steps 的平均 loss。
- 固定窗口移动平均。
- NaN/Inf 数量。

“稳定下降”的判断以移动平均和前后区间均值为主，不把单步波动误判为失败，也不只截取曲线的局部下降区间。

### 9.4 TensorBoard

```bash
tensorboard \
  --logdir "$WEEK2_DIR/saves" \
  --bind_all \
  --port 6006
```

截图要求：

- 位于 Scalars 页面。
- 显示训练 loss tag。
- 横轴选择 step。
- 显示完整 run 名称和全部曲线范围。
- smoothing 数值可见；同时保留原始数据或将 smoothing 调低，避免只展示过度平滑结果。
- 如有失败重试，各 run 使用不同目录和名称，不覆盖或隐藏。

### 9.5 训练完成检查

检查：

- `training_exit_code=0`。
- 实际 epoch 和 step 与数据量、batch、gradient accumulation 相符。
- 日志无 Traceback、CUDA OOM、NaN、Inf 或手动中断。
- adapter 目录有 `adapter_config.json` 和 adapter 权重。
- TensorBoard event 文件存在。
- `trainer_state.json` 或等价状态文件能导出每步 loss。
- 最终配置、数据哈希和 LLaMA-Factory 版本与 Day 8 预检一致。

### 9.6 LoRA 合并配置

先复制当前版本官方 export 示例，再按本实验填写，至少明确：

- `model_name_or_path`：与训练相同的 Qwen2.5-7B-Instruct 基座。
- `adapter_name_or_path`：本次训练成功的最终 LoRA adapter 目录。
- `template`：与训练一致的 Qwen 模板。
- `finetuning_type`：`lora`。
- `export_dir`：独立的合并模型目录。
- `export_size`：权重分片大小。
- `export_device`：根据 CPU/GPU 内存选择并记录。
- `export_legacy_format`：按当前 Transformers/LLaMA-Factory 兼容要求设置。

训练时如果使用 4-bit 量化，合并导出仍需按官方流程加载基座权重并生成可独立加载的完整模型。不能把 adapter 文件夹改名后当成合并模型。

### 9.7 执行合并

```bash
set -o pipefail
llamafactory-cli export \
  "$WEEK2_DIR/configs/qwen25_7b_week2_export.yaml" \
  2>&1 | tee "$WEEK2_DIR/logs/day9_export.txt"
status=${PIPESTATUS[0]}
echo "export_exit_code=$status" | tee -a "$WEEK2_DIR/logs/day9_export.txt"
test "$status" -eq 0
```

合并后检查：

- `config.json`。
- Tokenizer 文件。
- generation 配置（如果官方流程生成）。
- 一个或多个完整模型权重文件。
- 权重索引（如果分片）。
- 总目录大小符合 7B 模型所需量级。
- 不存在只含几十或几百 MB adapter 的误判。
- `AutoModelForCausalLM.from_pretrained(merged_dir)` 能独立加载，不再传 adapter 路径。

完整合并模型体积较大，不提交 Git。仓库提交 `merged_model_inventory.txt`、`merged_model.sha256`、导出配置、导出日志和服务器绝对路径；按老师要求保留可验收的服务器文件夹。

### 9.8 三个新问题

问题在查看合并模型回答前固定，至少覆盖：

1. 与训练数据主要任务分布一致、但文本未出现过的指令。
2. 需要遵循明确格式或多约束的任务。
3. 用于观察通用对话质量和异常退化的问题。

固定问题后对清洗后训练集执行精确匹配和模糊相似度检查，保存“未命中”证据。不能从训练样本轻微改写后称为新问题。

### 9.9 基座与合并模型公平对比

两次推理保持一致：

- 同一个 Qwen Chat Template。
- 同一 system prompt。
- 同一 `max_new_tokens`。
- 同一 decoding 参数。
- 同一随机种子；质量验收优先采用确定性解码。
- 同一设备和精度口径。

每条记录保存：

- 问题 ID 和原始 messages。
- 模板渲染文本或 token IDs。
- 生成参数。
- 基座原始回答。
- 合并模型原始回答。
- 输入/输出 token 数、耗时和显存。
- 评分与理由。

评分维度在生成前固定，可使用：

- 指令完成度。
- 正确性或事实一致性。
- 格式遵循。
- 相关性与完整性。
- 异常重复、乱码或角色泄漏。

最终同时给出逐题结果和总体结论。如果三题不能支持“合并模型优于基座”，必须如实记录并回查数据、模板、训练强度和评测题设计，不能改写回答或只保留胜出的题。

### 9.10 预计用时与 GPU 需求

预计 4–8 小时：训练收尾和 loss 导出约 0.5–1 小时，TensorBoard 证据约 0.5 小时，LoRA 合并和独立加载约 1–3 小时，基座/合并三问对比约 1.5–3 小时。

TensorBoard 和 CSV 导出可用 CPU；模型合并需要足够内存与磁盘，最终推理使用 RTX 3090。

### 9.11 本地交付目录

```text
deliverables/week2/day9/
├── README.md
├── configs/
│   └── qwen25_7b_week2_export.yaml
├── evidence/
│   └── tensorboard_loss.png
└── source/
    └── results/
        ├── train_loss_by_step.csv
        ├── loss_summary.json
        ├── day9_export.txt
        ├── merged_model_inventory.txt
        ├── merged_model.sha256
        ├── merged_model_load_test.txt
        ├── evaluation_questions.json
        ├── evaluation_novelty_check.txt
        ├── base_responses.jsonl
        ├── merged_responses.jsonl
        ├── comparison.md
        └── day9_files.sha256
```

服务器同时保留：

```text
week2/merged/qwen25-7b-week2-sft-merged/
```

### 9.12 Day 9 最终检查

- [x] 训练退出码为 0，计划 epoch 和 step 全部完成。
- [x] 每一步 loss 均已导出为 CSV。
- [x] loss 无 NaN/Inf，移动平均和前后区间支持稳定下降结论。
- [x] TensorBoard 截图显示完整 loss 曲线、step 和 run 名称。
- [x] LoRA adapter 配置与权重完整。
- [x] `llamafactory-cli export` 退出码为 0。
- [x] 合并模型文件夹包含配置、Tokenizer 和完整权重。
- [x] 合并模型不依赖 adapter 路径即可加载。
- [x] 3 个问题均为训练集外的新问题。
- [x] 基座与合并模型使用完全相同的推理条件。
- [x] 原始对话记录和评分理由均已保存。
- [x] 证据能够支持或如实否定“合并后质量优于基座”。

## 10. Day 10：周报与问题复盘

### 10.1 老师原始要求

1. 整理本周遇到的报错，包括 OOM、数据格式错误及解决方案。
2. 撰写《第 2 周：数据与 SFT 入门报告》。
3. 提交周报和 FAQ 记录。

### 10.2 当日完成标准

- FAQ 中每个“实际遇到”的错误都能链接到原始失败日志、失败配置或触发样本。
- 错误记录包含现象、完整报错摘要、环境、根因、排查过程、最终修复、验证和预防措施。
- OOM 和数据格式错误各有独立条目；如果本周没有真实发生，明确写“未实际发生”，并把预防或演练口径与实际故障分开，不能编造。
- 周报覆盖 Day 6–Day 10 的要求、方法、结果、证据和限制。
- 周报逐项回答五条最终验收。
- 数据数量、过滤原因、长度分布、训练配置、loss 和三问比较均引用真实产物。
- 不把训练 loss 下降扩大解释为通用能力提升。
- 报告中的命令、路径、版本、样本数和图表互相一致。

### 10.3 FAQ 记录模板

每个问题使用以下结构：

1. 问题标题和发生时间。
2. 所属阶段：收集、转换、清洗、训练、导出或推理。
3. 环境和相关文件版本。
4. 触发命令或最小复现步骤。
5. 原始报错摘要及完整日志路径。
6. 影响范围。
7. 候选原因和排查顺序。
8. 已确认根因及证据。
9. 单变量修复。
10. 修复后的验证结果。
11. 如何预防再次发生。
12. 是否仍有风险。

重点错误类型：

- CUDA OOM。
- Alpaca 缺字段、空 output 或类型错误。
- ShareGPT conversations 为空、角色标签不匹配、消息错序。
- `dataset_info.json` 注册项与文件字段不一致。
- tokenizer 长度超过训练 `cutoff_len`。
- YAML 参数名与 LLaMA-Factory 版本不兼容。
- TensorBoard 找不到 event 或 loss tag。
- export 路径错误、adapter 与基座不匹配。
- 合并模型缺 Tokenizer 或权重分片。

只有实际发生过的错误写入“本周实际问题”；其余内容放入“预防性检查清单”。

### 10.4 OOM 复盘要点

如果发生 OOM，必须回答：

- OOM 发生在模型加载、前向、反向、optimizer step、导出还是推理。
- 当时的剩余显存和其他 GPU 进程。
- 是否实际使用 LoRA、量化、BF16/FP16 和 gradient checkpointing。
- batch size、gradient accumulation、cutoff length 和最长样本。
- 每次只修改了哪个变量。
- 修复后峰值显存、训练速度和有效 batch 怎样变化。
- 是否改变了 2048-token 要求；如果改变，如何重新处理数据并向老师说明。

### 10.5 数据格式错误复盘要点

如果发生格式错误，必须保存：

- 失败记录的 `source` 和 `source_id`。
- 期望 schema 与实际字段。
- parser 或 LLaMA-Factory 的原始报错。
- 是数据源差异、转换 bug、空值清洗还是注册配置造成。
- 修复前后的最小样例。
- 防止回归的单元测试。
- 全量数据重新验证后的通过数量。

不能只写“修改格式后解决”，必须说明错误从哪里进入 Pipeline、为什么先前校验没有拦截。

### 10.6 第 2 周报告结构

《第 2 周：数据与 SFT 入门报告》按以下结构撰写：

1. 本周目标与验收矩阵。
2. 实验环境、模型、LLaMA-Factory 版本和目录。
3. 三个数据集的来源、revision、许可、split 和抽样方法。
4. Alpaca 与 ShareGPT 数据结构、字段映射和多轮转换策略。
5. 清洗 Pipeline 架构和执行顺序。
6. HTML、控制字符、空值和 2048-token 截断规则。
7. SimHash 原理、canonical text、阈值校准和重复簇处理。
8. 清洗前后样本数、各原因变化和长度分布。
9. LLaMA-Factory `qwen_lora.yaml` 的参数分组与六个重点参数。
10. 实际 SFT 配置、样本数、有效 batch、epoch、step 和环境。
11. 每步 loss、TensorBoard 曲线和稳定性判断。
12. LoRA adapter、export 合并流程和完整模型验证。
13. 3 个新问题的基座/合并模型公平对比。
14. 本周实际报错、根因、修复和预防措施。
15. 五条验收标准逐项结论。
16. 实验限制与下周计划。
17. 附录：复现命令、文件索引、哈希和服务器大文件路径。

### 10.7 报告必须解释的原理

- 为什么训练数据需要结构统一、清洗和去重。
- Alpaca 单轮结构与 ShareGPT 多轮结构的区别。
- token 截断为什么必须使用与训练相同的 Tokenizer。
- SimHash 怎样把文本特征映射为指纹，Hamming distance 表示什么。
- SFT 怎样用指令—回答数据学习条件生成。
- LoRA 的两个低秩矩阵怎样减少可训练参数。
- `lora_rank` 与 `lora_alpha` 的区别。
- Chat Template 为什么属于训练分布的一部分。
- batch size、gradient accumulation、learning rate、epoch 和 loss 的关系。
- adapter 与合并模型文件夹的区别。
- 为什么训练 loss 下降不自动等于真实对话质量提升。

### 10.8 本地交付目录

```text
deliverables/week2/day10/
├── README.md
├── REPORT.md
├── FAQ.md
└── source/
    └── results/
        ├── week2_acceptance_matrix.md
        ├── week2_file_manifest.csv
        ├── week2_files.sha256
        └── week2_final_validation.txt
```

### 10.9 Day 10 最终检查

- [x] 实际发生的 OOM、数据格式错误或其他错误均有日志和根因证据。
- [x] 未实际发生的错误没有被伪造成实验经历。
- [x] FAQ 包含复现、根因、修复、验证和预防。
- [x] 《第 2 周：数据与 SFT 入门报告》覆盖 Day 6–Day 10。
- [x] 报告中的样本数与清洗统计一致。
- [x] 报告中的训练配置与实际 YAML、日志一致。
- [x] 报告中的 loss 结论与 CSV、TensorBoard 一致。
- [x] 报告中的模型质量结论与三问原始对话一致。
- [x] 五条验收标准全部有明确证据入口。
- [x] 全部交付文件的哈希和索引已生成。

## 11. 最终口述准备

### 11.1 数据 Pipeline

能够按顺序说明：固定来源和 revision → 2k/2k/1k 确定性抽样 → 双格式转换 → schema 校验 → HTML/控制字符/空值清洗 → Qwen Tokenizer 2048-token 截断 → 精确去重 → SimHash 模糊去重 → 数量与长度统计 → LLaMA-Factory 数据注册。

### 11.2 Alpaca 与 ShareGPT

能够说明：Alpaca 使用 instruction、input、output 表示单个任务；ShareGPT 用带角色的 conversations 表示对话序列。两者可以是同一语义样本的不同表示，不能把等价副本同时训练后宣称数据量翻倍。

### 11.3 SimHash

能够按“特征—哈希—加权投票—64-bit 指纹—Hamming distance—阈值—重复簇”解释，并说明为何需要人工校准边界样本。

### 11.4 LoRA 与配置

能够说明：SFT 学习目标、LoRA 低秩更新、rank 控制容量、alpha 控制缩放、learning rate 控制优化步幅、template 控制模型实际看到的角色化 token 序列。

### 11.5 从 adapter 到合并模型

能够说明：训练只保存 LoRA 增量权重；推理时可加载“基座 + adapter”，也可通过 export 把增量合并进基座权重；合并后的文件夹必须能独立加载，并包含模型配置、Tokenizer 和完整权重。

## 12. 故障处理顺序

### 12.1 数据源或字段与预期不同

1. 停止批量转换。
2. 保存数据卡、revision、split 和一条脱敏原始样例。
3. 打印 schema 和角色标签分布。
4. 为该来源单独实现显式 adapter。
5. 添加回归测试后再全量转换。

### 12.2 清洗后样本异常减少

1. 核对各阶段计数是否守恒。
2. 分别查看空值、结构错误、精确重复和模糊重复数量。
3. 人工检查 SimHash 边界样本和最大重复簇。
4. 确认没有把 Alpaca 与 ShareGPT 等价表示同时输入去重。
5. 修正 bug 或经证据重新校准阈值，不为满足 1500 条而任意放宽。

### 12.3 数据格式错误

1. 保存失败 `source_id` 和原始记录。
2. 对照 dataset_info、实际列名和角色标签。
3. 用最小样例复现。
4. 修复转换器或注册项。
5. 添加单元测试并全量重跑验证。

### 12.4 CUDA OOM

1. 保存完整报错和 `nvidia-smi`。
2. 确认没有其他 GPU 进程。
3. 确认 LoRA、量化和精度设置真实生效。
4. 保持单卡 batch size 1。
5. 依次考虑 gradient checkpointing 和 accumulation。
6. 最后才降低 cutoff length，并同步更新数据和报告口径。

### 12.5 Loss 不下降或异常波动

1. 确认数据格式、label 和 template。
2. 检查是否大量空回答、重复或截断样本。
3. 检查每步 loss、learning rate、grad norm 和 warmup。
4. 确认 adapter 参数确实可训练。
5. 一次只调整 learning rate、epoch 或有效 batch 中的一个变量。
6. 保留原始 run，使用新输出目录重试。

### 12.6 TensorBoard 没有曲线

1. 确认 `report_to` 包含 TensorBoard。
2. 确认 `logging_steps=1`。
3. 查找实际 `events.out.tfevents.*`。
4. 让 `--logdir` 指向 event 文件的共同上级目录。
5. 检查 Scalars tag，不只查看首页。

### 12.7 Export 或合并模型加载失败

1. 确认 adapter 与基座模型完全匹配。
2. 检查 export 配置路径和当前 LLaMA-Factory 字段。
3. 检查磁盘和内存。
4. 保存导出目录清单，确认不是只有 adapter。
5. 检查配置、Tokenizer、权重分片和索引。
6. 用独立进程只传 merged path 做最小加载测试。

## 13. 第 2 周最终交付清单

- [ ] Day 6：三个原始子集、双格式转换、原始数据归档清单、转换与验证脚本。
- [ ] Day 7：可独立运行的 `clean_pipeline.py`、不少于 1500 条的清洗数据、SimHash 结果、统计和 matplotlib 图表。
- [ ] Day 8：官方 `qwen_lora.yaml` 副本、逐行中文注释配置、数据注册、首次 SFT 日志。
- [ ] Day 9：每步 loss CSV、TensorBoard loss 截图、adapter 清单、export 日志、合并模型文件夹清单、3 个新问题测试记录。
- [ ] Day 10：《第 2 周：数据与 SFT 入门报告》、FAQ、验收矩阵和最终文件清单。
- [ ] 验收 1：清洗脚本可通过 CLI 独立运行。
- [ ] 验收 2：训练集不少于 1500 条且格式正确。
- [ ] 验收 3：SFT loss 整体稳定下降。
- [ ] 验收 4：相同条件下，合并模型三问质量证据优于基座。
- [ ] 验收 5：周报和 FAQ 已提交。
- [ ] 全部 README 链接有效，核心结论能定位到脚本、配置、日志、CSV 或截图。
- [ ] 仓库不含模型权重、checkpoint、缓存、环境目录、密钥或其他敏感信息。

## 14. 开始 Day 6 前的待确认项

下午正式开始 Day 6 时，先确认以下事项，再运行下载：

1. 三个数据集的准确官方仓库、revision、split 和许可。
2. ShareGPT-zh 指定的是哪一个上游数据集，避免同名或近似名称混淆。
3. COIG-PC 使用哪个子集或配置。
4. 抽样使用固定随机抽样还是老师要求的前 N 条；若无额外要求，采用 `seed=42` 的确定性随机抽样。
5. 原始子集能否提交 Git；若许可或体积不允许，只提交 manifest、脚本、哈希和必要样例。
6. 训练继续使用 Week 1 的 Qwen2.5-7B-Instruct、LLaMA-Factory 0.9.3 和 RTX 3090。

这些项目确认前不开始下载，避免选错同名数据源后返工。

## 15. 参考资料记录原则

Day 6–Day 9 实际执行时，只引用并固定以下一手来源：

- 三个数据集的官方数据卡或官方仓库。
- 当前安装版本对应的 LLaMA-Factory 官方仓库、`qwen_lora.yaml`、数据格式说明和 export 示例。
- Qwen2.5-7B-Instruct 官方模型配置和 Tokenizer 文件。
- 当前环境中实际安装的 Transformers、PEFT 和 LLaMA-Factory 版本信息。

每个外部来源在报告中记录访问日期、URL 和 revision。博客或二手教程可以帮助理解，但不能替代配置字段、数据格式和命令兼容性的官方证据。
