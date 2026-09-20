# Week 2 Day 6：数据收集与格式统一

Day 6 已完成。三个固定 revision 的中文指令数据源分别确定性抽样
2000、2000、1000 条，共得到 5000 条语义样本；同一批样本分别转换成
5000 条标准 Alpaca 数据和 5000 条标准 ShareGPT 数据。独立校验结果为
`valid: true`，23 项单元测试全部通过。

本目录保存的是 Day 6 的原始归档子集和格式统一结果，不是 Day 7
清洗后的训练集。HTML 清理、控制字符清理、2048-token 截断和模糊去重
均尚未执行。

## 1. 老师要求与完成情况

| 老师要求 | 实际结果 | 证据 |
|---|---|---|
| Alpaca-GPT4-zh 取 2k 条 | 2000 条 | `source/data/raw/alpaca_gpt4_zh_2k.jsonl` |
| COIG-PC 取 2k 条 | 2000 条 | `source/data/raw/coig_pc_2k.jsonl` |
| ShareGPT-zh 取 1k 条 | 1000 条 | `source/data/raw/sharegpt_zh_1k.jsonl` |
| 统一为 Alpaca 格式 | 同一批 5000 条，字段仅为 `instruction`、`input`、`output` | `source/data/formatted/week2_5k_alpaca.jsonl` |
| 统一为 ShareGPT 格式 | 同一批 5000 条，字段仅为 `conversations` | `source/data/formatted/week2_5k_sharegpt.jsonl` |
| 原始数据归档清单 | CSV、Markdown 和机器可读 metadata 均已生成 | `source/manifests/` |

两份格式文件是同一批 5000 条语义样本的等价表示，不能相加并称为
10000 条训练数据。`day6_provenance.jsonl` 用稳定 `sample_id` 将两种表示
逐行一一对应。

## 2. 数据源与固定版本

| 数据源 | 本次选择 | revision | 上游条数 | 抽样条数 | 许可/条件 |
|---|---|---|---:|---:|---|
| [Alpaca-GPT4-zh](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh) | `train` / `alpaca_gpt4_data_zh.json` | `065394ee242c43928298de8f43a8748ffd16f3e3` | 42677 | 2000 | Apache-2.0 |
| [COIG-PC](https://huggingface.co/datasets/BAAI/COIG-PC) | `Top200PerTask` / 对应 Parquet 分片 | `4ca2778d69d7a9c17a0a6c1668c6e989b102176b` | 253558 | 2000 | 子数据集声明优先；未声明时默认 Apache-2.0；仓库访问条件同时适用 |
| [ShareGPT Chinese](https://huggingface.co/datasets/FreedomIntelligence/sharegpt-chinese) | `train` / `sharegpt-chinese.json` | `ef0a611d7c8c3cd7b17c152d4166c67a92015c2b` | 30015 | 1000 | Apache-2.0 |

ShareGPT 中文源沿用
`FreedomIntelligence/sharegpt-chinese`。它提供可固定 revision 的原始多轮
`conversations`、明确的数据卡和许可，适合验证多轮到两种目标格式的转换。
用户找到的 `kimnt93/zh-sharegpt` 是可行备选，但本次不混用两个来源，以免
改变已经确认的 1k 样本口径和数据血缘。

每个上游文件的文件名、字节数和 LFS SHA-256，以及每个本地原始子集的
字节数和 SHA-256，详见
[`raw_data_manifest.md`](source/manifests/raw_data_manifest.md) 和
[`raw_data_manifest.csv`](source/manifests/raw_data_manifest.csv)。

## 3. 抽样方法

抽样固定 `seed=42`。对每条上游记录的原始索引计算：

```text
SHA256("42:<source_name>:<source_index>")
```

每个来源选择哈希优先级最低的目标数量，再按原始索引升序写出。因此固定
revision、源文件和 seed 后，抽样集合与顺序都可复现。原始子集保留上游
字段，并增加 `_provenance`，记录来源、原始索引、revision、seed 和抽样
优先级哈希；模型可见的格式文件不混入这些血缘字段。

## 4. 格式转换规则

### 4.1 Alpaca 来源

- 原始 `instruction`、`input`、`output` 原样映射到标准 Alpaca 三字段；
  缺失 `input` 时使用空字符串。
- 转成 ShareGPT 时，非空 `input` 以
  `instruction + "\n\n" + input` 组成 human 消息，`output` 组成紧随其后的
  gpt 消息。

### 4.2 ShareGPT 来源

- `human`/`user` 统一为 `human`，`gpt`/`assistant` 统一为 `gpt`，消息文本
  不做内容清洗。
- ShareGPT 版保留全部完整且严格交替的多轮消息。
- 转成 Alpaca 时，最后一个完整 human-gpt 对分别成为 `instruction` 和
  `output`；更早的完整轮次按带角色前缀的固定文本序列化到 `input`。
  因而不会只取首轮而静默丢弃中间历史。

第一次严格转换发现抽中的 1000 条 ShareGPT 数据中，有 10 条在若干完整
对话轮后附带一个没有 assistant 回答的末尾 human 消息。处理原则是：

1. 原始 `raw/` 记录保持不变；
2. 仅在两个训练可见格式中去掉这个无法组成监督目标的末尾消息；
3. 所有先前完整轮次全部保留；
4. 在对应 provenance 的 `format_adjustments` 中记录
   `dropped_trailing_unanswered_human`。

受影响的上游原始索引为：
`175, 1744, 3971, 6532, 13286, 19638, 24316, 25479, 26488, 28093`。
除此之外的角色错序、非字符串消息或无法形成任何完整问答对仍会直接报错，
不会静默修复。

## 5. 文件结构

```text
deliverables/week2/day6/
├── README.md
└── source/
    ├── data/
    │   ├── raw/          # 三个带原始字段和 provenance 的固定抽样子集
    │   ├── formatted/    # 同一批 5000 条的 Alpaca/ShareGPT 等价表示
    │   └── interim/      # 两种表示共用的稳定 sample_id 血缘映射
    ├── manifests/        # 来源、许可、revision、大小和哈希归档
    ├── results/          # 命令记录、独立校验和全文件哈希
    ├── scripts/          # 采集、转换、验证三个独立 CLI
    └── tests/            # 23 项单元测试
```

主要入口：

- [`collect_day6.py`](source/scripts/collect_day6.py)：固定文件、固定 revision
  下载并确定性抽样。
- [`convert_formats.py`](source/scripts/convert_formats.py)：将三源数据统一成
  两种标准格式并生成血缘映射。
- [`validate_day6.py`](source/scripts/validate_day6.py)：独立检查计数、字段、
  角色顺序、一一对应、清单大小与哈希。
- [`day6_validation.json`](source/results/day6_validation.json)：最终验证结果。
- [`day6_files.sha256`](source/results/day6_files.sha256)：交付文件哈希清单。
- [`day6_collection.txt`](source/results/day6_collection.txt)：实际命令与结果。

## 6. 从缓存或网络复现

以下命令在仓库根目录执行。本机 `datasets==2.12.0` 的旧依赖链与当前
`urllib3` 不兼容，所以采集脚本有意使用
`huggingface_hub + json/pyarrow` 直接读取固定上游文件，没有为完成任务而
升级或降级共享 Python 环境。

```bash
/Users/yifanren/anaconda3/bin/python \
  deliverables/week2/day6/source/scripts/collect_day6.py \
  --raw-dir deliverables/week2/day6/source/data/raw \
  --cache-dir .cache/day6-huggingface \
  --manifest-json deliverables/week2/day6/source/manifests/collection_metadata.json \
  --seed 42

/Users/yifanren/anaconda3/bin/python \
  deliverables/week2/day6/source/scripts/convert_formats.py \
  --raw-dir deliverables/week2/day6/source/data/raw \
  --formatted-dir deliverables/week2/day6/source/data/formatted \
  --provenance-file \
    deliverables/week2/day6/source/data/interim/day6_provenance.jsonl

/Users/yifanren/anaconda3/bin/python \
  deliverables/week2/day6/source/scripts/validate_day6.py \
  --root deliverables/week2/day6 \
  --output deliverables/week2/day6/source/results/day6_validation.json

/Users/yifanren/anaconda3/bin/python -m pytest \
  deliverables/week2/day6/source/tests/test_day6_pipeline.py -v
```

COIG-PC 是 gated 数据集，首次联网复现前需要在 Hugging Face 接受其条款并
使用只读凭据登录。凭据由 Hugging Face 本机配置管理，不写入本仓库；
`.cache/day6-huggingface` 也不属于交付内容。

## 7. 最终核对

- 原始子集：2000 + 2000 + 1000 = 5000。
- 标准 Alpaca：5000。
- 标准 ShareGPT：5000。
- provenance：5000 个唯一 `sample_id`。
- 单元测试：23/23 通过。
- 独立验证：`valid: true`，`errors: []`。
- Day 7 清洗：尚未执行。
