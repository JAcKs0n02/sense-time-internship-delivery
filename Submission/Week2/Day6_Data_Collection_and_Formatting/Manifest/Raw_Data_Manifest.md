# Day 6 原始数据归档清单

- 固定抽样种子：`42`
- 归档生成时间：`2026-07-27T18:13:23+08:00`
- 抽样语义样本总数：`5000`

| 数据集 | revision | split | 许可/条件 | 上游条数 | 抽样条数 | 原始子集 | SHA-256 |
|---|---|---|---|---:|---:|---|---|
| [Alpaca-GPT4-zh](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh) | `065394ee242c43928298de8f43a8748ffd16f3e3` | `train` | Apache-2.0 | 42677 | 2000 | `source/data/raw/alpaca_gpt4_zh_2k.jsonl` | `57e9714f2ba02ca0323a062b72af92da1de67d1161bc71828e0638dfc8b4ed84` |
| [COIG-PC Top200PerTask](https://huggingface.co/datasets/BAAI/COIG-PC) | `4ca2778d69d7a9c17a0a6c1668c6e989b102176b` | `Top200PerTask` | Sub-dataset license takes precedence; Apache-2.0 applies by default when a sub-dataset has no declared license; repository access conditions also apply | 253558 | 2000 | `source/data/raw/coig_pc_2k.jsonl` | `c78f4ae29f9891a7dfdebf4b054085d71f8c4f5f38d95123590f54a19ee5e81c` |
| [ShareGPT Chinese](https://huggingface.co/datasets/FreedomIntelligence/sharegpt-chinese) | `ef0a611d7c8c3cd7b17c152d4166c67a92015c2b` | `train` | Apache-2.0 | 30015 | 1000 | `source/data/raw/sharegpt_zh_1k.jsonl` | `e7fb562bbe4ccf884c3968f2536b5a5d53b4ad48e72688e95b77203695f6e092` |

## 上游文件固定信息

| 数据集 | 上游文件 | 文件大小（bytes） | 上游文件 SHA-256 |
|---|---|---:|---|
| Alpaca-GPT4-zh | `alpaca_gpt4_data_zh.json` | 27862380 | `628c3b100e32af85ec8d5338c42745f6b82900f97cf4deeda307b98f94ab34ab` |
| COIG-PC Top200PerTask | `data/Top200PerTask-00000-of-00001-fe36897020230691.parquet` | 160399530 | `59a961b3517c3a46c572c6f031436552f863d3ee0331728901725ac668105219` |
| ShareGPT Chinese | `sharegpt-chinese.json` | 220056589 | `bacd961782259b43364306885a984487964379d5f7aeec54c25814574ffe6855` |

## 抽样口径

lowest SHA-256 priority of '42:<source>:<source_index>', then source_index ascending

所有 `raw/` 文件都是固定 revision 上游记录的确定性子集；
Day 6 未执行 HTML 清理、控制字符清理、token 截断或去重。
