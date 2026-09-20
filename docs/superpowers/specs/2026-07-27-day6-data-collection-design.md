# Day 6 数据收集与格式统一设计

## 1. 目标

完成老师要求的三个中文指令数据源收集：

- `llamafactory/alpaca_gpt4_zh`：2,000 条；
- `BAAI/COIG-PC`：2,000 条；
- `FreedomIntelligence/sharegpt-chinese`：1,000 条完整对话。

同一批 5,000 个语义样本分别生成标准 Alpaca 表示和标准 ShareGPT 表示，并提交可复核的原始数据归档清单。Day 6 只负责来源确认、下载、抽样和格式统一，不执行 Day 7 的 HTML 清理、控制字符清理、token 截断或模糊去重。

## 2. 固定数据源

| 数据集 | Hugging Face 仓库 | 固定 revision | 目标数量 |
|---|---|---|---:|
| Alpaca-GPT4-zh | `llamafactory/alpaca_gpt4_zh` | `065394ee242c43928298de8f43a8748ffd16f3e3` | 2,000 |
| COIG-PC | `BAAI/COIG-PC` | `4ca2778d69d7a9c17a0a6c1668c6e989b102176b` | 2,000 |
| ShareGPT-zh | `FreedomIntelligence/sharegpt-chinese` | `ef0a611d7c8c3cd7b17c152d4166c67a92015c2b` | 1,000 |

COIG-PC 使用官方仓库中每个任务最多保留 200 条的 `Top200PerTask` 数据文件，避免为 2,000 条样本下载 127 GB 全量仓库。其页面要求 Hugging Face 登录并接受使用声明；用户手动登录，凭据不进入脚本、日志或仓库。

## 3. 目录与组件

```text
deliverables/week2/day6/
├── README.md
└── source/
    ├── data/
    │   ├── raw/
    │   │   ├── alpaca_gpt4_zh_2k.jsonl
    │   │   ├── coig_pc_2k.jsonl
    │   │   └── sharegpt_zh_1k.jsonl
    │   ├── interim/
    │   │   └── day6_provenance.jsonl
    │   └── formatted/
    │       ├── week2_5k_alpaca.jsonl
    │       └── week2_5k_sharegpt.jsonl
    ├── manifests/
    │   ├── raw_data_manifest.csv
    │   └── raw_data_manifest.md
    ├── results/
    │   ├── day6_collection.txt
    │   ├── day6_validation.json
    │   └── day6_files.sha256
    ├── scripts/
    │   ├── collect_day6.py
    │   ├── convert_formats.py
    │   └── validate_day6.py
    └── tests/
        └── test_day6_pipeline.py
```

`collect_day6.py` 负责固定 revision 下载、解析、确定性抽样及原始子集落盘；`convert_formats.py` 只负责字段映射；`validate_day6.py` 对交付目录执行独立只读验收。测试使用小型本地 fixture，不依赖网络。

## 4. 数据流与抽样

1. 从固定 revision 获取上游文件，不执行远程代码。
2. 为每条记录附加 `source_index`，它表示该记录在固定上游文件中的原始顺序。
3. 使用 `seed=42` 的稳定哈希优先级：

   ```text
   sha256("<seed>:<source_name>:<source_index>")
   ```

4. 选择优先级最小的目标数量记录，再按 `source_index` 升序写出。
5. 原始子集保留上游字段，并增加 `_provenance`；不在 Day 6 修改文本内容。
6. 三个子集分别严格为 2,000、2,000、1,000 条。

稳定哈希抽样不依赖 Python `random` 的版本细节，同一上游 revision、种子和索引集合应产生完全相同的子集。

## 5. 格式转换

### 5.1 Alpaca

最终训练可见字段严格为：

```json
{"instruction": "...", "input": "", "output": "..."}
```

- 原生 Alpaca/COIG-PC：直接映射 `instruction`、`input`、`output`。
- ShareGPT 单轮：用户消息映射到 `instruction`，`input=""`，助手消息映射到 `output`。
- ShareGPT 多轮：最后一个完整 human→gpt 回合是训练目标；目标 human 消息写入 `instruction`，此前完整对话按固定的 `[human]`/`[gpt]` 行标签序列化到 `input`，目标 assistant 消息写入 `output`。

### 5.2 ShareGPT

最终训练可见字段严格为：

```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

- 原生 Alpaca/COIG-PC：非空 `input` 使用 `instruction + "\n\n" + input` 组成 human 消息；空 `input` 时只使用 `instruction`。
- ShareGPT：显式规范角色为 `human` 与 `gpt`，保留所有可形成完整 human→gpt 配对的轮次。
- system、tool、连续同角色、首条 assistant、末尾未回答 human 等异常结构不静默猜测；收集阶段记录结构审计，无法形成至少一组完整对话的记录判为格式错误并停止。

训练文件不混入来源字段。`sample_id`、来源仓库、revision、原始索引和原始记录哈希写入独立的 `day6_provenance.jsonl`，并与两份 5k 文件保持相同行序。

## 6. 校验与错误处理

独立校验器必须检查：

- 原始子集计数恰为 2,000、2,000、1,000；
- 两份格式化文件都恰为 5,000 条；
- Alpaca 每条只有 `instruction`、`input`、`output`，且类型正确；
- ShareGPT 每条只有 `conversations`，对话非空、长度为偶数、角色严格交替；
- provenance 恰为 5,000 条，`sample_id` 唯一；
- 两种格式和 provenance 的行数与顺序一一对应；
- manifest 的三个数据源、固定 revision、数量、路径、大小和 SHA-256 完整；
-所有 JSONL 均能逐行解析，所有清单内哈希与磁盘文件一致。

下载失败、授权失败、上游字段变化、有效记录不足或哈希不一致时立即失败，不使用其他数据源静默补齐。失败日志保留真实错误信息，但必须清除凭据。

## 7. 交付文档

`raw_data_manifest.csv` 提供机器可读归档；`raw_data_manifest.md` 解释许可、字段、映射和限制；`README.md` 汇总实际数量、文件入口、复现命令、Day 6 与 Day 7 的边界以及验收结果。最终生成 `day6_files.sha256`，但不把 Hugging Face 缓存、访问凭据或完整 127 GB COIG-PC 仓库加入 Git。

## 8. 自审结论

- 没有未决占位符。
- 三个来源、三个目标数量、两个目标格式和一个归档清单均有明确产物。
- “5,000 个样本的两种表示”不会被误记为 10,000 个训练样本。
- `FreedomIntelligence/sharegpt-chinese` 已按用户确认锁定；不再使用 `kimnt93/zh-sharegpt`。
- Day 7 清洗操作明确排除在本设计之外。
