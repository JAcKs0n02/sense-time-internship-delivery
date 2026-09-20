# Day 4：Tokenizer 与分词实验

## 完成结论

Day 4 已完成。实验对象为 `Qwen/Qwen2.5-7B-Instruct` 的真实 Tokenizer，交付的 Jupyter Notebook 已从头无交互执行成功。实验覆盖 15 个固定极端用例、三个 Qwen 特殊令牌、Chat Template 生成前缀、长文本左右截断，以及 ByteLevel BPE 与 SentencePiece Unigram 的受控比较。

核心结果：

- 15 个固定用例全部生成字符数、UTF-8 字节数、token 数、token IDs、token 文本、decode 和 round-trip 记录。
- `<|endoftext|>`、`<|im_start|>`、`<|im_end|>` 的 token ID 分别为 `151643`、`151644`、`151645`。
- `add_generation_prompt=True` 比 `False` 增加 3 个 token，对应文本为 `<|im_start|>assistant\n`。
- 长中文样本截断前为 780 个 token；左截断和右截断后均严格为 64 个 token。
- ByteLevel BPE 与 SentencePiece Unigram 使用同一份 2328 行语料，目标词表与实际词表均为 800。
- 四组受控对比样本在两种分词器中均为 0 个 `<unk>`，并且都能正确 round-trip。
- 执行版 Notebook 的全部代码单元均有执行编号，不包含 traceback 或 error output。

## 老师要求逐项对应

| 老师要求 | 完成情况 | 文件入口 |
|---|---|---|
| 测试中英文混合、特殊符号、数学公式、长文本截断 | 已完成；15 个固定用例覆盖这些类别及 Unicode、罕见 CJK、emoji、全半角、空白控制符、代码等边界输入 | [执行版 Notebook](tokenizer_experiments.executed.ipynb)、[极端用例 CSV](source/results/tokenizer_extreme_cases.csv)、[截断结果](source/results/truncation_results.json) |
| 观察 `<|endoftext|>`、`<|im_start|>` 等特殊令牌的编码与解码 | 已完成；同时区分协议令牌处理与普通字面量拆分，并比较 `skip_special_tokens=True/False` | [特殊令牌结果](source/results/special_token_results.json) |
| 对比 BPE 与 SentencePiece 在中文分词上的表现 | 已完成；明确 SentencePiece 是框架，本实验选择 Unigram，并与 ByteLevel BPE 使用同语料、同目标及实际词表大小比较 | [对比 CSV](source/results/tokenizer_comparison.csv)、[训练语料](source/data/tokenizer_comparison_corpus.txt) |
| 提交包含 10 个以上极端用例的 Jupyter Notebook | 已完成；Notebook 固定包含 15 个用例，并已从头执行成功 | [源 Notebook](tokenizer_experiments.ipynb)、[执行版 Notebook](tokenizer_experiments.executed.ipynb) |

## 实验设计

### 统一观测口径

15 个用例均调用同一个观测函数，避免不同样本使用不同统计逻辑。每条记录包含：

- 原始文本、类别和固定 ID。
- 字符数与 UTF-8 字节数。
- token 数、token IDs 和 token 字符串。
- 保留特殊令牌的 decode 结果。
- decode 后是否与原始字符串逐码点相等。

`unicode_composition` 是唯一 `roundtrip_equal=False` 的 Qwen 用例。原始字符串同时包含预组合字符 `é/Å` 和 combining 写法 `e + U+0301`、`A + U+030A`；decode 后 combining 写法被规范化为预组合字符。文本语义与显示内容不变，但 Unicode 码点序列不同，因此如实记录为 `False`，没有修改实验结果。

### 特殊令牌与 Chat Template

特殊令牌实验分别记录：

- `convert_tokens_to_ids` 得到的词表 ID。
- 默认 Tokenizer 将字符串视为协议令牌时的编码。
- `split_special_tokens=True` 时将同一字符串作为普通文本拆分的编码。
- `skip_special_tokens=False/True` 的解码差异。
- 同一组 messages 在 `add_generation_prompt=False/True` 下的完整模板、token IDs 与新增后缀。

### 截断实验

长中文样本固定使用 `max_length=64`。实验分别设置 `truncation_side="right"` 与 `"left"`，保存截断前 token 数、截断后 token 数、保留文本、最后一个 token ID 和 token 字符串，之后恢复原始截断方向。

### BPE 与 SentencePiece 对比

SentencePiece 不是单一分词算法，而是可训练 BPE 或 Unigram 等模型的框架。本实验选择：

- `ByteLevel BPE`：ByteLevel pre-tokenizer、ByteLevel decoder、字节初始字母表。
- `SentencePiece Unigram`：`model_type=unigram`、`byte_fallback=True`、`hard_vocab_limit=True`。

双方读取同一份 UTF-8 语料，并使用 `vocab_size=800`。最终验证器同时检查目标词表和实际词表都为 800。比较指标包括中文 token/char、英文 token/空格词、混合文本 token 数、`<unk>` 数和 round-trip。结果只说明这份固定小语料与当前配置下的差异，不推广为所有 BPE 或 SentencePiece 模型的普遍结论；语料、normalizer、pre-tokenizer、byte fallback 和词表大小都会改变结果。

## 交付文件

```text
day4/
├── README.md
├── tokenizer_experiments.ipynb
├── tokenizer_experiments.executed.ipynb
├── evidence/
│   ├── 01_extreme_cases.jpg
│   ├── 02_special_tokens.jpg
│   ├── 03_truncation.jpg
│   ├── 04a_tokenizer_comparison_algorithms.jpg
│   ├── 04b_tokenizer_comparison_vocab.jpg
│   ├── 04c_tokenizer_comparison_metrics.jpg
│   ├── 05_execution_success.jpg
│   └── 06_final_validation.jpg
└── source/
    ├── data/tokenizer_comparison_corpus.txt
    ├── results/
    │   ├── day4_environment.txt
    │   ├── day4_files.sha256
    │   ├── day4_final_validation.json
    │   ├── day4_final_validation_console.txt
    │   ├── day4_local_validation.json
    │   ├── day4_nbconvert.txt
    │   ├── day4_notebook_completion.txt
    │   ├── day4_preflight.txt
    │   ├── special_token_results.json
    │   ├── tokenizer_comparison.csv
    │   ├── tokenizer_extreme_cases.csv
    │   └── truncation_results.json
    ├── scripts/
    │   ├── build_notebook.py
    │   └── validate_day4.py
    └── tests/test_day4_artifacts.py
```

`source/results/` 中除 `day4_local_validation.json` 外的结果文件均由 AutoDL 上的执行版 Notebook 或验证命令直接生成，并按 `day4_files.sha256` 校验后复制到本仓库。复制过程中未修改结果内容。

## 复现与验证

重新生成源 Notebook 和固定语料：

```bash
python3 deliverables/week1/day4/source/scripts/build_notebook.py \
  --notebook deliverables/week1/day4/tokenizer_experiments.ipynb \
  --corpus deliverables/week1/day4/source/data/tokenizer_comparison_corpus.txt
```

在具有 `llm_exp` 内核和本地 Qwen Tokenizer 文件的实验环境中执行：

```bash
cd /root/autodl-tmp/qwen25-week1/week1/day4
DAY4_ROOT="$PWD" \
QWEN_MODEL_DIR=/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct \
DAY4_OUTPUT_DIR="$PWD/source/results" \
/root/autodl-tmp/conda/envs/llm_exp/bin/python -m jupyter nbconvert \
  --to notebook --execute tokenizer_experiments.ipynb \
  --output tokenizer_experiments.executed.ipynb --output-dir . \
  --ExecutePreprocessor.timeout=900
```

本地测试与只读验收：

```bash
python3 -m unittest discover \
  -s deliverables/week1/day4/source/tests -v

python3 deliverables/week1/day4/source/scripts/validate_day4.py \
  --day4-root deliverables/week1/day4 \
  --notebook deliverables/week1/day4/tokenizer_experiments.executed.ipynb

cd deliverables/week1/day4
sha256sum -c source/results/day4_files.sha256
```

最终结果：4 项单元测试全部通过；独立验证器 20 项检查全部通过；13 个远端原始文件的 SHA-256 全部匹配。

## 证据图

15 个固定极端用例：

![15 个固定极端用例](evidence/01_extreme_cases.jpg)

特殊令牌与 Chat Template：

![特殊令牌与 Chat Template](evidence/02_special_tokens.jpg)

长文本左右截断：

![长文本截断](evidence/03_truncation.jpg)

相同算法对比样本与分词器：

![对比算法](evidence/04a_tokenizer_comparison_algorithms.jpg)

目标词表与实际词表均为 800：

![实际词表大小](evidence/04b_tokenizer_comparison_vocab.jpg)

Notebook 执行完成：

![Notebook 执行完成](evidence/05_execution_success.jpg)

最终严格验收：

![最终严格验收](evidence/06_final_validation.jpg)

## 验收结论

- [x] Notebook 包含 15 个固定极端用例。
- [x] 中英文混合、特殊符号、数学公式和长文本截断全部覆盖。
- [x] 三个 Qwen 特殊令牌的 ID、协议编码、字面量编码和解码行为已记录。
- [x] `add_generation_prompt=True/False` 的模板与 token 差异已记录并解释。
- [x] BPE 与 SentencePiece 使用相同语料，目标词表和实际词表均为 800。
- [x] Notebook 从头无交互执行成功，全部代码单元有执行编号且无 error output。
- [x] 结论明确说明语料、normalizer、pre-tokenizer、byte fallback 和词表大小的影响。
- [x] 原始结果、执行日志、校验清单与截图均已保存到仓库。
