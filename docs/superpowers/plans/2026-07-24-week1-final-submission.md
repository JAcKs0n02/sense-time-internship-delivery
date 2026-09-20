# 第 1 周最终提交整理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建一个包含 19 个必要文件、可直接交给大模型方向导师的第 1 周提交目录。

**Architecture:** 保留现有 `deliverables/` 和 `archive/` 不变，从已核验的正式交付中复制必要材料。只改写提交目录中的三份 Markdown，其他文件逐字节保持原样。

**Tech Stack:** Markdown、Python 3、Jupyter Notebook JSON、Git、SHA-256

## Global Constraints

- 目标目录固定为 `第1周_最终提交_2026-07-24/`。
- 最终文件数固定为 19。
- 不移动或删除仓库已有文件。
- 不修改图片、脚本、配置、CSV、日志、模型回答或 Notebook。
- 文风面向大模型领域导师，使用直接的实验记录表达。

---

### Task 1: 建立目录并复制原始材料

**Files:**

- Create: `第1周_最终提交_2026-07-24/`
- Copy from: `deliverables/week1/day1/`
- Copy from: `deliverables/week1/day2/`
- Copy from: `deliverables/week1/day3/`
- Copy from: `deliverables/week1/day4/`
- Copy from: `deliverables/week1/day5/`

- [ ] **Step 1: 创建五个按日期划分的目录**

Run:

```bash
mkdir -p \
  第1周_最终提交_2026-07-24/Day1_环境初始化 \
  第1周_最终提交_2026-07-24/Day2_模型下载与原生推理 \
  第1周_最终提交_2026-07-24/Day3_架构分析 \
  第1周_最终提交_2026-07-24/Day4_Tokenizer实验 \
  第1周_最终提交_2026-07-24/Day5_LLaMA-Factory与周报
```

Expected: 五个目录存在，目录内尚无文件。

- [ ] **Step 2: 复制 16 个不需要改写的文件**

Run:

```bash
cp deliverables/week1/day1/evidence/day1_environment.png 第1周_最终提交_2026-07-24/Day1_环境初始化/
cp deliverables/week1/day1/evidence/day1_toolchain.png 第1周_最终提交_2026-07-24/Day1_环境初始化/
cp deliverables/week1/day2/evidence/00_model_download.jpg 第1周_最终提交_2026-07-24/Day2_模型下载与原生推理/model_download.jpg
cp deliverables/week1/day2/source/scripts/inference.py 第1周_最终提交_2026-07-24/Day2_模型下载与原生推理/
cp deliverables/week1/day2/source/results/day2_code_generation.txt 第1周_最终提交_2026-07-24/Day2_模型下载与原生推理/code_generation.txt
cp deliverables/week1/day2/source/results/day2_logic_reasoning.txt 第1周_最终提交_2026-07-24/Day2_模型下载与原生推理/logic_reasoning.txt
cp deliverables/week1/day2/source/results/day2_role_play.txt 第1周_最终提交_2026-07-24/Day2_模型下载与原生推理/role_play.txt
cp deliverables/week1/day2/source/results/day2_chat_template.txt 第1周_最终提交_2026-07-24/Day2_模型下载与原生推理/chat_template.txt
cp deliverables/week1/day3/source/config/config.json 第1周_最终提交_2026-07-24/Day3_架构分析/
cp deliverables/week1/day3/source/scripts/analyze_params.py 第1周_最终提交_2026-07-24/Day3_架构分析/
cp deliverables/week1/day3/source/results/day3_parameter_counts.csv 第1周_最终提交_2026-07-24/Day3_架构分析/parameter_counts.csv
cp deliverables/week1/day4/tokenizer_experiments.executed.ipynb 第1周_最终提交_2026-07-24/Day4_Tokenizer实验/
cp deliverables/week1/day5/configs/qwen25_7b_identity_qlora_run2.yaml 第1周_最终提交_2026-07-24/Day5_LLaMA-Factory与周报/
cp deliverables/week1/day5/source/results/day5_train_run2.txt 第1周_最终提交_2026-07-24/Day5_LLaMA-Factory与周报/
cp deliverables/week1/day5/evidence/training_complete.png 第1周_最终提交_2026-07-24/Day5_LLaMA-Factory与周报/
cp deliverables/week1/day5/evidence/tensorboard_loss.png 第1周_最终提交_2026-07-24/Day5_LLaMA-Factory与周报/
```

Expected: 16 个文件复制完成，原文件仍在原路径。

### Task 2: 编写三份提交文档

**Files:**

- Create: `第1周_最终提交_2026-07-24/提交说明.md`
- Create: `第1周_最终提交_2026-07-24/Day3_架构分析/Qwen2.5架构分析报告.md`
- Create: `第1周_最终提交_2026-07-24/Day5_LLaMA-Factory与周报/第1周总结报告.md`

- [ ] **Step 1: 编写提交说明**

内容固定包含：实验环境、Day 1–Day 5 文件入口、五项验收对应关系、Day 2 代码回答测试失败说明。全文不使用营销式或验收口号。

- [ ] **Step 2: 编写 Day 3 架构报告**

内容固定包含：

- 配置字段表：`vocab_size=152064`、`hidden_size=3584`、`num_hidden_layers=28`、`num_attention_heads=28`、`num_key_value_heads=4`、`rope_theta=1000000`。
- 28Q/4KV 的 GQA 分组关系与 KV Cache 元素比例 `1/7`。
- RoPE 的 `R(n-m)` 相对位置解释，以及 `rope_theta` 不等于上下文长度。
- 单层参数量 `233057792`、总参数量 `7615616512` 的公式。
- 与原始 Meta-Llama-3-8B 的层数、宽度、头数、词表、FFN、RoPE 和 bias 对比。
- 只比较架构，不宣称未经同硬件测试的性能差异。

- [ ] **Step 3: 编写第 1 周总结报告**

内容固定包含：硬件软件环境、Day 1–Day 5 实验过程、三类推理结果、Tokenizer 结果、QLoRA 配置与 Run 2 指标、问题处理、对 GQA/RoPE 的理解及下周计划。保留失败与局限。

### Task 3: 验证最终提交

**Files:**

- Verify: `第1周_最终提交_2026-07-24/`

- [ ] **Step 1: 核对文件数和目录结构**

Run:

```bash
find 第1周_最终提交_2026-07-24 -type f | sort
find 第1周_最终提交_2026-07-24 -type f | wc -l
```

Expected: 文件清单与设计一致，总数为 `19`。

- [ ] **Step 2: 核对未修改副本的 SHA-256**

Run: 对 Task 1 中每个源文件和目标文件执行 `shasum -a 256`。

Expected: 16 组源文件与目标文件哈希逐组相同。

- [ ] **Step 3: 验证 Python、JSON 和 Notebook**

Run:

```bash
python3 -m py_compile \
  第1周_最终提交_2026-07-24/Day2_模型下载与原生推理/inference.py \
  第1周_最终提交_2026-07-24/Day3_架构分析/analyze_params.py
jq empty 第1周_最终提交_2026-07-24/Day3_架构分析/config.json
jq empty 第1周_最终提交_2026-07-24/Day4_Tokenizer实验/tokenizer_experiments.executed.ipynb
```

Expected: 三个命令均为退出码 `0`。

- [ ] **Step 4: 重新运行 Day 3 参数统计**

Run:

```bash
python3 第1周_最终提交_2026-07-24/Day3_架构分析/analyze_params.py \
  --config 第1周_最终提交_2026-07-24/Day3_架构分析/config.json \
  --output /tmp/week1_submission_parameter_counts.csv
```

Expected: 输出包含 `decoder_layer_parameters=233057792` 和 `total_parameters=7615616512`，生成 CSV 与提交副本一致。

- [ ] **Step 5: 检查 Notebook 和训练日志状态**

Run:

```bash
jq -e '[.cells[] | select(.cell_type == "code") | .execution_count] | all(. != null)' \
  第1周_最终提交_2026-07-24/Day4_Tokenizer实验/tokenizer_experiments.executed.ipynb
jq -e '[.cells[].outputs[]? | select(.output_type == "error")] | length == 0' \
  第1周_最终提交_2026-07-24/Day4_Tokenizer实验/tokenizer_experiments.executed.ipynb
rg -n 'training_exit_code=0' \
  第1周_最终提交_2026-07-24/Day5_LLaMA-Factory与周报/day5_train_run2.txt
```

Expected: 两个 `jq` 检查返回 `true`，训练日志找到成功退出标志。

- [ ] **Step 6: 检查文风与 Markdown 链接**

Run:

```bash
rg -n '全面完成|三重验证|核心设计哲学|最终结论|赋能|闭环|一站式|综上所述' \
  第1周_最终提交_2026-07-24 -g '*.md'
```

Expected: 无匹配。逐一确认三份 Markdown 中的相对链接均指向存在的文件。

- [ ] **Step 7: 提交变更**

Run:

```bash
git add docs/superpowers/plans/2026-07-24-week1-final-submission.md 第1周_最终提交_2026-07-24
git commit -m "docs: prepare week 1 final submission"
```

Expected: Git 提交成功，工作区干净。
