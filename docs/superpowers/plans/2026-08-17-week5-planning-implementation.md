# Week 5 多模态规划文件 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将老师 Day 22–Day 27 的多模态实践要求整理为一份总执行计划、一个 Week 5 工程索引和六份逐日说明，使后续实验可按统一输入、证据与验收门槛执行。

**Architecture:** `docs/week5_execution_plan.md` 是唯一长期执行入口，负责冻结模型、数据、指标和跨日依赖；`deliverables/week5/README.md` 只做工程归档导航和状态汇总；六个 Day README 分别描述当天要求、术语、步骤、预期文件、完成门禁和故障处理。规划阶段不创建教师提交目录，也不写入任何尚未运行的模型、指标或 PASS 结论。

**Tech Stack:** Markdown、Git、ripgrep、POSIX shell；后续实验栈预注册为 Python 3.10、PyTorch、Transformers、Qwen2-VL、`qwen-vl-utils`、LLaMA-Factory、PEFT、Pillow、NumPy、pandas、matplotlib 和 NVIDIA RTX 3090 24GB。

## Global Constraints

- 老师新版 [`实习需求.pdf`](../../../实习需求.pdf) 中的 Week 5 原始要求具有最高优先级。
- Day 22–Day 27 的状态在本轮统一写为“未开始”；计划值、预期路径和验收门槛不等于实验事实。
- 既有 RTX 3090 为 24GB，按老师规则正式模型固定为 `Qwen/Qwen2-VL-7B-Instruct`；下载时必须固定不可变 revision。
- 五张 Day 22 图片类型固定为表格截图、自然风景、Logo、手写公式和 UI 界面，且必须自建、自拍或许可明确。
- Day 23 固定为 5 张图片乘 5 类问题，共 25 条唯一推理记录，不能删除困难或失败项。
- Day 24 至少产出 3 张可追溯热力图；Qwen2-VL 的实现口径是目标文本 token 对视觉 token 区间的语言模型 self-attention，不伪称存在独立 `cross_attn` 模块。
- Day 25 固定 10 题，`hallucination_rate = hallucination_count / 10`，不能根据回答删题或改分母。
- Day 26 正式训练集为 200 条；另建 20 条隔离评测，二者不得共享图片 SHA-256。
- Day 26 使用 LLaMA-Factory LoRA 并设置 `freeze_vision_tower: true`；先做 processor/schema 和单步 GPU smoke，再正式训练。
- “明显提升”必须同时满足：平均加权分提升至少 `0.50/5`、LoRA 至少胜出 12/20、视觉幻觉率不高于 Base。
- Git 只保存代码、配置、报告、小型数据和审计元数据；模型权重、checkpoint、缓存、凭据和大文件留在 AutoDL 持久化目录。
- 本轮不创建 `Submission/Week5/`；只有对应实验完成且证据通过后才增量建立教师交付。
- 每次使用 AutoDL 后必须确认关机，但教师提交不包含平台关机证据。
- 不得提交账号、密码、SSH 连接信息、私有盲评映射、真实客户数据或个人敏感信息。

---

## File Structure

| 文件 | 单一职责 |
|---|---|
| `docs/week5_execution_plan.md` | 冻结 Week 5 全局范围、逐日依赖、数据合同、技术路线、验收阈值和最终交付结构 |
| `deliverables/week5/README.md` | 提供 Week 5 工程归档入口、六日状态表、目录边界和跨日输入输出关系 |
| `deliverables/week5/day22/README.md` | 说明模型准备、环境确认和五张图片素材包如何实施与验收 |
| `deliverables/week5/day23/README.md` | 说明 25 条图文推理矩阵、结果字段、生成参数和人工审查口径 |
| `deliverables/week5/day24/README.md` | 说明 Qwen2-VL 跨模态注意力定义、Hook 提取、热力图绘制和失败回退 |
| `deliverables/week5/day25/README.md` | 说明 10 题错误前提测试、二元幻觉评分和报告计算方式 |
| `deliverables/week5/day26/README.md` | 说明 200 条训练数据、20 条隔离评测、冻结视觉塔 LoRA 和提升门槛 |
| `deliverables/week5/day27/README.md` | 说明周报结构、验收矩阵、最终 VLM 归档元数据和教师提交边界 |

## Interfaces

- `docs/week5_execution_plan.md` 产生六日共享的模型名、数据数量、字段名、固定阈值和状态词。
- `deliverables/week5/README.md` 消费总计划路径，并产生到六份 Day README 的唯一导航。
- Day 22 产生冻结的 `model_revision`、环境版本和 `image_manifest.csv`；Day 23–Day 25 只能消费这些冻结输入。
- Day 23 产生 25 条能力基线；Day 24 产生至少 3 个可解释性案例；Day 25 产生固定 10 题 Base 幻觉率。
- Day 26 消费 Day 22 的基座与独立训练/评测图片，产生 adapter、训练日志和 Base/LoRA 隔离评测。
- Day 27 只汇总已经验证的 Day 22–Day 26 事实，并投影最终教师提交；不重新解释或改写原始结果。

---

### Task 1: 创建 Week 5 总执行计划

**Files:**
- Create: `docs/week5_execution_plan.md`
- Reference: `docs/superpowers/specs/2026-08-17-week5-multimodal-deliverables-design.md`
- Reference: `docs/week4_execution_plan.md`

**Interfaces:**
- Consumes: 老师 Day 22–Day 27 原始要求、已批准设计规范、RTX 3090 24GB 基线。
- Produces: 六个 Day README 共同引用的模型、数据、指标、文件命名和完成定义。

- [ ] **Step 1: 先验证计划文件尚不存在**

Run:

```bash
test ! -e docs/week5_execution_plan.md
```

Expected: exit code `0`。若文件已存在，先读取并仅补充缺失内容，不覆盖用户已有修改。

- [ ] **Step 2: 创建带固定头部和全局约束的总计划**

使用 `apply_patch` 创建文件。头部必须包含 writing-plans 标准说明、Goal、Architecture、Tech Stack 和 Global Constraints；Global Constraints 原样覆盖本计划顶部的 14 项边界，尤其是 7B 模型、25 条推理、3 张热力图、10 题、200+20 数据和三项提升门槛。

- [ ] **Step 3: 写入老师要求、术语和跨日结构**

按以下顺序写入实际章节：

```text
1. 文档用途与依据
2. 老师原始要求与最终验收
3. 当前已知环境与未验证项
4. 当前进度（Day 22–27 全部未开始）
5. 核心术语
6. Week 5 数据流与目标文件结构
7. 每日通用执行规则
8. Day 22 实施计划
9. Day 23 实施计划
10. Day 24 实施计划
11. Day 25 实施计划
12. Day 26 实施计划
13. Day 27 实施计划
14. 故障处理顺序
15. 最终交付清单
16. 一手参考
```

“核心术语”必须定义 VLM、视觉编码器/ViT、视觉 token、动态分辨率、Hook、self-attention、跨模态注意力、热力图、视觉幻觉、ground truth、LoRA、adapter、冻结视觉塔、隔离评测、model revision 和 SHA-256。

- [ ] **Step 4: 写入精确的数据合同与验收公式**

总计划必须列出：

```text
Day 22 image_manifest.csv:
image_id,relative_path,image_type,source,license,width,height,file_bytes,sha256

Day 23 inference_results.csv:
record_id,image_id,question_type,prompt,raw_response,input_tokens,
output_tokens,latency_seconds,peak_gpu_memory_mib,strength_label,
hallucination_label,reviewer_reason,model_revision,image_sha256,
generation_config_sha256

Day 25:
hallucination_rate = hallucination_count / 10

Day 26:
train_count = 200
heldout_count = 20
mean_gain >= 0.50 / 5
lora_wins >= 12 / 20
lora_hallucination_rate <= base_hallucination_rate
```

同时明确训练集和隔离评测集图片 SHA-256 零重叠，且三项 Day 26 门槛必须同时满足。

- [ ] **Step 5: 验证总计划没有虚构完成状态**

Run:

```bash
rg -n "Day 2[2-7]|未开始|Qwen/Qwen2-VL-7B-Instruct|25 条|3 张|200 条|20 条|0.50/5|12/20" docs/week5_execution_plan.md
```

Expected: 六天、模型、数量和阈值均可定位。

Run:

```bash
rg -n "已完成|训练成功|下载成功|PASS|实际结果" docs/week5_execution_plan.md
```

Expected: 仅能出现在“禁止将预期写成已完成/PASS”或完成定义的否定语境，不出现任何 Week 5 实验已经成功的陈述。

- [ ] **Step 6: 提交总计划**

```bash
git add docs/week5_execution_plan.md
git diff --cached --check
git commit -m "docs: add week5 multimodal execution plan"
```

### Task 2: 创建 Week 5 工程归档索引

**Files:**
- Create: `deliverables/week5/README.md`
- Reference: `deliverables/week4/README.md`
- Reference: `docs/week5_execution_plan.md`

**Interfaces:**
- Consumes: 总计划中的六日状态、输入输出与目录边界。
- Produces: 到 Day 22–Day 27 README 的入口和后续逐日更新的状态表。

- [ ] **Step 1: 创建 Week 5 目录与索引**

使用 `apply_patch` 创建 `deliverables/week5/README.md`。文件必须有以下章节：

```text
# Week 5 Engineering Archive
## 执行入口
## 当前状态
## 固定模型与资源口径
## 跨日数据流
## 目录边界
## 更新规则
```

“当前状态”表包含 Day 22–Day 27 六行，每行状态均为“未开始”，并分别链接 `day22/README.md` 至 `day27/README.md`。老师交付列依次写为：模型下载确认与图片包、25 条推理记录、至少 3 张热力图、幻觉检测报告、微调 VLM 与日志、周报与最终 VLM 归档。

- [ ] **Step 2: 写明工程目录和教师目录边界**

准确写入：

```text
deliverables/week5/  = 完整工程说明与后续实验归档
Submission/Week5/    = 仅收录已完成且验证通过的教师交付，本轮不创建
/root/autodl-tmp/    = 模型、checkpoint、缓存和大文件的远端持久化位置
```

不要写入远端凭据、平台实例标识或假定的模型绝对路径。

- [ ] **Step 3: 验证六个入口和未开始状态**

Run:

```bash
rg -n "day22/README.md|day23/README.md|day24/README.md|day25/README.md|day26/README.md|day27/README.md" deliverables/week5/README.md
```

Expected: 六个链接各出现一次。

Run:

```bash
test "$(rg -o '未开始' deliverables/week5/README.md | wc -l | tr -d ' ')" -ge 6
```

Expected: exit code `0`。

- [ ] **Step 4: 提交 Week 5 索引**

```bash
git add deliverables/week5/README.md
git diff --cached --check
git commit -m "docs: add week5 engineering archive index"
```

### Task 3: 创建 Day 22 与 Day 23 实施说明

**Files:**
- Create: `deliverables/week5/day22/README.md`
- Create: `deliverables/week5/day23/README.md`
- Reference: `docs/week5_execution_plan.md`

**Interfaces:**
- Consumes: 7B 模型选择、五类图片定义、确定性生成参数和 CSV 字段合同。
- Produces: Day 24–Day 25 使用的冻结模型 revision、环境、图片哈希和 25 条 Base 推理基线。

- [ ] **Step 1: 创建 Day 22 README**

使用 `apply_patch` 创建文件，依次包含：状态、老师要求、当日交付、术语解释、固定输入、详细实施步骤、计划文件结构、完成门禁、故障处理、下一日交接。

详细步骤必须明确：

1. 检查 24GB GPU、磁盘、Python/CUDA/依赖；
2. 锁定 `Qwen/Qwen2-VL-7B-Instruct` revision 后下载；
3. 安装并记录 `qwen-vl-utils` 等版本；
4. 对模型文件生成逐文件清单、总大小和 SHA-256；
5. 用 `local_files_only=True` 做离线加载与一条图文 smoke；
6. 创建五张许可明确图片及 ground truth；
7. 生成 `image_manifest.csv` 并校验五类各一张；
8. 只有模型和图片证据都通过才把 Day 22 标记完成。

状态保持“未开始”，所有尚未知道的 revision、SHA 和指标写成“执行时记录”，不得填写虚假示例值。

- [ ] **Step 2: 创建 Day 23 README**

文件使用与 Day 22 相同的十段结构。详细步骤必须明确：

1. 校验 Day 22 的模型 revision 和五张图片 SHA；
2. 冻结 5 类 Prompt 模板：描述、OCR、结构/图表解释、美学评价、隐含信息推理；
3. 生成 25 个唯一 `record_id`；
4. 固定 `do_sample=false`、`seed=42`、`max_new_tokens`、`min_pixels` 和 `max_pixels`；
5. 保存每条原始回答、token、延迟、峰值显存和配置哈希；
6. 人工记录优势、边界和幻觉，不润色模型回答；
7. 验证 5×5 组合无缺失、无重复；
8. 汇总能力边界，但不把主观评价当成训练结论。

- [ ] **Step 3: 验证数量、字段与状态**

Run:

```bash
rg -n "Qwen/Qwen2-VL-7B-Instruct|image_manifest.csv|SHA-256|离线加载|未开始" deliverables/week5/day22/README.md
```

Expected: 模型、素材清单、哈希、离线 smoke 和状态均存在。

Run:

```bash
rg -n "5 张.*5 类|25 条|do_sample=false|seed=42|raw_response|hallucination_label|未开始" deliverables/week5/day23/README.md
```

Expected: 25 条矩阵、确定性生成、审查字段和状态均存在。

- [ ] **Step 4: 提交 Day 22–Day 23 说明**

```bash
git add deliverables/week5/day22/README.md deliverables/week5/day23/README.md
git diff --cached --check
git commit -m "docs: plan week5 model setup and vlm inference"
```

### Task 4: 创建 Day 24 与 Day 25 实施说明

**Files:**
- Create: `deliverables/week5/day24/README.md`
- Create: `deliverables/week5/day25/README.md`
- Reference: `docs/week5_execution_plan.md`

**Interfaces:**
- Consumes: Day 22 冻结图片/模型和 Day 23 的 Prompt、回答与视觉事实。
- Produces: 至少 3 个注意力案例及固定 10 题 Base 幻觉检测结果。

- [ ] **Step 1: 创建 Day 24 README**

文件使用统一十段结构，并明确专业口径：Qwen2-VL 视觉 token 经视觉编码器和 merger 进入语言模型序列；本项目的“跨模态注意力”是第 20 层或最接近的有效语言模型层中，目标文本 token 对视觉 token 区间的 self-attention。

详细步骤必须包含：

1. 从 processor 输入定位视觉 token 起止索引和 `image_grid_thw`；
2. 切换 `attn_implementation="eager"` 并尝试读取 attention；
3. 不可直接返回时，用 Hook 捕获第 20 层 Q/K，按缩放点积和真实 mask 重算；
4. 对固定目标词做单步或 teacher-forced forward；
5. 保存逐头数组，并对 heads 求均值；
6. 按 merger 后网格还原二维权重，上采样并输出原图/热力图/overlay 三联图；
7. 至少选择 3 张不同图片和 3 个不同目标词；
8. 每个案例保存 layer、token、索引区间、网格、聚合方式和所有哈希；
9. 明确禁止用随机图、Grad-CAM 或纯 ViT self-attention 冒充文本到图像映射。

- [ ] **Step 2: 创建 Day 25 README**

文件使用统一十段结构。详细步骤必须包含：

1. 在运行模型前冻结 10 条 `image-question-fake_answer-ground_truth` 记录；
2. 每题都基于冻结图片构造可核验的不存在物体、文字、数值、关系或动作；
3. 固定模型、revision、图像哈希和生成参数；
4. 保存逐题原始回答；
5. 按二元规则评分：确认/扩展不存在事实为 1，纠正/拒绝错误前提为 0；
6. 计算 `hallucination_count / 10`；
7. 报告所有失败案例，不删题、不改 ground truth、不改分母。

- [ ] **Step 3: 验证技术口径与固定分母**

Run:

```bash
rg -n "self-attention|第 20 层|Hook|Q/K|image_grid_thw|3 张|Grad-CAM|未开始" deliverables/week5/day24/README.md
```

Expected: 架构定义、提取回退、网格映射、最低数量和禁止项均存在。

Run:

```bash
rg -n "10 条|fake_answer|ground_truth|hallucination_count / 10|不删题|未开始" deliverables/week5/day25/README.md
```

Expected: 测试输入、评分依据、公式和防止结果挑选规则均存在。

- [ ] **Step 4: 提交 Day 24–Day 25 说明**

```bash
git add deliverables/week5/day24/README.md deliverables/week5/day25/README.md
git diff --cached --check
git commit -m "docs: plan week5 attention and hallucination tests"
```

### Task 5: 创建 Day 26 与 Day 27 实施说明

**Files:**
- Create: `deliverables/week5/day26/README.md`
- Create: `deliverables/week5/day27/README.md`
- Reference: `docs/week5_execution_plan.md`

**Interfaces:**
- Consumes: Day 22 基座 lineage、200 条训练数据、20 条隔离评测和 Day 23–Day 25 基线证据。
- Produces: 可审计 LoRA adapter/日志/前后对比，以及 Week 5 周报和最终模型归档规范。

- [ ] **Step 1: 创建 Day 26 README**

文件使用统一十段结构。详细步骤必须包含：

1. 建立 200 条正式训练数据和额外 20 条隔离评测；
2. 检查 ID 唯一、图片存在、消息 schema、非空回答、许可和图片 SHA 零重叠；
3. 用锁定版本 LLaMA-Factory 的真实 processor 做 schema smoke；
4. 配置 Qwen2-VL 多模态 SFT LoRA，明确 `freeze_vision_tower: true`；
5. 运行少样本单步 smoke，验证 adapter 非空和视觉塔参数未更新；
6. 正式训练保存有效 YAML、命令、日志、loss、显存、adapter manifest 和 SHA；
7. Base 与 LoRA 在同一 20 条隔离集上以相同生成参数推理；
8. 隐藏模型身份后按五维权重评分；
9. 同时检查 `mean_gain >= 0.50/5`、`lora_wins >= 12/20` 和 LoRA 幻觉率不高于 Base；
10. 未通过时保留失败 run 摘要，不用 train loss 代替质量提升。

必须解释 LoRA、adapter、冻结视觉塔、隔离评测和数据污染。

- [ ] **Step 2: 创建 Day 27 README**

文件使用统一十段结构。详细步骤必须包含：

1. 审计 Day 22–Day 26 证据完整性；
2. 整理模型/environment lineage、25 条推理、3+ 热力图、10 题幻觉率、200 条数据和训练日志；
3. 汇总 Base/LoRA 20 条盲评；
4. 编写《第 5 周：多模态实践报告》；
5. 建立老师四项验收矩阵；
6. 建立最终 VLM adapter/merged 模型 manifest、路径、大小、SHA 和加载 smoke；
7. 创建不含大权重和敏感信息的 `Submission/Week5/` 教师投影；
8. 若任何硬门槛未过，周报如实写为未通过，不得宣称 Week 5 完成。

- [ ] **Step 3: 验证训练与最终归档门槛**

Run:

```bash
rg -n "200 条|20 条|freeze_vision_tower: true|LoRA|0.50/5|12/20|幻觉率|未开始" deliverables/week5/day26/README.md
```

Expected: 数据、配置和三个效果条件均存在。

Run:

```bash
rg -n "25 条|3 张|10 题|200 条|验收矩阵|模型归档|Submission/Week5|未开始" deliverables/week5/day27/README.md
```

Expected: Week 5 全部证据与教师投影边界均存在。

- [ ] **Step 4: 提交 Day 26–Day 27 说明**

```bash
git add deliverables/week5/day26/README.md deliverables/week5/day27/README.md
git diff --cached --check
git commit -m "docs: plan week5 vlm finetuning and report"
```

### Task 6: 执行跨文件一致性与范围审计

**Files:**
- Verify: `docs/week5_execution_plan.md`
- Verify: `deliverables/week5/README.md`
- Verify: `deliverables/week5/day22/README.md`
- Verify: `deliverables/week5/day23/README.md`
- Verify: `deliverables/week5/day24/README.md`
- Verify: `deliverables/week5/day25/README.md`
- Verify: `deliverables/week5/day26/README.md`
- Verify: `deliverables/week5/day27/README.md`

**Interfaces:**
- Consumes: 前五个任务生成的八份规划文件。
- Produces: 可证明需求全覆盖、链接可解析、状态真实且未污染教师目录的 Week 5 规划基线。

- [ ] **Step 1: 验证八份文件全部存在**

Run:

```bash
for f in docs/week5_execution_plan.md deliverables/week5/README.md deliverables/week5/day{22,23,24,25,26,27}/README.md; do test -f "$f" || exit 1; done
```

Expected: exit code `0`。

- [ ] **Step 2: 验证每个 Day README 的统一章节**

Run:

```bash
for f in deliverables/week5/day{22,23,24,25,26,27}/README.md; do for h in 状态 老师要求 专业术语 实施步骤 计划文件 完成门禁 故障处理; do rg -q "$h" "$f" || exit 1; done; done
```

Expected: exit code `0`，六份文件均具备七类核心信息。

- [ ] **Step 3: 验证老师数量要求和预注册门槛**

Run:

```bash
rg -l "Qwen/Qwen2-VL-7B-Instruct" docs/week5_execution_plan.md deliverables/week5/{README.md,day22/README.md}
rg -l "25 条" docs/week5_execution_plan.md deliverables/week5/{README.md,day23/README.md,day27/README.md}
rg -l "3 张" docs/week5_execution_plan.md deliverables/week5/{README.md,day24/README.md,day27/README.md}
rg -l "10 题|10 条" docs/week5_execution_plan.md deliverables/week5/{README.md,day25/README.md,day27/README.md}
rg -l "200 条" docs/week5_execution_plan.md deliverables/week5/{README.md,day26/README.md,day27/README.md}
```

Expected: 每个命令列出传入的全部文件；无某项即修正文档后重跑。

- [ ] **Step 4: 验证状态、链接和教师目录边界**

Run:

```bash
for f in deliverables/week5/README.md deliverables/week5/day{22,23,24,25,26,27}/README.md; do rg -q "未开始" "$f" || exit 1; done
```

Expected: exit code `0`。

Run:

```bash
test ! -e Submission/Week5
```

Expected: exit code `0`。

Run:

```bash
for f in deliverables/week5/day{22,23,24,25,26,27}/README.md; do rg -q "../../docs/week5_execution_plan.md|../../../docs/week5_execution_plan.md" "$f" || exit 1; done
```

Expected: exit code `0`；若采用的相对路径统一为 `../../../docs/week5_execution_plan.md`，六份均应匹配。

- [ ] **Step 5: 扫描占位词、虚假结果和敏感信息**

Run:

```bash
rg -n "TBD|TODO|placeholder|之后补|填入真实|示例哈希|训练成功|下载成功|已完成|password|passwd|ssh-rsa|BEGIN.*PRIVATE KEY" docs/week5_execution_plan.md deliverables/week5
```

Expected: 没有占位词、敏感内容或肯定式虚假实验结果；“已完成”如仅出现在禁止性语句中，人工确认语境后保留。

- [ ] **Step 6: 检查 Git 范围和格式**

Run:

```bash
git diff --check b410088..HEAD
git status --short
```

Expected: 无空白错误；Week 5 规划提交不包含用户更新的 `实习需求.pdf`、`实习需求.docx` 或旧 PDF 删除状态。用户文档变化可以继续保持未暂存。

- [ ] **Step 7: 记录最终审计提交（仅在修正产生新改动时）**

```bash
git add docs/week5_execution_plan.md deliverables/week5
git diff --cached --check
git commit -m "docs: validate week5 planning baseline"
```

若审计未产生任何修正，不创建空提交。

---

## Completion Definition

执行完本计划后，只能声称“Week 5 规划文件已整理完成”，其证据是八份 Markdown 存在、六日状态均为“未开始”、老师所有数量与验收条件均可定位、`Submission/Week5/` 不存在、Git 提交未包含用户需求文档变更。不得声称模型已下载、GPU 已运行、25 条推理已生成、热力图已产出、幻觉率已计算、LoRA 已训练或周报已完成。
