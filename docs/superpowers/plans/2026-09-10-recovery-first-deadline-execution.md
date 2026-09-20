# 模型质量恢复优先与 Week 7 截止前交付 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task in the current session. Steps use checkbox syntax. 保持当前对话内执行；没有用户另行要求，不启动子代理、不另建任务。

**Goal:** 从原始 Qwen2.5-7B-Instruct 重建可审计的 SFT 候选，按质量门禁决定是否继续 DPO 和 Week 7；在英国时间 2026-09-11 08:00 前交付真实成果和明确的未完成项。

**Architecture:** 复用现有恢复分支处理数据、训练及质量评测，复用现有 Week 7 分支处理量化、服务和界面。原始审核表、历史权重和旧证据只读；所有新实验使用独立版本目录。数据就绪、训练成功、质量通过、部署验收是不同状态，不能互相替代。

**Tech Stack:** Python、现有审核发布器、LLaMA-Factory、PyTorch/Transformers/PEFT、Qwen2.5-7B-Instruct、单张 RTX 3090；后续 Week 7 使用隔离的 AutoAWQ、GPTQ、vLLM、Gradio 环境。优先复用仓库锁定版本，不在历史环境原地升级。

**Spec:**

- [质量恢复主规格](/Users/yifanren/Documents/商汤科技实习/.worktrees/pre-week7-model-recovery/docs/week2_week4_model_quality_recovery_plan.md)
- [已完成前两步及剩余数据收口计划](/Users/yifanren/Documents/商汤科技实习/docs/superpowers/plans/2026-09-08-recovery-data-closeout-next-steps.md)
- [历史复评协议与八项门禁](/Users/yifanren/Documents/商汤科技实习/.worktrees/pre-week7-model-recovery/docs/superpowers/specs/2026-09-03-pre-week7-model-recovery-design.md)
- [包含后续质量阻断的 Week 7 规格](/Users/yifanren/Documents/商汤科技实习/.worktrees/pre-week7-model-recovery/docs/superpowers/specs/2026-08-31-week7-quantized-deployment-design.md)
- [可复用的 Week 7 工程计划](/Users/yifanren/Documents/商汤科技实习/.worktrees/week7-quantized-deployment/docs/superpowers/plans/2026-08-31-week7-quantized-deployment-implementation.md)

## Global Constraints

- 唯一正式实例：320，实例 ID `be044ebe99-be706b14`。344 只读归档，不训练、不新建容器。
- 用户已授权 AutoDL 启动和使用开销，不重复询问费用授权；每次不再使用后关机，并保留关机结果。无法关机时立即报告，不谎称已关。
- 不覆盖任何历史权重、原始 XLSX、冻结数据、旧实验输出或用户已有改动；不使用 `git reset --hard`、全目录覆盖或无目标批量清理。
- 用户本人主审、另一人独立复审的参与事实已经确认，不再要求证明真实性。真实参与、具体内容结论、审核覆盖范围和机器字段绑定分别处理，不虚构姓名、日期、签署或最终裁决。
- 267 条未关闭记录保持隔离；不要求全部救回才允许推进。3,885 条只是待发布候选，不保证最终数量不变。
- SFT 从原始基座重新开始，不接续旧 Week 3 merged，不把旧 DPO 权重用于新恢复训练起点。
- 每个正式 SFT 配置至少 3 个 seed；开发集选模，锁定唯一候选后才使用隐藏终测。失败、OOM、中断和非预期重跑全部记录。
- SFT 未验收通过不得进入 DPO；DPO 未通过最终门禁不得正式进入 Week 7。截止时间不自动放宽门禁。
- 用户尚未选择“基座部署演示”或“旧 DPO 工程演示”替代路线，本计划不擅自切换。
- 一个 GPU 重任务串行执行；本地整理可以在 GPU 工作期间交错进行，不在单张 3090 上并行训练、量化和视觉服务。
- 本文件是计划，不是执行回执。编写本文件未启动 AutoDL、训练、量化或新的定时监控。

## Material Passport

- Origin Skill: `writing-plans` + `academic-research-suite / experiment-agent`
- Origin Mode: plan, inline execution handoff
- Origin Date: 2026-09-10 Europe/London
- Verification Status: UNVERIFIED（指尚未执行的实验；下节单元测试和文件状态另有实际检查）
- Version Label: recovery_first_deadline_plan_v1
- Hypothesis: 修复训练内容与多轮监督后，保守 SFT 能在保留基座通用、代码及安全能力的同时改善目标业务；这是假设，不是已经证明的结论。

## 1. 当前状态：已有什么，缺什么

| 项目 | 已核实的状态 | 不能推出的结论 |
|---|---|---|
| 9 月 8 日前两步 | 原 4,908 条处置完整；KEEP 1,720、REVISE 2,165、HOLD 838、DROP 185；候选 3,885，排除 1,023；123 份被隔离修订正文已归档 | 不能说正式训练数据已经发布 |
| 审核人员 | 用户明确确认真实主审和独立第二人参与 | 不等于所有缺失最终裁决、修订正文独立复核字段自动通过 |
| 历史模型 | 旧权重与历史证据仍保留；旧 corrective 没有质量晋级通过 | 不能因模型文件完整就宣称模型合格 |
| 恢复分支 | `fix/pre-week7-model-recovery`，HEAD `c91b363`，有前轮未提交改动 | 不能把工作区当成干净版本覆盖 |
| Week 7 分支 | `feature/week7-quantized-deployment`，HEAD `75ae023`，工作区干净；已有量化、服务、Gradio、验收代码 | Week 7 不需要从零开发，但也没有完成真实验收 |
| 本轮 Week 7 复测 | 54 项单元测试通过；只读严格验收返回 `INCOMPLETE`，11 项远端收据缺失 | 单元测试不证明 GPU 可运行、量化成功或接口可用 |
| 320 当前连接 | 本轮没有重新核实 GPU/SSH；此前浏览器刷新后进入工具禁止操作的登录页 | 不引用缓存中的 GPU 空闲状态，不绕过受限页面 |

重要覆盖规则：Week 7 分支中 8 月 31 日的旧模型路径、旧校准集和 `READY_FOR_REMOTE` 只描述当时工程状态。9 月质量恢复规格中的阻断优先；新模型产生后必须更新血缘和验收绑定。

要求来源也须区分：双量化、基准、服务、界面、图片、录屏和报告来自老师的 Week 7 要求；三 seed、独立隐藏集和恢复晋级门禁来自此前质量恢复方案，不冒充老师新加的作业要求。

## 2. 时间预算与截止策略

截止为 **2026-09-11 08:00 BST / 07:00 UTC**。本轮核对时间为 9 月 10 日约 11:40 BST，剩余约 20 小时；以下是目标窗口，不是保证完成时间。若执行开始已晚于窗口，按真实剩余时间重算，不补造过去的执行记录。

历史依据：同类单张 3090、约 4,999 条数据，SFT 两轮训练约 2 小时 7 分，三轮约 3 小时 11 分。仅按条数缩放到 3,885 条会得到约 1.65 小时，但新正文长度、多轮监督和配置变化使其不能作为承诺；首轮暂按 **1–4 小时纯训练**预算。数据发布与评测准备按 **4–8 小时**、真实输入与环境检查按 **1–3 小时**预算，部分只读工作可交错进行。

| 英国时间目标窗口 | 优先工作 | 放行条件或截止处置 |
|---|---|---|
| 9 月 10 日 12:00–13:00 | 核对工作树、输入清单、审核覆盖、320 正常连接渠道 | 确认确切资产和最小缺口；连接不可用不阻止本地审计 |
| 12:00–16:00，复杂情况延至 20:00 | 全量技术审计、已有审核绑定、正式数据发布、开发/隐藏集协议 | 不把 3,885 条直接改名成训练集；未过门禁不训练 |
| 最早 16:00–18:00 | 320 环境、真实 ShareGPT loader 审计、10–20 step smoke、耗时测量 | 全量监督检查通过；smoke 输出独立于正式训练 |
| 最早 18:00–22:00 | 第一配置第一 seed 正式 SFT | 仅称候选训练完成，保留配置、日志、权重和身份 |
| 22:00–次日 04:00 | 其余 seed 与开发集比较；条件齐备才进行正式终测 | 3 seed、评分或真实复核不足就标未完成；不靠单次好结果晋级 |
| 次日 04:00–06:30 | 回收证据、更新真实状态、验收包、冷启动文档核对 | 04:00 不再开始重型计算；预先限定本轮作业结束预算 |
| 06:30–07:00 | 交付可提交目录、校验清单、完成/缺项表；不再使用即关机 | 预留 1 小时给用户检查和提交 |
| 07:00–08:00 | 仅修复打包、链接或说明错误 | 不伪造缺少的实验指标，不以最后一分钟训练替换未审核模型 |

这里没有把“重新 SFT + 多 seed 验收 + 重建 DPO + Week 7 全部实测”承诺在今晚完成。历史 DPO 的约 1 小时训练或 40-step 的约 10 分钟都不包含新偏好数据、评测和验收。后续 Week 7 的双量化、三模型基准、服务及录屏另需实测预算；首次环境兼容问题可能显著延长时间。

每轮开始前用 smoke 的实际 steps/s 预测：`剩余优化步数 / 实测优化步率 + 保存时间 + 评测预算 + 至少 1 小时缓冲`。预测超过本轮计算截止则不启动该轮。不能靠缩减已冻结 seed 数、跳过评测或更换模型身份挤进截止。

## 3. 工作目录与交接文件

以下别名仅用于本计划，均为明确路径；尚未产生的文件均标为“新建”。新输出目录要求独占创建，若已存在则停止并使用经检查的下一 attempt，不覆盖。

| 别名 | 路径 |
|---|---|
| `MAIN` | `/Users/yifanren/Documents/商汤科技实习` |
| `RECOVERY` | `/Users/yifanren/Documents/商汤科技实习/.worktrees/pre-week7-model-recovery` |
| `W7` | `/Users/yifanren/Documents/商汤科技实习/.worktrees/week7-quantized-deployment` |
| `SOURCE` | `RECOVERY/deliverables/model_recovery/source` |
| `OLD` | `MAIN/outputs/01a057f0-219e-7f43-9161-67e37e5d00d4/recovery_closeout_20260908`，只读 |
| `RUN`（新建） | `MAIN/outputs/01a057f0-219e-7f43-9161-67e37e5d00d4/recovery_run_20260910` |
| `REMOTE`（拟新建） | `/root/autodl-tmp/pre-week7-recovery/run-20260910`，仅 320 |

共享接口：每份新回执记录 `schema_version`、`run_id`、`created_at`（带时区）、`status`、输入文件字节 SHA-256、代码/环境身份、输出清单和失败原因。文件字节哈希与规范化 JSON 哈希分字段保存，不混为一个摘要。后继阶段按清单实际重算，不只信任上游的 `status=PASS`。

状态顺序：`PROVISIONAL_NOT_RELEASED → REVIEWED_DATA_ONLY_NOT_TRAINING_READY → INPUT_AUDIT_PASS → SMOKE_PASS → CANDIDATE_TRAINED → DEVELOPMENT_EVALUATED → SFT_ACCEPTED → DPO_ACCEPTED → WEEK7_ACCEPTED`。其中任一阶段可以是 `BLOCKED`、`FAIL` 或 `INCONCLUSIVE`；不得跳级。

## 4. Task 1：固定本轮入口，拆开历史收口与训练门禁

**Files:** 只读 `OLD` 全部回执、三份定点修正 XLSX、原始主/复审表及侧车清单；新建 `RUN/inventory.json`、`RUN/review_coverage.json`、`RUN/minimal_review_gaps.csv`、`RUN/historical_closeout.json`。

**Interfaces:** 输入冻结哈希和既有用户确认；输出逐 ID 的已覆盖/缺失清单，不改变原表。

- [ ] 记录主仓库和两个工作树的 HEAD、完整 dirty 状态；标出用户修改的老师 DOCX/PDF与前轮恢复代码。不得 `git add .`。
- [ ] 重算 `OLD/completion_files.sha256.json` 所列 12 文件及固定输入摘要，核对 4,908 条唯一 ID、3,885/1,023 分流和 123 条正文归档。
- [ ] 读取已有审核字段与声明，分别核对主审处置、全 assistant 轮覆盖、2,165 条修订正文的独立检查、81 条最终裁决。已有真实证据直接复用，不重新让人审核整表。
- [ ] 只输出真实缺失项：具体 sample ID、原内容摘要、需要判断的字段、已有依据和所需最小确认。第二人完成盲评不自动推导其已检查全部修订正文；也不把程序格式缺失误报成人员真实性问题。
- [ ] 历史缺证据的 `TRAIN-0024/0038/0055/0078/0107` 保留事实状态未知，事实有效样本统计明确为 195/200；其他维度按其实际有效分母计算，不把未知填成 0 或 1，不机械映射为同编号 SFT 样本。
- [ ] 81 条分歧中，建议与最终裁决分开；未确认则保留未决、展示两份原评分和分歧范围。不能自动将 78 条普通建议及 3 条安全建议写成最终人工意见。
- [ ] 审计依赖：历史 5 条与旧模型 81 条主要用于历史结论收口，不应无理由阻塞一个独立、合格的新 SFT 数据发布；真正阻塞训练的是拟保留样本缺审核依据、输入监督或新评测隔离不通过。

**验收：** 原表哈希不变；所有缺口有明确范围；没有虚构真实签署日期。若现有发布器需要的声明文字/签署范围尚无真实认可，导出待确认声明，不替用户签名。

## 5. Task 2：全量候选技术审计与正式发布

**Files:** 复用 `SOURCE/model_recovery/review_release.py`、`SOURCE/scripts/build_reviewed_sft_dataset.py` 和 9 月 8 日计划 Task 4–5；新建 `RUN/corpus_audit.json`、`RUN/corpus_findings.jsonl`、`RUN/release_inputs/` 和 `RUN/released/`。

**Interfaces:** 审核输入为全 4,908 条处置；正文只来自对应 KEEP 原文或已审核 REVISE 正文。输出为正式已审核 ShareGPT 和逐条 lineage，仍不等于训练许可。

- [ ] 逐条检查角色交替、最后一轮 assistant、空值、完整问答、全轮正文、控制字符、凭据、代码围栏、占位符、重复及跨样本泄漏；同时记录语言、来源、轮数、代码比例、拒答率、回答长度分布。
- [ ] 使用 `validate_conversations`、`scan_all_turns` 等既有接口；扫描信号是定位工具，不是事实真假结论。HTML/代码等误报必须按原文解释，不能全局关闭检查。
- [ ] 对照原 4,908 与拟保留 3,885 的分布，列出因隔离产生的业务、长回答、多轮或代码样本变化。来源不明、证据不足的事实不由 AI 补造。
- [ ] 发现新问题时生成新的处置版本，保留先前记录和原因；不得静默修改已签署正文。缺独立修订核验时先核对已有材料；确实缺失才补核或保持隔离，不擅自改成只训练 1,720 条 KEEP 的另一套实验。
- [ ] 从真实材料构造 `decisions_4908.jsonl`、`attestations.json`、`revision_attestations.json`。绑定原文、修订正文、实际人员和范围；不得批量填 `revision_verified=true` 或 `reviewed_all_assistant_turns=true`。
- [ ] 运行现有发布器，失败保留错误回执；通过后独立回读输出、处置计数、lineage 和字节摘要。

执行时在 `RECOVERY` 下使用以下现有接口；`release_inputs` 必须先真实完成，命令本身不会补足人员或事实依据：

```bash
RECOVERY_RUN='/Users/yifanren/Documents/商汤科技实习/outputs/01a057f0-219e-7f43-9161-67e37e5d00d4/recovery_run_20260910'
/Users/yifanren/anaconda3/bin/python deliverables/model_recovery/source/scripts/build_reviewed_sft_dataset.py \
  --decisions "$RECOVERY_RUN/release_inputs/decisions_4908.jsonl" \
  --attestations "$RECOVERY_RUN/release_inputs/attestations.json" \
  --revision-attestations "$RECOVERY_RUN/release_inputs/revision_attestations.json" \
  --output-dir "$RECOVERY_RUN/released"
```

**验收：** 发布器输出 `REVIEWED_DATA_ONLY_NOT_TRAINING_READY`；其最低 1,500 条限制只是程序下限，不是质量或分布合格证明。

## 6. Task 3：冻结新评测、训练协议与选择规则

**Files:** 新建 `RUN/protocol/evaluation_protocol_v2.json`、`RUN/protocol/split_manifest.json`、`RUN/protocol/overlap_audit.json`、`RUN/protocol/model_registry_v2.json`、`RUN/protocol/sft_config_a.yaml`；新终测材料存放受控位置，只把摘要放入训练合同。

**Interfaces:** 已审核语料、全历史训练/开发/公开题清单 → 来源隔离的开发集、SFT 隐藏终测、DPO 独立隐藏终测以及训练合同。

- [ ] 新题不得直接复用或轻微改写旧公开 200 题；按来源、问题、答案和近重复簇与全部旧训练、新训练、偏好数据排查。确切重复必须排除，近重复命中逐项处理，不能以 SHA 不同证明无泄漏。
- [ ] 设计目标为每套 200 题，沿用类别构成：通用 50、数学 50、业务 40、安全 40（有害/良性各 20）、代码 20；开发、SFT 终测、DPO 终测相互独立。先审阅题目、参考答案和代码测试本身，再冻结；样本不足不得拿训练数据凑数。当前先完成 SFT 所需开发/隐藏集；DPO 独立终测在 DPO 开始前完成即可，不把尚未开始的 DPO 题集编制额外设为 SFT 启动条件。
- [ ] 用任务/来源簇作为必要的重采样单元，检查有效样本数；200 题只是协议起点，不保证各类别置信区间足够窄。若预先功效/精度检查表明不足，要增加独立样本后再训练，或明确本轮只能做探索性候选，不能降低门禁。
- [ ] 固定评分 rubric、指标归一化和类别权重，继承既有实现并记录版本；修正 CODE-03 类测试缺陷必须另存新协议，不能改写旧成绩。
- [ ] 本轮保守提案：单配置 A，QLoRA 4-bit NF4、double quant、rank 8、alpha 16、dropout 0.05、学习率 `2e-5`、最多 2 epoch、batch 1、梯度累积 4，seed `42/43/44`。其他优化器和 scheduler 字段逐项从历史生效配置核对并冻结，不盲抄旧矩阵。cutoff 先检查真实长度；若 2,048 会截断目标监督，则先调整配置与显存预算，不静默截断。
- [ ] 明确 `train_on_prompt=false`、`mask_history=false`、`packing=false`，模板为核实后的 Qwen 模板；正式训练三个 seed 的数据、超参数和评测合同相同。先只做配置 A，未完成三个 seed 不扩大搜索。
- [ ] 预登记：开发集比较各 seed 的完整结果及均值/离散度，按既定总体分选唯一候选，平分按业务分再按 seed 升序；任何硬安全失败不能靠高总分掩盖。一次锁定后生成 `candidate_lock.json`，才允许打开对应隐藏终测。
- [ ] 预登记正式非劣效规则，不沿用“差异不显著就是合格”：默认严格容差为 0，配对 95% 区间下界满足预设底线才可支持非劣效；不足则 `INCONCLUSIVE`。保留总体、业务、代码不低于基座，安全有害可操作回答为 0、拒答率至少 95%、良性过拒答至多 5%，全量推理有效等门禁；DPO 同时对照已接受 SFT。

**验收：** 训练前冻结协议、数据与选模规则；隐藏集不进入训练目录或调参上下文。隐藏集一旦用于失败分析即退役，不给下一候选重复作为未见终测。

## 7. Task 4：补足真实多轮监督检查，再做 320 smoke

**Files:** 复用只读 `SOURCE/scripts/audit_remote_training_inputs.py` 中真实 LLaMA-Factory loader 的装载方式；新建 `SOURCE/model_recovery/sharegpt_supervision.py`、`SOURCE/scripts/audit_recovery_sharegpt_inputs.py`、`SOURCE/tests/test_sharegpt_supervision.py`。输出 `RUN/input_audit/`、`RUN/environment_receipt.json`、`RUN/smoke_receipt.json`。

**Interfaces（拟新增，不是已经存在的命令）:** `assert_supervision(expected_ids, expected_labels, actual_ids, actual_labels, attention_mask)` 检查独立渲染结果与 loader 逐 token 一致；新 CLI 参数为 `--config`、`--dataset`、`--lineage`、`--output-dir`，单独审计新 ShareGPT，不假装支持旧入口未实现的参数。

- [ ] 先写并运行回归测试，证明有效多轮中“assistant 后接被 mask 的 user”不能被误判。旧 `audit_sft_tokenized` 的连续监督假设不适合直接给新多轮判 PASS。

```python
from model_recovery.sharegpt_supervision import assert_supervision

def test_masked_user_between_two_assistants_is_valid():
    ids = [11, 12, 13, 14, 15, 16]
    labels = [-100, 12, -100, 14, 15, 16]
    assert_supervision(ids, labels, ids, labels, [1] * 6)

def test_user_token_cannot_receive_loss():
    import pytest
    with pytest.raises(ValueError):
        assert_supervision([11, 12], [-100, 12], [11, 12], [11, 12], [1, 1])
```

- [ ] 实现最小逐 token 比较核，期望值必须来自原文角色边界的独立渲染，不从 actual labels 自证：

```python
def assert_supervision(expected_ids, expected_labels, actual_ids, actual_labels, attention_mask):
    sizes = {len(x) for x in (expected_ids, expected_labels, actual_ids, actual_labels, attention_mask)}
    if len(sizes) != 1 or not expected_ids:
        raise ValueError("empty or unequal token arrays")
    if list(expected_ids) != list(actual_ids) or list(expected_labels) != list(actual_labels):
        raise ValueError("source-to-loader token or supervision mismatch")
    if not any(label != -100 for label in actual_labels):
        raise ValueError("no supervised assistant tokens")
    for token, label, mask in zip(actual_ids, actual_labels, attention_mask):
        if mask not in (0, 1) or (mask == 0 and label != -100):
            raise ValueError("invalid attention or supervised padding")
        if label != -100 and label != token:
            raise ValueError("label differs from unshifted input token")
```

- [ ] 补充缺中间 assistant、EOS 丢失、padding 受监督、全 mask、角色反转、重复/缺失 ID、输入摘要变化的失败测试；独立渲染与真实 loader 双方都保留 token/轮次证据。逐行比较只是核心，还必须证明与原文语义和角色映射一致。
- [ ] 通过合法现有连接核对确为 320；浏览器受限登录页不继续操作、不绕过。记录 `nvidia-smi`、活动进程、磁盘、完整基座与 tokenizer 哈希、Python/CUDA/依赖、生效配置。无连接则报告访问缺口，不猜 host/port，不索取私钥明文。
- [ ] 在 `REMOTE` 新目录装载正式发布数据，核验 loader 行数、ID 映射、全部 assistant 轮监督、human/system/padding 屏蔽、EOS、截断、空监督及数据加载警告。任何丢行、角色错误、身份漂移或未说明截断都阻止训练。
- [ ] 用正式配置做 10–20 个 optimizer step 的 smoke，保存有限 loss、梯度、显存、耗时、可加载 adapter 和短生成。它只验证运行链路，不能证明质量。
- [ ] 正式训练从基座与预登记 seed 重新初始化；不把 smoke 已更新的 adapter 暗中作为正式起点。smoke 目录与正式 attempt 分离。

测试命令在 `RECOVERY` 下运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/yifanren/anaconda3/bin/python -m pytest -q -p no:cacheprovider deliverables/model_recovery/source/tests/test_sharegpt_supervision.py
```

**验收：** 新检查器红/绿测试成立；正式数据全量 source-to-loader 一致；smoke 有真实日志与模型加载回执；不复用旧 SFT 审计的 PASS。

## 8. Task 5：正式 SFT、过程监测与模型身份

**Files:** 新建 `RUN/run_contract.json`、`RUN/training_runs.jsonl`、`REMOTE/sft/config-a/seed-42/attempt-001/`、对应 seed-43/44 目录；后续新增 attempt 只增不改。

**Interfaces:** 训练合同包含发布数据、overlap、真实输入、环境、配置摘要和各门禁结果；输出每轮 adapter、完整配置、训练日志、运行结果、资源曲线和模型清单。

- [ ] 启动前重新核对合同所列文件摘要及门禁，不只读 `PASS` 字样；基座固定为 `/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct` 的已核验资产。
- [ ] 在已核对的远端环境执行 `llamafactory-cli train`，唯一参数为本轮冻结 YAML 的确切路径；先记录解析出的 CLI 路径和版本，拒绝引用历史实验矩阵或旧输出目录。
- [ ] 先 seed 42。以 optimizer step、最近日志时间、进程、GPU、磁盘和有效 token 速率监测，约每 10 分钟检查一次；状态无变化不反复打扰用户。loss 只做健康诊断。
- [ ] OOM、NaN、监督异常、磁盘不足、身份漂移或异常进程退出立即停止当前作业并保存证据。只操作本轮明确 PID，不全局杀进程。可安全重试的基础设施错误使用新 attempt；参数改动意味着新配置而非续写旧成功结果。
- [ ] 训练结束核对步数、epoch、退出码、保存文件完整性和可加载性；恢复断点必须包含相同配置及 optimizer/scheduler/RNG 状态，只加载 adapter 权重不能宣称等价续训。
- [ ] 记录吞吐，重新预测剩余两 seed 与开发集耗时；有预算才串行继续。正式候选配置不因第一 seed 看起来良好而省略其余 seed。
- [ ] 对待评模型做 base / adapter / merged 行为与权重身份核验，统一 tokenizer、padding、精度和生成参数；数值差异有证据再解释，不能把历史 18 个探针的结果当作新模型合并正确证明。
- [ ] 本地回收完整清单和日志，核对摘要后才安排不再使用的实例关机；模型权重仍保存在 320，不擅自清理旧资产。若需要跨本轮持续自动监测，执行阶段使用产品定时机制并核对既有任务，避免重复启动；本计划未创建监控。

**验收：** 单 seed 只能获得 `CANDIDATE_TRAINED`；三 seed 和开发评测完成后才进入下一门禁。

## 9. Task 6：新模型评测、真实盲评与晋级判断

**Files:** 复用 `SOURCE/model_recovery/` 的推理、评分、受限代码执行组件；版本化扩展评测入口以读取 `model_registry_v2.json`，不直接套用只接受旧模型别名的 `build_promotion_decision.py`。新建 `RUN/evaluation/`、`RUN/candidate_lock.json`、`RUN/sft_promotion_decision.json`。

**Interfaces:** 每条结果必须绑定题目、模型、生成合同、原始输出和状态；盲评映射与评分表分离；最终决策消费已锁定候选、客观评分和真实人工结论。

- [ ] 新入口先增加失败测试：新模型冒用旧 alias、模型哈希不匹配、非锁定候选读取终测、缺少一个 seed、缺失评分或把 AI 字段当人工意见，都必须拒绝。旧历史入口与历史结果保持原样。
- [ ] 同环境先跑 base 与三个 SFT seed 的开发集；记录各类别成绩、输出 token 长度、失败数、均值和离散度。不能只报告最好 seed 或只用训练 loss 选模。
- [ ] 检查代码题测试本身能在受限环境执行；模型生成代码仅在已配置的受限运行器执行，未提供沙箱则阻断代码验收，不能直接在主机执行。
- [ ] 按冻结规则锁定唯一候选；只有协议、题集质量和所需审核条件齐全才进行一次 SFT 隐藏终测。自动评分、主审与独立复审分别留存，安全项必须真实审阅。
- [ ] 导出新输出所需的最小盲评包供真实人员完成。既有 400 条审核属于旧模型，不能搬到新模型上；AI 可以预筛和说明，但不能冒充任一真人评分。
- [ ] 按预登记标准计算配对区间、总体及类别门禁；结论为 `PASS`、`FAIL`、`INCONCLUSIVE` 或 `AWAITING_HUMAN_REVIEW`。安全硬失败不可用总分抵消，统计不确定不可写成“不退化”。

**验收：** 只有真实证据支持的 `SFT_ACCEPTED` 才能继续。未通过则交付失败或不确定结论、具体反例和下一轮建议，不把新训练模型自动作为 Week 7 输入。

## 10. Task 7：仅在 SFT 通过后重建 DPO

本任务沿用质量恢复主规格阶段 3，是有条件的后续工作，不预设能够在本次截止前完成。

**Files:** 新建 `RUN/dpo/data_manifest.json`、`RUN/dpo/pair_audit.json`、`RUN/dpo/configs/`、`RUN/dpo/runs/` 和最终 `RUN/promotion_decision.json`。

- [ ] 以已接受 SFT 为起点，检查新 preference 训练/开发/终测互斥及来源；SFT 的终测也不能进入 DPO 训练数据。
- [ ] 逐对核验 chosen/rejected 方向、真正质量差异、长度/拒答偏置、空值、模板、事实与安全边界。旧 18 条疑似方向问题不一律翻转；双方同等的对删除或基于真实差异重建。
- [ ] 用真实 DPO loader 检查共享 prompt、角色、截断与 chosen/rejected 是否在处理后变成相同序列；先 smoke，再小步单因素、多 seed 实验，保存失败运行。
- [ ] 开发集选唯一 DPO 候选，使用独立 DPO 隐藏集，按相同推理合同比较 base / 已接受 SFT / DPO；完成八项门禁、配对区间、代码真执行和真实盲评。
- [ ] 只从实际完整证据生成 `promotion_decision.json`；必须绑定新模型摘要和协议，不继承旧 corrective 的路径或 PASS 字段。

**验收：** `DPO_ACCEPTED` 且最终晋级 `PASS`。没有该证据，Week 7 工程可以整理，正式量化部署继续阻断。

## 11. Task 8：复用 Week 7 工程，逐项补真实验收

**Files:** 在 `W7` 工作树修改 `docs/week7_execution_plan.md`、`deliverables/week7/README.md`、Day 34–39 配置/说明与验收代码；已有测试在 `deliverables/week7/tests/`。真实实验按新版本保存，旧收据只读。

**可先完成、不依赖训练的部分：** 回归测试、老师要求逐项映射、旧模型硬编码检查、FP16/BF16 口径修正设计、验收器摘要绑定、部署说明占位状态检查。不得将这些写成量化已完成。

- [ ] 先增加回归测试：摘要不匹配的收据即使写着 `PASS` 也不能通过；旧模型收据不能用于新模型；BF16 结果不能标为 FP16。当前顶层验收器主要检查文件与状态，必须结合明细验证器及输入/输出绑定，不能只凑齐 11 个 JSON。
- [ ] DPO 通过后把其模型、tokenizer、合并过程、数据和决策摘要绑定到全部 Week 7 配置。重建合格的 128 条校准、256 条 PPL 及生成诊断数据，检查相互及训练/终测的适当隔离，不盲用旧低质量候选来源。校准可来自合格训练数据，PPL/终测不可与训练污染混用。
- [ ] **Day 34：** 重跑版本/源码/运行时兼容检查，实际 AWQ 导出并加载 smoke。固定版本如需 AutoAWQ 替代老师提示的 LLaMA-Factory AWQ 导出，保留兼容性与替代实现收据，不冒充同一命令。
- [ ] **Day 35：** GPTQ 实际导出、加载；按老师要求补 FP16/AWQ/GPTQ 同口径测量。现有 BF16 可保留为附加实验，不能改文件名冒充 FP16；配置、基准实现、指标公式、表头及验收路径一起更新。记录权重体积、加载/峰值显存、tokens/s、NLL/PPL，至少一种量化显存下降 30%，同时报告质量变化和失败。
- [ ] **Day 36：** 从通过验收的量化模型启动 vLLM，验证模型名、同步和流式 `/v1/chat/completions`；保留完整日志、真实请求与响应和计时。
- [ ] **Day 37：** 浏览器端到端检查 Gradio 历史、temperature/top_p 参数透传、增量输出、清空和后端错误提示；单元测试不能替代真实页面验证。
- [ ] **Day 38：** 停止文本后端，再启动 Week 5 已验证 Qwen2-VL 基座；使用已核验图片验证路由，录屏覆盖文字流式、模型切换和图片问答。明确 Week 5 LoRA 尚未质量通过，不描述成已通过的微调 VLM。
- [ ] **Day 39：** 冷启动复现部署指南，验证环境变量和端口、启动/停止顺序，清除提交包中的敏感值；周报逐项引用证据、偏差和未完成项。

现有本地回归命令，在 `W7` 下：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/yifanren/anaconda3/bin/python -m pytest -q -p no:cacheprovider deliverables/week7/tests
```

最终使用现有 `day39/source/scripts/validate_week7.py --require-remote` 入口，并明确传入新的 `--output`、`--acceptance-csv` 路径，避免默认覆盖旧结果；执行前完成上述身份绑定和 FP16 路径更新。严格验收失败或缺项则保留真实状态。

**验收：** AWQ、GPTQ、三模型基准、vLLM、真实 UI、图片、录屏、指南和报告全部有证据；少一项不能声称老师 Week 7 全量要求完成。

## 12. Task 9：截止前交付与文档同步

**Files:** 新建 `RUN/FINAL_STATUS.md`、`RUN/artifact_manifest.json`、`RUN/teacher_requirements_matrix.csv`、`RUN/unresolved_items.csv`、`RUN/shutdown_receipt.json`；按真实成果更新相关工作树 README、恢复报告、恢复计划最新检查点、Week 7 日报/周报/部署指南。只有真正具备提交内容时才生成新的 `Submission/Week7` 教师投影。

- [ ] 每个阶段结束即时保存短回执；完整周报等在实验结果稳定后补写，不等到最后才保存原始日志。
- [ ] 区分“旧历史归档”“本轮新成果”“未完成/失败”；不删除历史弱模型结果，也不批量改写过去几周的结论。
- [ ] 提交包逐文件重算哈希、检查空文件和断链、脱敏，读取报告确认每个数字有来源，所有状态与实际门禁一致。
- [ ] 如果仅完成数据发布或首个 SFT，交付这些真实成果和剩余门禁，明确不能作为 Week 7 完成证明。未经用户另行选择，不换成旧 DPO 或基座来填满验收表。
- [ ] 审查本轮新增 diff，仅暂存本轮明确文件；每个独立任务测试通过后才做小提交，禁止夹带老师 DOCX/PDF 或前轮未审变化；不自动合并两个工作树或推送远端。
- [ ] 在英国时间 07:00 前给用户可检查的包和简明结论；用户自行向老师提交，除非另有明确提交授权。
- [ ] 结束 GPU 使用后核对本轮进程退出、证据回收和 320 关机。不能关闭不属于本轮且仍在使用的作业；有冲突立即说明。

## 13. 人工依赖与实际交接

可以由代理直接完成：资料核对、技术审计、字段/哈希整理、代码与测试、合格输入的训练运行、自动评测、盲评包导出、统计、量化部署、文档和打包。

只在确实缺失时需要人：正常可用的实例访问渠道；审核声明的具体绑定范围；真正未决的内容/分歧裁决；新模型输出所需的真实盲评。首先利用已有确认和材料，只交付最小缺项清单，不重复要求全部审核或证明人员身份，不代签或冒充。

执行先做 Task 1–3 的可完成部分，同时核对合法访问渠道；具备门禁后再使用 GPU。若有必要的人工作业暂时缺失，继续不依赖它的本地任务；把依赖精确标出来，不谎称全自动可越过。

本计划的明确承诺边界是：尽量把时间用于可复现的恢复成果，不以期限压力宣布未经证据支持的质量通过。是否能在截止前完整满足老师 Week 7 要求，取决于实际恢复与验收结果；目前不能保证。
