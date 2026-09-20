# Week8 正式训练放行 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 在当前任务内顺序执行，每一阶段审核后进入下一阶段。

**Goal:** 将已验收的三步链路推进为可追溯的正式 SFT 5轮 → 合并 → DPO 40步 → 合并，并为新模型建立独立身份。

**Architecture:** 本轮完成候选配置、输入清单、预算建议和验收设计。后续用独立的正式训练入口消费哈希绑定的执行许可，保留旧入口与smoke入口原状。训练与评测分开放行，每个训练阶段最多一次启动，失败保留证据。

**Tech Stack:** Python、LLaMA-Factory 0.9.3、Transformers 4.50.0、PyTorch 2.5.1+cu121、4bit QLoRA、AutoDL RTX3090 24GB。

**Spec:** `docs/superpowers/specs/2026-09-14-week8-automation-design.md`、`docs/week8_data_protocol.md`、`docs/week8_training_smoke_result.md`；老师原文 `reports/week8/requirements_snapshot.txt:275`。

## Global Constraints

- 老师Day40.2要求按预设最优超参数依次SFT、DPO；Day40.3要求错误重试机制和模型合并。老师没有指定“三步”或本轮预算。
- SFT采用Week3 epoch-e5；DPO采用Week4 reward_corrective_40step。样本源改为已冻结的Week8训练1422/验证158；DPO训练783/验证87。
- cutoff_len=2048，4bit NF4，rank8/alpha16，dropout=0.05，seed42，bf16；SFT batch1/累积4，DPO batch1/累积8。
- 只使用320机 `be044ebe99-be706b14`，仅通过内嵌浏览器和已登录Jupyter操作。不开新实例，不删除已有模型，不使用三步adapter续训，不向老师发送内容。
- 选择最终完成的checkpoint，不用custom20、CEval/CMMLU挑选模型。DPO基座和隐式参考均为本次完整SFT合并模型。
- 原始评估锁当前只绑定original_base。生成新SFT/DPO后必须建立新身份并重新绑定评估锁，不能套用基座锁。
- 本轮是正式训练准备；候选配置不是开机或训练放行凭据。`training_allowed=false`继续保留。

## 文件职责与当前状态

| 文件 | 职责 | 状态 |
| --- | --- | --- |
| `configs/week8_full_training_candidate/sft.yaml` | 完整SFT候选 | 已生成与审核 |
| `configs/week8_full_training_candidate/dpo.yaml` | 完整DPO候选 | 已生成与审核 |
| `configs/week8_full_training_candidate/export_sft.yaml`、`export_dpo.yaml` | 两次CPU合并候选 | 已生成与审核 |
| `reports/week8/phase2_full_training_plan/prepare_candidates.py` | 只做本地生成和静态核验，无训练/网络操作 | 已运行通过 |
| 同目录 `input_manifest.json`、`config_overrides.json`、`verification.json` | 输入哈希、逐字段差异、检查结果 | 已生成 |
| `scripts/week8_full_training.py`、`tests/test_week8_full_training.py` | 后续正式训练入口及失败门禁测试 | 已实现，本地验证见正式入口报告 |

### Task 1：冻结候选与资源规划（已完成）

**Consumes:** 老师需求源文件哈希、smoke锁和本地复核、两份历史训练YAML。
**Produces:** 四份候选YAML及只用于准备阶段的输入清单。

- [x] 再运行 `reports/week8/phase2_training_smoke/review_target_evidence.py`，确保回收30文件、31冻结输入仍一致。
- [x] 复核当前docx/pdf与需求快照来源哈希一致；定位Day40.2–40.3。
- [x] 从历史YAML生成候选，保留原学习率和调度器；SFT删除max_samples、显式max_steps=-1，避免遗留三步上限。
- [x] 使用Transformers 4.50.0实际 `Trainer.set_initial_training_values` 和CPU DataLoader核算：`(1422 // 4) * 5 = 1775`。这只是当前版本的预计步数；不擅自改为1780或以此声称每条样本恰好参与5次。
- [x] 审核路径：SFT从原基座开始，DPO只指向新正式SFT；两次合并不覆盖smoke或历史模型。

复核命令：

```bash
/tmp/week8-oc-fix-20260915/bin/python reports/week8/phase2_training_smoke/review_target_evidence.py
/tmp/week8-oc-fix-20260915/bin/python reports/week8/phase2_full_training_plan/prepare_candidates.py
```

配置调整均在 `config_overrides.json` 明列：沿用smoke验证过的预处理1线程/加载0 worker、关闭外部日志服务和训练时绘图，但保留原始日志；SFT增加每轮验证与eval batch1以控制显存；两阶段均不加载best模型。

**费用与效果原则（按用户最新要求更新）：** 完整执行既定SFT 5轮与DPO 40步，不因图便宜减少数据、轮数或关键参数。原¥4.44/3小时仅是上一轮的初步建议，不再作为训练效果的取舍依据。部署前根据实际吞吐量和合并耗时留出充分时间，设置与有效执行窗口一致的平台关机保护；当前候选release的10800秒仍只是待部署时复核的保护值。若预计不足，应在开训前调整执行窗口并重建release，不应明知不足仍启动、被截断后从头重跑。开机即执行，失败立即止损、保存证据，空闲及时关机；不扩大到无目的参数搜索或新增实例。

**资源门槛：** 开机后实时核对无其他任务、60GiB内存限额、至少45GiB空闲盘。本次两个合并模型仅权重约28.37GiB，另留adapter/checkpoint、缓存和临时空间。最后观测数据盘已用80.15%，不以此旧值替代实时检查；不足则停止，不自动清理历史资产。

### Task 2：实现独立正式入口及本地测试

**Files:** 新建 `scripts/week8_full_training.py`、`tests/test_week8_full_training.py`；复用已冻结的输入校验与合并验收；独立调用CLI，旧main不修改。
**Consumes:** 候选配置、31份冻结证据、smoke复核、独立执行许可。
**Produces:** `--dry-run`审计报告或真实训练阶段状态；不自动评测。

- [x] 新建验收函数 `validate_formal_state(state, expected_steps)`，返回训练日志行；要求global_step等于expected_steps，训练步序列完整1…N，loss/梯度/学习率均有限，梯度和学习率不能全零。复用smoke张量检查思想，但不得调用固定3步的验收器。
- [x] 先写失败测试，覆盖步数不足、重复步、NaN、全零更新、错误模型路径、输入被修改、已有输出、过期/缺失执行许可、未通过SFT却启动DPO。核心状态测试：

```python
def test_incomplete_formal_state_rejected(self):
    with self.assertRaises(ValueError):
        validate_formal_state({'global_step': 3, 'log_history': []}, 1775)

def test_dpo_zero_or_missing_updates_rejected(self):
    state = {'global_step': 40, 'log_history': [
        {'step': i, 'loss': 0.6931, 'grad_norm': 0.0, 'learning_rate': 1e-7}
        for i in range(1, 41)]}
    with self.assertRaises(ValueError):
        validate_formal_state(state, 40)
```

- [x] 实现 `--release`、`--release-sha256`、`--run-dir`、`--dry-run` 参数。release包含实例ID、候选及数据哈希、资源策略/时间保护、允许阶段、原基座身份；必须先验证输入再创建输出。dry-run不导入CUDA执行、不启动CLI、不调用API。真实执行要求许可execution_enabled=true且证据无变化。
- [x] 每阶段启动前写RUNNING，训练退出后写退出码，核对实际总步数、优化器步数、adapter数值及非零更新；只有通过才允许导出。每次CLI只尝试一次，因为当前batch已是1，OOM无法再安全降低batch。不得降低长度、学习率或轮数。
- [x] 将执行工作目录固定为独立runtime，日志置于 `logs/full-run-01`。为每个子进程设置剩余时限，失败/超时写阶段状态并终止本任务进程组。不得终止其他进程。
- [x] 本地用模拟CLI验证阶段阻断、超时、已有目录、错误哈希和dry-run无副作用；跑 `python -m unittest discover -s tests -p 'test_week8*.py'`，并重新检查候选/源文件哈希。

### Task 3：补齐两次合并的独立加载验收（代码已实现，实际GPU验收待执行）

**Files:** 上述正式入口与测试；扩展现有加载检查逻辑，但不修改smoke已冻结文件。
**Consumes:** 当前阶段通过的adapter、CPU合并目录。
**Produces:** 各模型文件身份清单、加载/前向回执。

- [x] CPU逐分片检查权重键与索引完全一致、所有数值有限、文件哈希齐全，沿用smoke合并检查；另保存config、tokenizer及全部模型文件身份。
- [x] 在新的独立子进程加载合并模型，不提供adapter路径；使用固定非评测提示“你好”，执行BF16前向及2-token贪心生成，核对logits有限并保存设备、输出、最大显存和退出码。此项只验可加载，不计正式分数。
- [x] SFT独立加载完成退出后再启动DPO。DPO完整40步后执行第二次合并及同样加载核验。任一失败即停止，不生成“正式成功”标记。
- [x] 测试缺失分片、索引重复、NaN、子进程非0以及模型目录指向smoke的拒绝路径。

### Task 4：正式会话资源审核与受控执行

**Consumes:** 已通过本地测试的新入口、冻结部署包、当前会话执行许可和资源保护。
**Produces:** 真实SFT/DPO训练日志、验证loss、模型及完整回执。

- [ ] 依据最新质量优先原则落实执行窗口后，只检查320机；GPU不足保持关机。如果需要恢复监控，更新原监控为正式阶段任务，不能复用已暂停的“三步”提示直接运行。
- [ ] 启动前核对实际价格，并设置覆盖计划执行窗口的平台自动关机。无法设置则停止；进入目标机再次核对身份、进程、磁盘、依赖、CUDA BF16、原基座15文件和输入包。
- [ ] 部署到候选中指定的独立runtime；如路径存在则停止并核查，不覆盖。由正式入口调用候选CLI，不手动绕过release检查。
- [ ] 训练开始日志核对SFT总步数1775；若不同，先停止分析长度/loader/版本，不能静默改变验收目标。SFT每500步保存，最多2份中间checkpoint，结束另存最终adapter；每轮验证仅记录诊断，不选best。
- [ ] SFT验收与合并加载通过，再执行DPO40步，20/40步验证，40步保存。失败立即停止并回收证据。
- [ ] 到保护时限仍未完成：记录INCOMPLETE_TIMEOUT、保留已有checkpoint，关闭实例。本计划不自动续时或重启训练；恢复必须绑定checkpoint文件和原配置，不能仅把路径塞入resume参数。

### Task 5：回收、复核与后续评测准备

**Files:** 新阶段执行报告、主预检索引；原历史smoke记录保持不变。
**Consumes:** 两次训练/合并/加载回执和原始日志。
**Produces:** 正式训练是否通过的结论、关机记录、新模型身份。

- [ ] 打包原始日志、配置、Trainer状态、checkpoint配置及身份、验收回执；下载后逐文件哈希核对。大权重保留目标机，明确本地是否已下载。
- [ ] 对照源参数、实际数据条数、完整步数、验证loss、更新/optimizer状态复核；没有后续质量评测前，只称“完整训练与加载链路通过”。
- [ ] 保存证据后关机，确认页面“已关机”，暂停本次正式监控。
- [ ] 为新SFT/DPO生成模型身份与评估锁，再推进CEval/CMMLU及custom20；不得沿用基座身份锁，不由训练入口自动运行。

## 本轮自审

Day40.2映射Task1/2/4；Day40.3映射Task2/3/5。Task2和Task3的本地实现已完成，验收证据见 `reports/week8/phase2_full_training_entry/verification.json` 与 `docs/week8_full_training_entry.md`。Task3的勾选表示代码和本地失败路径检查完成，不表示真实7B模型GPU冷加载通过。正式GPU训练、回收和质量评测仍未完成。候选release保持execution_enabled=false，待目标机部署时绑定实际关机期限；旧smoke证据保持原样。

## 正式会话启动更新

2026-09-16 17:51:43 UTC已启动Task4。目标预检通过并绑定21:45 UTC平台关机保护；SFT训练头确认1422条、5轮、1775步。完整训练及Task5回收复核尚待完成；进度见 `docs/week8_full_training_target.md`。

## 正式会话完成更新

2026-09-16完整SFT1775步、DPO40步及两次合并/独立加载已通过；90份证据回收并本地复核，实例已关机，监控暂停。Task4完成，Task5回收/复核/关机完成；新模型身份已有原始清单，评估锁重新绑定与正式评测仍待下一阶段。验证loss上升已如实记录。详见 `docs/week8_full_training_target.md`。

## 三模型评测锁更新

2026-09-16已完成320机无卡审核：43个模型文件身份一致，60次custom20输入token对照一致，生成参数一致。合并tokenizer配置差异仅padding_side，不影响当前逐题无padding协议。原基座/SFT/DPO各自的评测锁已部署并通过实际入口校验，结果及回执见`docs/week8_three_model_evaluation.md`。实例已关机、会话定时清除。Task5的新模型评估锁绑定完成，正式GPU质量评测尚未执行。
