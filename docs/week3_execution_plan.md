# Qwen2.5 实习第 3 周执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定模型、数据、随机种子和训练条件下完成 9 次单变量 QLoRA 对比训练，通过固定 20 题、自动辅助证据和双人盲评选择最优 SFT 模型，再以 OpenCompass 0.5.3 对基座与最优模型运行 CEval、CMMLU，并形成可供 Week 4 DPO 使用的归档。

**Architecture:** 以 Week 2 的成功配置作为不可变基线，由实验清单生成九份隔离配置；顺序调度器逐次训练并统一采集 loss、耗时和显存。Day 14 在不合并九个完整模型的前提下，以“基座 + adapter”逐个评测并由两位真实评分者盲评选模；Day 15 只合并已选最优 adapter，OpenCompass 仅验证泛化能力，不参与选模。

**Tech Stack:** Python 3.10.20、PyTorch 2.5.1+cu121、CUDA 12.1、Transformers 4.50.0、LLaMA-Factory 0.9.3、PEFT、bitsandbytes 0.43.3、OpenCompass 0.5.3、pytest、PyYAML、pandas、matplotlib、TensorBoard、NVIDIA RTX 3090 24GB。

## Global Constraints

- 基座模型固定为 `/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct`，不得重新下载或更换 revision。
- 训练数据固定为 Week 2 Day 7 的 4,999 条清洗后 Alpaca 数据；不得同时加入等价 ShareGPT 副本。
- 所有正式训练固定 `seed=42`、Qwen template、`cutoff_len=2048`、4-bit NF4 QLoRA、`lora_target=all`、`lora_alpha=16`、BF16、micro batch 1、gradient accumulation 4、cosine scheduler、warmup ratio 0.1、weight decay 0.01 和 gradient checkpointing。
- Rank 组只修改 `lora_rank`。由于 `lora_alpha=16` 固定，`alpha/rank` 会变化；结果只解释为“不同 rank 配置的整体训练表现”，不宣称是严格隔离缩放后的纯容量比较。
- 正式执行恰好 9 次训练、7 个唯一超参数组合；三次基线重复只衡量实际运行波动，不参与新的超参数比较。
- 为满足老师要求，实验记录保留最后一个有效 logging step 的 loss 作为 **Final Loss**；同时额外报告最近 100 个 logging steps 的平均 loss，仅作为辅助分析指标，不替代 Final Loss。
- Day 14 自动评测只提供数学答案、代码测试和格式检查等辅助证据，不进入模型排序键；最终排序由两位真实评分者的人工评分决定。
- OpenCompass 不参与超参数选择，只验证 Day 14 已选模型在 CEval、CMMLU 上的公开基准泛化能力。
- 九个 run 只保存 adapter 和审计文件；只为 Day 14 已选出的最优 adapter 导出完整合并模型。
- 原始日志、失败日志和模型回答必须保留真实内容；不得覆盖失败目录、补写不存在的数据或选择性删除低分回答。
- 未经明确确认，不提交模型权重、数据集大文件、缓存、checkpoint 或含隐私信息的材料到 Git。

---

## 1. 文档用途与执行依据

这份文档是 Week 3 的长期执行入口，用于回答：每天做什么、为什么这样设计、如何运行、保留哪些证据、怎样验收、哪些结论不能下。

执行优先级：

1. 老师发布的 Week 3 原始要求和验收标准最高。
2. 已批准的 [Week 3 SFT 优化设计](superpowers/specs/2026-08-03-week3-sft-optimization-design.md) 规定实验控制、选模和证据边界。
3. Week 2 的真实配置、数据、日志和质量限制是本周基线，不把预期值写成实际结果。
4. 每日完成后更新 `deliverables/week3/dayN/README.md`；Day 12–13 使用联合目录 `day12_13/`。
5. 如果老师后续修改要求，先更新本计划和实验清单，再启动受影响的正式 run。

本计划只设计和组织工作，不表示 AutoDL GPU、九次训练、人工评分或 OpenCompass 已经完成。

## 2. 本周原始要求与最终验收

| 日期 | 老师要求 | 主要交付 |
|---|---|---|
| Day 11 | rank、learning rate、epochs 三组对比；固定 seed 和数据 | 含假设的实验计划表 |
| Day 12–13 | 依次运行 9 组；记录 Final Loss、训练耗时、显存 | 九次原始日志文件夹 |
| Day 14 | `eval_harness.py`、20 个固定高难题、五维双人盲评 | 雷达图、评分表、最优模型标记 |
| Day 15 | OpenCompass 加载 CEval、CMMLU；比较基座和最优 SFT | 客观分数表 |
| Day 16 | 第 3 周报告；归档最优 SFT，准备 DPO | 周报、最优模型路径说明 |

最终验收逐项解释：

1. “至少 3 组有效对比实验”指 Rank、Learning rate、Epoch 三个单变量实验组均有成功 run 和可解释数据。
2. “明确的数据支撑最优选择”指冻结的 20 题、两位真实评分者、预注册权重与排序规则、原始回答和评分理由均可追溯。
3. “OpenCompass 跑通且出分”指基座与 Day 14 最优 SFT 都完成 CEval、CMMLU，退出码为 0，并产生原始结果及 CSV/TXT 汇总。
4. “周报提交”指报告逐条回答老师要求、结果、限制与路径，不把 loss 下降扩张成质量提升。

## 3. 当前基线与工作目录

| 项目 | 当前已验证事实 |
|---|---|
| GPU | RTX 3090，24GB |
| 数据盘 | `/root/autodl-tmp`，100GB；Week 2 导出前约 75GB 可用 |
| Conda 环境 | `llm_exp`，Python 3.10.20 |
| 基座模型 | Qwen2.5-7B-Instruct |
| 数据 | 4,999 条清洗后 Alpaca |
| Week 2 正式配置 | rank 8、alpha 16、lr `1e-4`、3 epochs |
| Week 2 训练 | 3,747 steps，3:12:12，loss 全部有限 |
| Week 2 adapter | 80,792,096 bytes |
| Week 2 质量结论 | 三题盲评基座 25/30、合并模型 22/30，未证明 SFT 更优 |

远端目录固定为：

```bash
export WEEK3_ROOT=/root/autodl-tmp/qwen25-week3
export BASE_MODEL=/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct
export WEEK2_DATA=/root/autodl-tmp/qwen25-week2/day8/data

mkdir -p \
  "$WEEK3_ROOT/configs/experiments" \
  "$WEEK3_ROOT/data" \
  "$WEEK3_ROOT/manifests" \
  "$WEEK3_ROOT/scripts" \
  "$WEEK3_ROOT/staging" \
  "$WEEK3_ROOT/smoke" \
  "$WEEK3_ROOT/runs" \
  "$WEEK3_ROOT/eval" \
  "$WEEK3_ROOT/opencompass" \
  "$WEEK3_ROOT/best_model" \
  "$WEEK3_ROOT/logs"

test -f "$WEEK2_DATA/week2_clean_alpaca.jsonl"
test -f "$WEEK2_DATA/dataset_info.json"
test ! -e "$WEEK3_ROOT/data/week2_clean_alpaca.jsonl"
install -m 0644 "$WEEK2_DATA/week2_clean_alpaca.jsonl" "$WEEK3_ROOT/data/week2_clean_alpaca.jsonl"
install -m 0644 "$WEEK2_DATA/dataset_info.json" "$WEEK3_ROOT/data/dataset_info.json"
sha256sum "$WEEK3_ROOT/data/week2_clean_alpaca.jsonl" "$WEEK3_ROOT/data/dataset_info.json" \
  | tee "$WEEK3_ROOT/manifests/training_data.sha256"
```

开始任何 GPU 工作前：

```bash
conda activate llm_exp
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"
nvidia-smi
df -h /root/autodl-tmp
llamafactory-cli version
python -c "import opencompass; print(opencompass.__version__)"
```

## 4. 九次实验矩阵

| Run ID | 组 | rank | learning rate | epochs | 学习变量 | 预注册假设 |
|---|---|---:|---:|---:|---|---|
| `rank-r8` | Rank | 8 | `1e-4` | 3 | rank | 资源最低但可能容量不足 |
| `rank-r32` | Rank | 32 | `1e-4` | 3 | rank | 可能在容量与资源间取得平衡 |
| `rank-r64` | Rank | 64 | `1e-4` | 3 | rank | 容量最高，但资源与过拟合风险更高 |
| `lr-1e-4` | Learning rate | 8 | `1e-4` | 3 | learning rate | Week 2 稳定基线 |
| `lr-2e-4` | Learning rate | 8 | `2e-4` | 3 | learning rate | 可能更快收敛，也可能振荡或退化 |
| `lr-5e-5` | Learning rate | 8 | `5e-5` | 3 | learning rate | 更新保守，可能保留能力但欠拟合 |
| `epoch-e2` | Epoch | 8 | `1e-4` | 2 | epochs | 时间短、过拟合风险低，但可能学习不足 |
| `epoch-e3` | Epoch | 8 | `1e-4` | 3 | epochs | Week 2 稳定基线 |
| `epoch-e5` | Epoch | 8 | `1e-4` | 5 | epochs | 可能继续拟合，也可能加重能力偏移 |

三次重复基线是 `rank-r8`、`lr-1e-4`、`epoch-e3`。它们的学习参数必须相同，只有 run ID、输出目录和日志路径不同。

按 Week 2 实测速度线性估算：

- Rank 组：约 9.6 GPU 小时，较大 rank 可能更慢。
- Learning-rate 组：约 9.6 GPU 小时。
- Epoch 组：约 10.7 GPU 小时。
- 九次训练合计约 30 GPU 小时，不含 smoke test、失败重试、Day 14 推理和 OpenCompass。

## 5. 目标文件结构与接口

```text
deliverables/week3/
├── day11/
│   ├── README.md
│   ├── configs/
│   │   ├── qwen25_7b_week3_base.yaml
│   │   └── experiments/*.yaml
│   └── source/
│       ├── manifests/experiment_matrix.json
│       ├── results/day11_validation.json
│       ├── scripts/generate_experiment_configs.py
│       ├── scripts/validate_experiment_matrix.py
│       └── tests/test_experiment_matrix.py
├── day12_13/
│   ├── README.md
│   └── source/
│       ├── scripts/run_experiments.py
│       ├── scripts/summarize_experiments.py
│       ├── scripts/collect_experiment_evidence.py
│       ├── tests/test_run_experiments.py
│       ├── tests/test_summarize_experiments.py
│       ├── tests/test_collect_experiment_evidence.py
│       └── results/
│           ├── raw_logs/<run-id>/attempt-<NNN>/
│           ├── raw_logs.sha256
│           ├── experiment_summary.csv
│           ├── experiment_summary.json
│           ├── baseline_variability.json
│           └── completed_adapters.json
├── day14/
│   ├── README.md
│   ├── eval_harness.py
│   ├── evidence/model_scores_radar.png
│   └── source/
│       ├── data/evaluation_questions.json
│       ├── data/evaluation_rubric.json
│       ├── scripts/check_evaluation_novelty.py
│       ├── scripts/run_code_checks.py
│       ├── scripts/prepare_blind_review.py
│       ├── scripts/aggregate_human_scores.py
│       ├── scripts/select_best_model.py
│       ├── scripts/plot_radar.py
│       ├── tests/
│       └── results/
├── day15/
│   ├── README.md
│   ├── configs/opencompass_week3.py
│   ├── configs/qwen25_7b_week3_best_export.yaml
│   └── source/
│       ├── scripts/summarize_opencompass.py
│       ├── tests/test_summarize_opencompass.py
│       └── results/opencompass_scores.csv
└── day16/
    ├── README.md
    ├── REPORT.md
    └── source/results/
        ├── best_model_path.json
        ├── week3_acceptance_matrix.md
        ├── week3_file_manifest.csv
        ├── week3_files.sha256
        └── week3_final_validation.txt
```

核心接口固定如下，后续实现不得随意改名。函数名、参数和返回类型构成跨任务契约：

```python
@dataclass(frozen=True)
class ExperimentSpec:
    run_id: str
    group: str
    lora_rank: int
    learning_rate: float
    num_train_epochs: float
    hypothesis: str

@dataclass(frozen=True)
class RunSummary:
    run_id: str
    attempt_id: str
    status: str
    final_loss: float | None
    last_100_loss_mean: float | None
    train_runtime_seconds: float | None
    wall_clock_runtime_seconds: float | None
    sampled_peak_gpu_memory_mib: int | None
    optimizer_steps: int | None
    expected_optimizer_steps: int
    adapter_path: str | None
    source_config_sha256: str
    runtime_config_sha256: str

```

- `generate_configs(base_config: dict, specs: list[ExperimentSpec]) -> dict[str, dict]`：返回以 run ID 为键的九份配置。
- `validate_matrix(specs: list[ExperimentSpec], configs: dict[str, dict]) -> list[str]`：返回验证错误列表；空列表表示通过。
- `verify_frozen_inputs(matrix_path: Path, base_config_path: Path, config_dir: Path, validation_path: Path) -> list[str]`：把 manifest、基线和九份配置的实际哈希与 Day 11 预期值比较；空列表才允许启动 GPU。
- `build_runtime_config(source_config: dict, run_id: str, attempt_id: str, attempt_dir: Path, resume_checkpoint: Path | None = None) -> dict`：普通 attempt 只改变 `output_dir` 和 `run_name`；显式恢复时才额外写入同 run 的 `resume_from_checkpoint`。
- `summarize_run(run_id: str, attempt_dir: Path, spec: ExperimentSpec) -> RunSummary`：从一个正式 attempt 生成不可变汇总，并重新执行严格成功门禁。
- `run_evaluation(base_model: str, adapter_path: str | None, questions: list[dict]) -> list[dict]`：返回逐题原始推理记录。
- `aggregate_human_scores(rows: list[dict], rubric: dict) -> list[dict]`：返回按模型聚合的两人人工分。
- `select_best_model(human_scores: list[dict], run_summaries: list[dict]) -> dict`：只用人工分和最终资源 tie-break 选择 SFT run。
- `summarize_opencompass(result_root: Path) -> list[dict]`：返回两个模型与两个数据集的四行汇总。

## 6. Day 11：对比实验设计

### 6.1 当日完成标准

- 九行矩阵与老师给出的取值完全一致，三个组各三行。
- 三个重复基线的学习参数相同，并明确标注为复现性运行。
- 九份配置从 Week 2 成功 YAML 生成，不手工复制后逐份改动。
- 验证器证明每组除目标变量及输出标识外无其他差异。
- 数据、模型、基线配置和软件版本均有哈希或版本快照。
- smoke test 与正式 run 分离，不计入九组结果。

### Task 1: 实验矩阵、配置生成器与验证器

**Files:**
- Create: `deliverables/week3/day11/source/manifests/experiment_matrix.json`
- Create: `deliverables/week3/day11/configs/qwen25_7b_week3_base.yaml`
- Create: `deliverables/week3/day11/source/scripts/generate_experiment_configs.py`
- Create: `deliverables/week3/day11/source/scripts/validate_experiment_matrix.py`
- Create: `deliverables/week3/day11/source/tests/test_experiment_matrix.py`
- Create: `deliverables/week3/day11/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Week 2 annotated YAML and the nine rows in Section 4.
- Produces: nine immutable experiment YAML files and `day11_validation.json` for the Day 12–13 scheduler.

- [x] **Step 1: Write the matrix tests**

```python
def test_matrix_has_required_groups_and_values():
    specs = load_specs(MATRIX_PATH)
    assert len(specs) == 9
    assert {s.lora_rank for s in specs if s.group == "rank"} == {8, 32, 64}
    assert {s.learning_rate for s in specs if s.group == "learning_rate"} == {1e-4, 2e-4, 5e-5}
    assert {s.num_train_epochs for s in specs if s.group == "epoch"} == {2.0, 3.0, 5.0}

def test_repeated_baselines_have_identical_learning_parameters():
    rows = {s.run_id: s for s in load_specs(MATRIX_PATH)}
    for run_id in ["rank-r8", "lr-1e-4", "epoch-e3"]:
        row = rows[run_id]
        assert (row.lora_rank, row.learning_rate, row.num_train_epochs) == (8, 1e-4, 3.0)
```

- [x] **Step 2: Run tests and confirm the missing implementation failure**

Run:

```bash
pytest -q deliverables/week3/day11/source/tests/test_experiment_matrix.py
```

Expected: FAIL because `generate_experiment_configs.py` and the matrix do not exist yet.

- [x] **Step 3: Implement the nine-row JSON manifest and generator**

The generator loads the Week 3 base YAML, changes only `lora_rank`, `learning_rate`, `num_train_epochs`, `run_name` and `output_dir`, and writes filenames equal to `<run_id>.yaml`. It rejects duplicate run IDs and existing output files unless `--check` is used.

Run:

```bash
python deliverables/week3/day11/source/scripts/generate_experiment_configs.py \
  --matrix deliverables/week3/day11/source/manifests/experiment_matrix.json \
  --base deliverables/week3/day11/configs/qwen25_7b_week3_base.yaml \
  --output deliverables/week3/day11/configs/experiments
```

- [x] **Step 4: Implement difference validation**

Within each group, the only permitted learning-field difference is its named variable. Globally permitted metadata differences are `run_name` and `output_dir`. The validator also checks fixed seed, data path, model path, template, cutoff, alpha, target, quantization, batch, scheduler and precision.

Run:

```bash
python deliverables/week3/day11/source/scripts/validate_experiment_matrix.py \
  --matrix deliverables/week3/day11/source/manifests/experiment_matrix.json \
  --base deliverables/week3/day11/configs/qwen25_7b_week3_base.yaml \
  --config-dir deliverables/week3/day11/configs/experiments \
  --output deliverables/week3/day11/source/results/day11_validation.json
```

Expected: `valid: true`, `run_count: 9`, `group_counts: {rank: 3, learning_rate: 3, epoch: 3}`.

- [x] **Step 5: Run unit tests and YAML parser validation**

```bash
pytest -q deliverables/week3/day11/source/tests/test_experiment_matrix.py
python - <<'PY'
from pathlib import Path
import yaml
for path in sorted(Path('deliverables/week3/day11/configs/experiments').glob('*.yaml')):
    yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_count=9')
PY
```

Expected: all tests pass and `yaml_count=9`.

- [x] **Step 6: Perform one isolated smoke test on AutoDL**

Create a smoke-only config with `max_samples=8`, `num_train_epochs=1`, `save_steps=1000` and a path under `$WEEK3_ROOT/smoke/`. It may validate loading and one short training, but its output must never appear in the formal matrix or summary.

- [x] **Step 7: Document Day 11 and update the repository entry**

`day11/README.md` records the teacher requirement, matrix, hypotheses, fixed controls, rank scaling limitation, smoke result and formal run paths. Root `README.md` adds the Week 3 plan and Day 11 entry without changing completed Week 1/2 facts.

- [x] **Step 8: Commit Day 11**

```bash
git add README.md docs/week3_execution_plan.md deliverables/week3/day11
git commit -m "docs: define week 3 experiment matrix"
```

## 7. Day 12–13：九次批量训练

### 7.1 当日完成标准

- 调度器严格按 manifest 顺序串行执行九次正式训练。
- Day 12 负责实现、测试、冻结输入并启动；Day 13 继续同一个任务，完成终态核验、日志回收与汇总，不重复启动矩阵。
- 每个 run 有冻结 source config；每个 attempt 有独立 runtime config、目录、完整日志、退出码和 adapter 逐文件哈希清单。
- 启动前使用 Day 11 验证文件中的预期 SHA-256 校验 manifest、基线和九份 YAML。
- 每个成功 run 都记录 Final Loss、最近 100 步平均 loss、框架训练耗时、墙钟耗时、1 秒采样峰值显存和 optimizer steps。
- 退出码、步数、loss、trainer 状态和 adapter 完整性必须全部通过，才能标记为成功。
- 九组原始日志按 run/attempt 回收到本地 `raw_logs/`，但不回收权重、checkpoint、optimizer state 或训练数据。
- 三次基线分别保留，并计算实际运行波动，不作为新超参数。

### Task 2: 顺序训练调度器与显存采样

**Files:**
- Create: `deliverables/week3/day12_13/source/scripts/run_experiments.py`
- Create: `deliverables/week3/day12_13/source/tests/test_run_experiments.py`
- Create: `deliverables/week3/day12_13/README.md`

**Interfaces:**
- Consumes: Day 11 manifest、`day11_validation.json` and nine validated YAML files.
- Produces: `$WEEK3_ROOT/runs/<run-id>/attempt-<NNN>/` with source/runtime configs, logs, status, memory samples and adapter output.

- [ ] **Step 1: Write scheduler state tests**

```python
def test_completed_run_is_not_overwritten(tmp_path):
    write_status(tmp_path, run_id="rank-r8", status="completed", exit_code=0)
    assert decide_action(tmp_path / "rank-r8") == "skip_completed"

def test_hash_mismatch_blocks_before_launch(tmp_path):
    bundle = write_frozen_bundle(tmp_path, actual_config_hash="actual", expected_config_hash="expected")
    assert verify_frozen_inputs(
        bundle.matrix_path,
        bundle.base_config_path,
        bundle.config_dir,
        bundle.validation_path,
    ) == ["config hash mismatch: rank-r8"]

def test_failed_run_is_not_automatically_retried(tmp_path):
    write_attempt_status(tmp_path, run_id="rank-r32", attempt=1, status="failed", exit_code=1)
    assert decide_action(tmp_path / "rank-r32", allow_retry=False) == "skip_failed"

def test_explicit_retry_uses_next_attempt_directory(tmp_path):
    write_attempt_status(tmp_path, run_id="rank-r32", attempt=1, status="failed", exit_code=1)
    assert next_attempt_dir(tmp_path / "rank-r32") == tmp_path / "rank-r32" / "attempt-002"

def test_stale_running_status_becomes_interrupted(tmp_path):
    write_attempt_status(tmp_path, run_id="rank-r64", attempt=1, status="running", pid=999999)
    assert decide_action(tmp_path / "rank-r64", process_alive=False) == "mark_interrupted"

def test_preflight_blocks_busy_gpu():
    assert preflight_errors(gpu_process_count=1, free_gpu_mib=24000, free_disk_gib=50) == ["gpu busy"]

def test_preflight_blocks_low_resources():
    assert preflight_errors(gpu_process_count=0, free_gpu_mib=19000, free_disk_gib=14) == [
        "free GPU memory below 20000 MiB",
        "free disk below 15 GiB",
    ]
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
pytest -q deliverables/week3/day12_13/source/tests/test_run_experiments.py
```

Expected: FAIL because scheduler functions are not implemented.

- [ ] **Step 3: Implement sequential execution**

For each manifest row the scheduler must:

1. Load `day11_validation.json`; verify the manifest、base config and each source config against its stored expected SHA-256 before any GPU subprocess starts.
2. Select `attempt-001` for an unstarted run. Never auto-retry a failed/interrupted attempt; an explicitly approved retry uses the next unused zero-padded directory such as `attempt-002`.
3. Copy the frozen source YAML into the attempt and create a runtime YAML. A fresh attempt may change only `output_dir` and `run_name`; an explicitly approved checkpoint recovery may additionally set `resume_from_checkpoint` to a hashed checkpoint from the same run/source config. Save both config SHA-256 values and a normalized diff proving no learning field changed.
4. Reject a non-empty attempt directory. A stale `running` status whose recorded process no longer exists becomes `interrupted`; it is never treated as completed.
5. Save preflight time, `df`, full `nvidia-smi`, GPU process list and environment snapshots. Pass only if the target GPU has no other compute process, free GPU memory is at least 20,000 MiB, and `/root/autodl-tmp` has at least 15 GiB free.
6. Start `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 1` into `gpu_memory.csv` and retain its exact PID.
7. Record wall-clock start with `time.monotonic()`; run `llamafactory-cli train <runtime-config>` with unbuffered stdout/stderr into `train.log`; record wall-clock end in `finally`.
8. Stop only the recorded sampler PID in `finally`, then write atomic `status.json` containing timestamps, source/runtime hashes, exit code, wall-clock seconds and failure reason.
9. A failed run is not retried automatically. The scheduler may continue to the next distinct manifest row only while the disk and GPU preflight gates still pass; otherwise it writes `blocked_preflight` and stops launching new runs.

- [ ] **Step 4: Run scheduler tests**

```bash
pytest -q deliverables/week3/day12_13/source/tests/test_run_experiments.py
```

Expected: all scheduler tests pass using mocked subprocesses; no GPU training occurs in tests.

- [ ] **Step 5: Transfer the frozen Day 11 artifacts to AutoDL**

Upload the validated manifest、Day 11 validation、base YAML、nine experiment YAMLs and three Day 12–13 scripts through the established AutoDL file transfer entry into `/root/autodl-tmp/qwen25-week3/staging/`. Generate a new `day12_13_transfer.sha256` covering this exact bundle; do not reuse the earlier smoke-test transfer list. On AutoDL, verify the bundle before installation:

```bash
cd /root/autodl-tmp/qwen25-week3/staging
sha256sum -c day12_13_transfer.sha256
install -m 0644 experiment_matrix.json /root/autodl-tmp/qwen25-week3/manifests/experiment_matrix.json
install -m 0644 day11_validation.json /root/autodl-tmp/qwen25-week3/manifests/day11_validation.json
install -m 0644 qwen25_7b_week3_base.yaml /root/autodl-tmp/qwen25-week3/configs/qwen25_7b_week3_base.yaml
install -m 0644 experiments/*.yaml /root/autodl-tmp/qwen25-week3/configs/experiments/
install -m 0755 run_experiments.py summarize_experiments.py collect_experiment_evidence.py \
  /root/autodl-tmp/qwen25-week3/scripts/
```

The local and remote hash lists must match before formal training. Authentication and upload are performed through the user-authorized AutoDL session; no credential is written into the repository.

- [ ] **Step 6: Launch the formal matrix in a persistent terminal**

```bash
cd /root/autodl-tmp/qwen25-week3
nohup python /root/autodl-tmp/qwen25-week3/scripts/run_experiments.py \
  --matrix /root/autodl-tmp/qwen25-week3/manifests/experiment_matrix.json \
  --validation /root/autodl-tmp/qwen25-week3/manifests/day11_validation.json \
  --base-config /root/autodl-tmp/qwen25-week3/configs/qwen25_7b_week3_base.yaml \
  --config-dir /root/autodl-tmp/qwen25-week3/configs/experiments \
  --run-root /root/autodl-tmp/qwen25-week3/runs \
  > /root/autodl-tmp/qwen25-week3/logs/matrix_scheduler.log 2>&1 &
echo $! > /root/autodl-tmp/qwen25-week3/logs/matrix_scheduler.pid
```

Monitor without restarting:

```bash
tail -f /root/autodl-tmp/qwen25-week3/logs/matrix_scheduler.log
nvidia-smi
```

- [ ] **Step 7: Verify nine terminal statuses**

```bash
python /root/autodl-tmp/qwen25-week3/scripts/run_experiments.py \
  --matrix /root/autodl-tmp/qwen25-week3/manifests/experiment_matrix.json \
  --run-root /root/autodl-tmp/qwen25-week3/runs \
  --status-only
```

Expected after an uninterrupted successful matrix: exactly nine `completed` rows. If the matrix did not finish, the command still prints all nine run IDs and explicitly distinguishes `failed`、`interrupted`、`blocked_preflight` and `not_started`; no missing or blocked run is silently treated as complete.

`completed` is only a preliminary scheduler state. Task 3 must re-evaluate the strict success gate before a run enters the final summary or Day 14 adapter list. 目标是九次均成功；若存在失败，保留失败证据并如实报告，不能因为最低验收只要求三组有效对比而停止剩余计划运行。

### Task 3: 训练日志解析与总表

**Files:**
- Create: `deliverables/week3/day12_13/source/scripts/summarize_experiments.py`
- Create: `deliverables/week3/day12_13/source/scripts/collect_experiment_evidence.py`
- Create: `deliverables/week3/day12_13/source/tests/test_summarize_experiments.py`
- Create: `deliverables/week3/day12_13/source/tests/test_collect_experiment_evidence.py`
- Create: `deliverables/week3/day12_13/source/results/experiment_summary.csv`
- Create: `deliverables/week3/day12_13/source/results/experiment_summary.json`
- Create: `deliverables/week3/day12_13/source/results/baseline_variability.json`
- Create: `deliverables/week3/day12_13/source/results/completed_adapters.json`
- Create: `deliverables/week3/day12_13/source/results/raw_logs/<run-id>/attempt-<NNN>/`
- Create: `deliverables/week3/day12_13/source/results/raw_logs.sha256`
- Modify: `.gitignore`（只为上述 `raw_logs/**/*.log` 增加窄范围例外，保留全局 `*.log` 忽略规则）

**Interfaces:**
- Consumes: nine run directories、Day 11 matrix and each attempt's `trainer_state.json`/`train_results.json`/`gpu_memory.csv`/`status.json`/adapter inventory.
- Produces: `RunSummary` rows used by Day 14 selection and Day 16 reporting.
- `collect_evidence(run_root: Path, output_root: Path, run_ids: list[str]) -> list[str]` 只复制冻结允许列表中的小型审计文件，返回缺失或违规文件错误；空列表才允许打包下载。

- [ ] **Step 1: Write parser edge-case tests**

```python
def test_final_loss_is_last_valid_logging_step():
    rows = [{"step": 1, "loss": 1.4}, {"step": 2, "loss": 1.1}]
    summary = summarize_log_history(rows)
    assert summary.final_loss == 1.1

def test_nonfinite_loss_invalidates_run():
    rows = [{"step": 1, "loss": 1.4}, {"step": 2, "loss": float("nan")}]
    assert summarize_log_history(rows).status == "invalid_nonfinite_loss"

def test_any_nonfinite_loss_invalidates_run_even_if_final_is_finite():
    rows = [{"step": 1, "loss": float("inf")}, {"step": 2, "loss": 1.1}]
    assert summarize_log_history(rows).status == "invalid_nonfinite_loss"

def test_last_100_mean_does_not_replace_final_loss():
    rows = [{"step": i, "loss": float(i)} for i in range(1, 102)]
    summary = summarize_log_history(rows)
    assert summary.final_loss == 101.0
    assert summary.last_100_loss_mean == 51.5

def test_exit_zero_without_expected_steps_is_not_completed(tmp_path):
    fixture = write_run_fixture(tmp_path, exit_code=0, global_step=100, expected_steps=3747)
    assert summarize_run("rank-r8", fixture, baseline_spec()).status == "invalid_incomplete_steps"

def test_exit_zero_without_adapter_weights_is_not_completed(tmp_path):
    fixture = write_run_fixture(tmp_path, exit_code=0, global_step=3747, adapter_weights=False)
    assert summarize_run("rank-r8", fixture, baseline_spec()).status == "invalid_adapter"

def test_framework_and_wall_clock_runtime_remain_separate(tmp_path):
    fixture = write_run_fixture(tmp_path, train_runtime=100.0, wall_clock_runtime=112.5)
    summary = summarize_run("rank-r8", fixture, baseline_spec())
    assert summary.train_runtime_seconds == 100.0
    assert summary.wall_clock_runtime_seconds == 112.5
```

- [ ] **Step 2: Run parser tests and confirm failure**

```bash
pytest -q deliverables/week3/day12_13/source/tests/test_summarize_experiments.py
```

- [ ] **Step 3: Implement `summarize_run` and matrix aggregation**

CSV columns are fixed to:

```text
run_id,attempt_id,group,lora_rank,learning_rate,num_train_epochs,status,final_loss,last_100_loss_mean,train_runtime_seconds,train_runtime_hhmmss,wall_clock_runtime_seconds,wall_clock_runtime_hhmmss,sampled_peak_gpu_memory_mib,optimizer_steps,expected_optimizer_steps,adapter_size_bytes,adapter_path,source_config_sha256,runtime_config_sha256,exit_code
```

`final_loss` comes only from the last valid logging step, but any NaN/Inf anywhere in the recorded loss history invalidates the attempt. `train_runtime_seconds` comes from `train_results.json` and is the formal teacher-facing training time; `wall_clock_runtime_seconds` comes from scheduler monotonic timestamps and is auxiliary audit evidence. `sampled_peak_gpu_memory_mib` is the maximum parsed `memory.used` sample and is labeled as sampled, not exact allocator or process-level peak.

汇总器只有在以下条件全部满足时才能输出 `completed`：scheduler exit code 为 0；`trainer_state.json` 和 `train_results.json` 解析成功；每一个已记录 loss 都是有限值；`global_step == expected_optimizer_steps`；`adapter_config.json` 和至少一个非空 adapter 权重文件存在；不存在中断或失败原因。根据本周已验证的 4,999 条数据、micro batch 1、gradient accumulation 4，预期 optimizer steps 明确为 2 epochs = 2,498、3 epochs = 3,747、5 epochs = 6,245；实现仍须由 manifest 和有效 batch 计算并与这三个冻结期望值交叉核对。

`completed_adapters.json` 只包含成功的正式 run，并记录 run ID、attempt ID、adapter 绝对路径、终态和 `adapter_files.sha256` 清单路径。adapter 是一个目录，因此不虚构含义不清的单一 “adapter SHA-256” 字段。Day 14 读取此文件，不按目录名称自行扫描模型。

- [ ] **Step 4: Compute repeated-baseline variability**

For `rank-r8`, `lr-1e-4`, `epoch-e3`, save each raw value plus mean, standard deviation, minimum and maximum for Final Loss, both runtime measures and sampled peak memory. If fewer than three succeeded, mark the variability result incomplete. These repeated runs quantify realized runtime variation and do not create a new hyperparameter comparison or enter the Day 14 quality ranking.

- [ ] **Step 5: Run tests and summarize real runs on AutoDL**

```bash
pytest -q deliverables/week3/day12_13/source/tests/test_summarize_experiments.py
python /root/autodl-tmp/qwen25-week3/scripts/summarize_experiments.py \
  --matrix /root/autodl-tmp/qwen25-week3/manifests/experiment_matrix.json \
  --run-root /root/autodl-tmp/qwen25-week3/runs \
  --csv /root/autodl-tmp/qwen25-week3/logs/experiment_summary.csv \
  --json /root/autodl-tmp/qwen25-week3/logs/experiment_summary.json \
  --baseline-output /root/autodl-tmp/qwen25-week3/logs/baseline_variability.json \
  --adapters-output /root/autodl-tmp/qwen25-week3/logs/completed_adapters.json
```

- [ ] **Step 6: Test and collect the teacher-required raw log folder**

`collect_experiment_evidence.py` uses an exact basename allowlist: `train.log`、`status.json`、`gpu_memory.csv`、`trainer_state.json`、`train_results.json`、`source_config.yaml`、`runtime_config.yaml`、`config_diff.json`、`preflight.txt`、`environment.txt` and `adapter_files.sha256`. It rejects symlinks and never traverses or copies adapter weight files、optimizer state、checkpoint directories or training data. Write tests proving an allowlisted log is copied and a `.safetensors` fixture is excluded, then run:

```python
def test_collector_copies_allowlisted_logs(tmp_path):
    files = minimal_complete_evidence()
    files["train.log"] = "real log"
    run_root = write_attempt_files(tmp_path, files)
    output_root = tmp_path / "evidence"
    assert collect_evidence(run_root, output_root, ["rank-r8"]) == []
    assert (output_root / "rank-r8" / "attempt-001" / "train.log").read_text() == "real log"

def test_collector_excludes_weights_and_checkpoints(tmp_path):
    files = minimal_complete_evidence()
    files.update({"adapter/model.safetensors": "weight", "adapter/checkpoint-500/state.bin": "state"})
    run_root = write_attempt_files(tmp_path, files)
    output_root = tmp_path / "evidence"
    assert collect_evidence(run_root, output_root, ["rank-r8"]) == []
    assert not list(output_root.rglob("*.safetensors"))
    assert not list(output_root.rglob("checkpoint-*"))
```

```bash
pytest -q deliverables/week3/day12_13/source/tests/test_collect_experiment_evidence.py
python /root/autodl-tmp/qwen25-week3/scripts/collect_experiment_evidence.py \
  --matrix /root/autodl-tmp/qwen25-week3/manifests/experiment_matrix.json \
  --run-root /root/autodl-tmp/qwen25-week3/runs \
  --output-root /root/autodl-tmp/qwen25-week3/logs/day12_13_evidence/raw_logs \
  --checksum-output /root/autodl-tmp/qwen25-week3/logs/day12_13_evidence/raw_logs.sha256
install -m 0644 \
  /root/autodl-tmp/qwen25-week3/logs/experiment_summary.csv \
  /root/autodl-tmp/qwen25-week3/logs/experiment_summary.json \
  /root/autodl-tmp/qwen25-week3/logs/baseline_variability.json \
  /root/autodl-tmp/qwen25-week3/logs/completed_adapters.json \
  /root/autodl-tmp/qwen25-week3/logs/day12_13_evidence/
cd /root/autodl-tmp/qwen25-week3/logs
tar -czf day12_13_evidence.tgz day12_13_evidence
sha256sum day12_13_evidence.tgz > day12_13_evidence.tgz.sha256
```

通过已授权的 AutoDL 文件传输入口下载证据包及其 SHA-256，在下载目录验证压缩包，再把包内 `raw_logs/`、`raw_logs.sha256` 和四个 summary 文件放入 `deliverables/week3/day12_13/source/results/`。在 `.gitignore` 增加 `!deliverables/week3/day12_13/source/results/raw_logs/**/*.log`，只允许本次老师要求的原始训练日志进入版本控制，不放开其他日志。最后在仓库根目录验证本地原始日志清单：

```bash
# 在包含证据包及其校验文件的下载目录执行
shasum -a 256 -c day12_13_evidence.tgz.sha256
# 解压并复制到仓库后，在仓库根目录执行
(cd deliverables/week3/day12_13/source/results && shasum -a 256 -c raw_logs.sha256)
test -z "$(find deliverables/week3/day12_13/source/results/raw_logs -type f \
  \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' \
  -o -path '*/checkpoint-*/*' \) -print -quit)"
```

Expected: all nine run IDs are present; every recorded attempt has complete log/status/config evidence; no forbidden large file is present; checksum verification exits 0. A failed run remains in `raw_logs/` with its real terminal state.

- [ ] **Step 7: Commit Day 12–13 evidence**

```bash
git add .gitignore deliverables/week3/day12_13
git commit -m "feat: complete week 3 training matrix"
```

## 8. Day 14：固定评测、双人盲评与选模

### 8.1 当日完成标准

- 20 题在读取模型回答前冻结，数学/推理/代码为 7/7/6。
- 题目对训练集执行精确与模糊污染检查。
- 基座和九个 adapter 使用相同模板、生成参数、设备与精度。
- 自动评测只生成辅助证据，不直接排序。
- 两位真实评分者独立盲评，其中一位是同事或导师。
- 评分权重为 30%/25%/20%/15%/10%，映射在评分冻结后解盲。
- 最优模型由预注册的人工排序规则选出，并生成五维雷达图和原始表。

### Task 4: 冻结 20 题与污染检查

**Files:**
- Create: `deliverables/week3/day14/source/data/evaluation_questions.json`
- Create: `deliverables/week3/day14/source/data/evaluation_rubric.json`
- Create: `deliverables/week3/day14/source/scripts/check_evaluation_novelty.py`
- Create: `deliverables/week3/day14/source/tests/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: 4,999 条训练数据和人工编写的题目/参考答案。
- Produces: frozen question SHA-256 and novelty report consumed by `eval_harness.py`.

- [ ] **Step 1: Write dataset contract tests**

```python
def test_question_distribution_and_unique_ids():
    questions = load_questions(QUESTIONS_PATH)
    assert len(questions) == 20
    assert len({q["id"] for q in questions}) == 20
    assert Counter(q["category"] for q in questions) == {"math": 7, "reasoning": 7, "code": 6}

def test_rubric_weights():
    rubric = load_rubric(RUBRIC_PATH)
    assert rubric["weights"] == {
        "accuracy": 0.30,
        "completeness": 0.25,
        "logic": 0.20,
        "safety": 0.15,
        "format": 0.10,
    }
```

- [ ] **Step 2: Write the 20 questions before model inference**

Every question contains `id`, `category`, `difficulty`, `messages`, `reference`, `automatic_check`, `human_scoring_notes`. Code questions define a function signature and pytest assertions; reasoning questions have explicit required claims; math questions have verifiable results and derivations.

- [ ] **Step 3: Run exact and fuzzy novelty checks**

```bash
python deliverables/week3/day14/source/scripts/check_evaluation_novelty.py \
  --questions deliverables/week3/day14/source/data/evaluation_questions.json \
  --training-data /root/autodl-tmp/qwen25-week3/data/week2_clean_alpaca.jsonl \
  --output deliverables/week3/day14/source/results/evaluation_novelty.json
```

Acceptance: no exact match; every high-similarity candidate is retained for manual review with source ID and similarity, not silently discarded.

- [ ] **Step 4: Freeze hashes**

```bash
shasum -a 256 \
  deliverables/week3/day14/source/data/evaluation_questions.json \
  deliverables/week3/day14/source/data/evaluation_rubric.json \
  > deliverables/week3/day14/source/results/evaluation_inputs.sha256
```

### Task 5: `eval_harness.py` 与自动辅助证据

**Files:**
- Create: `deliverables/week3/day14/eval_harness.py`
- Create: `deliverables/week3/day14/source/scripts/run_code_checks.py`
- Create: `deliverables/week3/day14/source/tests/test_eval_harness.py`
- Create: `deliverables/week3/day14/source/tests/test_code_checks.py`

**Interfaces:**
- Consumes: base path, optional adapter path, frozen questions and deterministic generation config.
- Produces: raw JSONL responses and automatic evidence JSON; neither file contains final model ranking.

- [ ] **Step 1: Write harness tests with fake model/tokenizer**

```python
def test_adapter_and_base_use_identical_generation_config(fake_model):
    base = run_evaluation("base", None, QUESTIONS, model_factory=fake_model)
    sft = run_evaluation("base", "adapter", QUESTIONS, model_factory=fake_model)
    assert base[0]["generation_config"] == sft[0]["generation_config"]

def test_raw_response_is_never_rewritten():
    result = make_result(raw_response="wrong but original")
    assert result["raw_response"] == "wrong but original"
```

- [ ] **Step 2: Implement deterministic inference**

Use Qwen chat template, `do_sample=false`, fixed `max_new_tokens=512`, identical stop tokens, BF16 compute and one model at a time. Save question hash, base hash, adapter hash, token counts, duration, raw response and error status for all 200 model-question pairs: one base plus nine adapters, each with 20 questions.

- [ ] **Step 3: Implement safe code-check preflight**

Code execution is allowed only if an isolation backend passes preflight. Preferred command is `bwrap` with a new network namespace, read-only runtime binds, temporary writable directory, CPU/memory/file-size limits and a 3-second timeout. If isolation is unavailable, set `code_execution_status=disabled_no_sandbox`; do not execute generated code directly on the host and do not fabricate pass/fail results.

- [ ] **Step 4: Run unit tests**

```bash
pytest -q \
  deliverables/week3/day14/source/tests/test_eval_harness.py \
  deliverables/week3/day14/source/tests/test_code_checks.py
```

- [ ] **Step 5: Run base and nine adapters**

```bash
python deliverables/week3/day14/eval_harness.py \
  --base-model /root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct \
  --adapter-manifest /root/autodl-tmp/qwen25-week3/manifests/completed_adapters.json \
  --questions deliverables/week3/day14/source/data/evaluation_questions.json \
  --output-dir /root/autodl-tmp/qwen25-week3/eval/raw_responses
```

Expected: 200 terminal records if all nine adapters are valid; failed runs remain explicit and reduce the count of evaluable SFT models.

### Task 6: 双人盲评、解盲与人工选模

**Files:**
- Create: `deliverables/week3/day14/source/scripts/prepare_blind_review.py`
- Create: `deliverables/week3/day14/source/scripts/aggregate_human_scores.py`
- Create: `deliverables/week3/day14/source/scripts/select_best_model.py`
- Create: `deliverables/week3/day14/source/tests/test_human_scoring.py`
- Create: `deliverables/week3/day14/source/results/blind_mapping.json`
- Create: `deliverables/week3/day14/source/results/best_model.json`

**Interfaces:**
- Consumes: raw responses, rubric and two frozen human score files.
- Produces: human-only ranking and the selected SFT run for Day 15.

- [ ] **Step 1: Write scoring and leakage tests**

```python
def test_automatic_metrics_are_not_ranking_inputs():
    assert "automatic_score" not in inspect.signature(select_best_model).parameters

def test_human_score_formula():
    row = {"accuracy": 5, "completeness": 4, "logic": 3, "safety": 5, "format": 4}
    assert weighted_score(row) == 4.25

def test_tie_breakers_remain_human_first():
    winner = select_best_model(HUMAN_TIE_FIXTURE, RUN_SUMMARIES)
    assert winner["tie_breaker"] in {"human_accuracy", "rater_agreement", "resource_cost"}
```

- [ ] **Step 2: Generate deterministic blinded packages**

Use a fixed blinding seed different from the training seed. Randomize model labels per question so position does not reveal identity. Store `blind_mapping.json` separately and do not place model paths or run IDs in reviewer files.

- [ ] **Step 3: Collect two real score files**

Each reviewer scores every answer 0–5 on accuracy, completeness, logic, safety and format, and writes a short reason. Reviewer 2 must be a real colleague or mentor. Both files are marked immutable with SHA-256 before unblinding.

- [ ] **Step 4: Aggregate and select**

Selection order is fixed:

1. Two-rater weighted total mean.
2. If tied, two-rater human accuracy mean.
3. If still tied, smaller rater total-score disagreement.
4. If still tied, smaller adapter then shorter runtime.

Automatic math/code/format evidence is included beside the human report but is never passed to `select_best_model`.

- [ ] **Step 5: Run scoring tests and real aggregation**

```bash
pytest -q deliverables/week3/day14/source/tests/test_human_scoring.py
python deliverables/week3/day14/source/scripts/aggregate_human_scores.py \
  --rubric deliverables/week3/day14/source/data/evaluation_rubric.json \
  --reviewer-1 /root/autodl-tmp/qwen25-week3/eval/reviewer_1_scores.csv \
  --reviewer-2 /root/autodl-tmp/qwen25-week3/eval/reviewer_2_scores.csv \
  --mapping deliverables/week3/day14/source/results/blind_mapping.json \
  --output deliverables/week3/day14/source/results/human_scores.csv
python deliverables/week3/day14/source/scripts/select_best_model.py \
  --human-scores deliverables/week3/day14/source/results/human_scores.csv \
  --run-summaries deliverables/week3/day12_13/source/results/experiment_summary.json \
  --output deliverables/week3/day14/source/results/best_model.json
```

### Task 7: 雷达图与 Day 14 报告

**Files:**
- Create: `deliverables/week3/day14/source/scripts/plot_radar.py`
- Create: `deliverables/week3/day14/source/tests/test_plot_radar.py`
- Create: `deliverables/week3/day14/evidence/model_scores_radar.png`
- Create: `deliverables/week3/day14/README.md`

**Interfaces:**
- Consumes: five-dimensional two-rater averages.
- Produces: deterministic PNG plus underlying CSV/JSON.

- [ ] **Step 1: Test that plot data and score table match**

```python
def test_radar_uses_human_dimension_means_only():
    data = build_radar_rows(HUMAN_SCORES)
    assert set(data.columns) == {"model_id", "accuracy", "completeness", "logic", "safety", "format"}
```

- [ ] **Step 2: Generate chart and verify dimensions**

```bash
python deliverables/week3/day14/source/scripts/plot_radar.py \
  --scores deliverables/week3/day14/source/results/human_scores.csv \
  --output deliverables/week3/day14/evidence/model_scores_radar.png
file deliverables/week3/day14/evidence/model_scores_radar.png
```

- [ ] **Step 3: Document the selected model and limitations**

README distinguishes group best, overall best SFT and base comparison. It states that automatic evidence did not rank models and that OpenCompass has not yet participated in selection.

- [ ] **Step 4: Commit Day 14**

```bash
git add deliverables/week3/day14
git commit -m "feat: evaluate and select week 3 SFT model"
```

## 9. Day 15：OpenCompass 通用评测

### 9.1 当日完成标准

- Day 14 `best_model.json` 在任何 OpenCompass 正式结果产生前已冻结。
- 只合并已选最优 adapter，合并模型可独立加载。
- OpenCompass 版本固定为 0.5.3，实际 CEval/CMMLU 配置名由当前安装列出并归档。
- 基座与最优 SFT 使用相同生成式 chat 配置和数据快照。
- 两个模型都生成 CEval、CMMLU 原始预测、学科分数和聚合表。
- OpenCompass 结果不改变 Day 14 最优模型。

### Task 8: 最优模型合并与 OpenCompass

**Files:**
- Create: `deliverables/week3/day15/configs/qwen25_7b_week3_best_export.yaml`
- Create: `deliverables/week3/day15/configs/opencompass_week3.py`
- Create: `deliverables/week3/day15/source/scripts/summarize_opencompass.py`
- Create: `deliverables/week3/day15/source/tests/test_summarize_opencompass.py`
- Create: `deliverables/week3/day15/source/results/opencompass_scores.csv`
- Create: `deliverables/week3/day15/README.md`

**Interfaces:**
- Consumes: Day 14 selected adapter and OpenCompass 0.5.3 CEval/CMMLU outputs.
- Produces: independently loadable best merged model and base-versus-SFT benchmark table.

- [x] **Step 1: Validate selected model input**

Check that `best_model.json` contains a successful run ID, adapter path, adapter hash, human score source hashes and the selection timestamp. Reject any file that contains an OpenCompass score field.

- [x] **Step 2: Create and validate the export config**

Set `model_name_or_path` to the fixed base, `adapter_name_or_path` to the selected adapter, `template: qwen`, `finetuning_type: lora`, and `export_dir: /root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged`. Save configuration hash before export.

- [x] **Step 3: Export and independently load the best model**

```bash
set -o pipefail
llamafactory-cli export \
  /root/autodl-tmp/qwen25-week3/configs/qwen25_7b_week3_best_export.yaml \
  2>&1 | tee /root/autodl-tmp/qwen25-week3/logs/best_model_export.log
test "${PIPESTATUS[0]}" -eq 0
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
p = '/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged'
AutoTokenizer.from_pretrained(p, local_files_only=True)
AutoModelForCausalLM.from_pretrained(p, local_files_only=True, device_map='cpu')
print('merged_model_load=PASS')
PY
```

- [x] **Step 4: Resolve OpenCompass 0.5.3 configs and data**

```bash
OPENCOMPASS_ROOT="$(python -c 'from pathlib import Path; import opencompass; print(Path(opencompass.__file__).resolve().parents[1])')"
cd "$OPENCOMPASS_ROOT"
python tools/list_configs.py ceval cmmlu \
  | tee /root/autodl-tmp/qwen25-week3/opencompass/list_configs.txt
python run.py /root/autodl-tmp/qwen25-week3/opencompass/opencompass_week3.py \
  --dry-run -w /root/autodl-tmp/qwen25-week3/opencompass/dry_run \
  2>&1 | tee /root/autodl-tmp/qwen25-week3/logs/opencompass_dry_run.log
```

Use the generated `_gen` configuration names reported by the installed version; record exact names rather than assuming hashes from another release.

- [x] **Step 5: Run a small smoke task**

Use one CEval subject for both models, verify chat rendering and result parsing, and keep the smoke output under `opencompass/smoke/`. Smoke scores are not copied into the formal table.

- [x] **Step 6: Run formal CEval and CMMLU**

```bash
OPENCOMPASS_ROOT="$(python -c 'from pathlib import Path; import opencompass; print(Path(opencompass.__file__).resolve().parents[1])')"
cd "$OPENCOMPASS_ROOT"
python run.py /root/autodl-tmp/qwen25-week3/opencompass/opencompass_week3.py \
  -w /root/autodl-tmp/qwen25-week3/opencompass/formal \
  --dump-eval-details \
  2>&1 | tee /root/autodl-tmp/qwen25-week3/logs/opencompass_formal.log
test "${PIPESTATUS[0]}" -eq 0
```

- [x] **Step 7: Test and generate the score table**

```python
def test_summary_contains_both_models_and_datasets():
    rows = summarize_opencompass(FIXTURE_ROOT)
    assert {(r["model"], r["dataset"]) for r in rows} == {
        ("base", "ceval"), ("base", "cmmlu"),
        ("best_sft", "ceval"), ("best_sft", "cmmlu"),
    }
```

```bash
pytest -q deliverables/week3/day15/source/tests/test_summarize_opencompass.py
python deliverables/week3/day15/source/scripts/summarize_opencompass.py \
  --result-root /root/autodl-tmp/qwen25-week3/opencompass/formal \
  --output deliverables/week3/day15/source/results/opencompass_scores.csv
```

- [x] **Step 8: Commit Day 15**

```bash
git add deliverables/week3/day15
git commit -m "feat: evaluate best SFT with OpenCompass"
```

## 10. Day 16：周报与最优模型归档

### 10.1 当日完成标准

- 报告覆盖九次状态、三组比较、Final Loss、辅助 loss、耗时、显存、盲评和 OpenCompass。
- 最优模型结论严格来自 Day 14 人工排序；OpenCompass 只验证泛化。
- 最优 adapter 和合并模型都有路径、文件清单和哈希。
- DPO 输入说明明确使用哪个模型形态，不使用模糊别名。
- 四项老师验收逐项给出通过、未通过或部分通过及证据。

### Task 9: 报告、归档清单与最终验证

**Files:**
- Create: `deliverables/week3/day16/REPORT.md`
- Create: `deliverables/week3/day16/README.md`
- Create: `deliverables/week3/day16/source/results/best_model_path.json`
- Create: `deliverables/week3/day16/source/results/week3_acceptance_matrix.md`
- Create: `deliverables/week3/day16/source/results/week3_file_manifest.csv`
- Create: `deliverables/week3/day16/source/results/week3_files.sha256`
- Create: `deliverables/week3/day16/source/results/week3_final_validation.txt`
- Create: `deliverables/week3/day16/source/scripts/validate_week3.py`
- Create: `deliverables/week3/day16/source/tests/test_validate_week3.py`
- Create: `Submission/Week3/README.md`
- Modify: `Submission/README.md`
- Modify: `Submission/SHA256SUMS.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: Day 11–15 immutable results and remote model inventory.
- Produces: teacher-facing Week 3 report and exact Week 4 DPO model input.

- [x] **Step 1: Write final consistency checks**

The validation script asserts:

- Experiment matrix has 9 runs and 3 groups.
- Every run has an explicit terminal status.
- Every successful run has Final Loss, runtime and sampled peak memory.
- Human score files come from two reviewers and weights sum to 1.00.
- `best_model.json` selection inputs contain no OpenCompass result.
- OpenCompass table contains base/best-SFT × CEval/CMMLU.
- Best adapter and merged model paths match their inventories and hashes.

- [x] **Step 2: Write the report with fixed structure**

`REPORT.md` sections:

1. 本周目标与验收矩阵。
2. 基线、环境、模型、数据和目录。
3. 九次实验设计、假设和变量控制。
4. Rank 组结果及固定 alpha 导致的解释限制。
5. Learning-rate 组结果。
6. Epoch 组结果。
7. 三次重复基线的实际运行波动。
8. Final Loss、最近 100 步均值、耗时和显存总表。
9. 20 题设计与训练集污染检查。
10. 自动辅助证据与双人盲评方法。
11. 人工评分、雷达图和最优模型选择。
12. 基座与最优 SFT 的对比及不支持的结论。
13. OpenCompass CEval、CMMLU 方法和结果。
14. 失败、异常、限制和复现性。
15. 最优模型归档与 Week 4 DPO 输入。
16. 复现命令、文件索引和 SHA-256。

- [x] **Step 3: Write exact best-model path metadata**

`best_model_path.json` 由脚本从冻结产物组装，核心代码固定为：

```python
best = json.loads(Path("deliverables/week3/day14/source/results/best_model.json").read_text())
metadata = {
    "selected_run_id": best["run_id"],
    "base_model_path": "/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct",
    "adapter_path": best["adapter_path"],
    "merged_model_path": "/root/autodl-tmp/qwen25-week3/best_model/qwen25-7b-week3-best-merged",
    "dpo_input_form": "merged_model",
    "selection_source": "Day 14 two-rater weighted human score",
    "opencompass_used_for_selection": False,
}
```

The two result-dependent fields are copied programmatically from frozen artifacts; they are never guessed while writing the plan.

- [x] **Step 4: Build the teacher submission**

Copy the final experiment table, evaluation rubric, two-rater aggregate scores, radar chart, OpenCompass table, report and best-model path explanation into English-named `Submission/Week3/` paths. Do not copy adapters or full model weights into Git.

- [x] **Step 5: Generate manifests and run final verification**

```bash
python deliverables/week3/day16/source/scripts/validate_week3.py \
  2>&1 | tee deliverables/week3/day16/source/results/week3_final_validation.txt
find deliverables/week3 Submission/Week3 -type f -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 \
  > deliverables/week3/day16/source/results/week3_files.sha256
git diff --check
pytest -q deliverables/week3
```

Expected: validation exit code 0, no malformed Markdown whitespace, and all Week 3 tests pass. If an acceptance item is not met, the validator and report must show its actual status rather than forcing exit 0 through fabricated data.

- [x] **Step 6: Commit Week 3 report and submission**

```bash
git add README.md Submission deliverables/week3 docs/week3_execution_plan.md
git commit -m "docs: complete week 3 optimization report"
```

## 11. 故障处理顺序

### 11.1 OOM

1. 保存原始日志、配置和发生阶段。
2. 检查残留 GPU 进程和实际量化、batch、cutoff、checkpointing。
3. 只修改一个资源变量做诊断。
4. 如果诊断改变正式学习条件，该 run 不再属于原矩阵；恢复原配置后创建新 attempt。
5. 记录修复前后采样峰值显存和速度，不覆盖失败证据。

### 11.2 中断与恢复

- 只从同一 run、同一模型/数据/配置的 checkpoint 恢复。
- 保存中断 step、恢复命令和 checkpoint 哈希。
- `trainer_state.json` 必须保持 step 连续；不连续时在总表中标记异常。

### 11.3 非有限 loss

- 任意 NaN/Inf 使该 run 标记为 `invalid_nonfinite_loss`。
- 不用最后一个有限值冒充 Final Loss。
- 先检查过高学习率、梯度、数据异常和精度，再决定是否按原配置重跑。

### 11.4 磁盘不足

- 暂停启动新 run。
- 先生成目录大小、文件清单和哈希。
- 优先避免创建九个合并模型；只保存 adapter、最近 checkpoint 和日志。
- 删除正式证据前必须取得确认；不能用清理掩盖运行失败。

### 11.5 人工评分缺失

- 自动结果可以完成，但不能代替第二位同事或导师。
- 第二评分者未完成时不解盲、不选最优、不启动正式 OpenCompass。
- 状态明确写作“等待真实第二评分者”，不生成模拟评分。

### 11.6 OpenCompass 问题

- 先保存 `list_configs.py`、dry run 和 smoke 日志。
- 核对 0.5.3 实际配置位置、CEval/CMMLU 数据路径和 chat 模型模板。
- 不通过临时升级 OpenCompass 改变本周固定环境；必须升级时另建环境并在报告说明版本变化。

## 12. 每日时间与 GPU 安排

| 日期 | CPU/准备 | GPU | 人工协作 | 主要风险 |
|---|---:|---:|---:|---|
| Day 11 | 4–6h | smoke 约 10–20min | 无 | 矩阵污染、路径或版本错误 |
| Day 12–13 | 2–4h | 正式训练约 30h | 无 | 中断、OOM、磁盘、日志缺失 |
| Day 14 | 4–6h | 10 模型 × 20 题推理 | 两位评分者各约 2–4h | 题目泄漏、盲评泄漏、评分缺失 |
| Day 15 | 2–4h | 合并、CEval、CMMLU | 无 | 数据下载、chat 配置、评测时长 |
| Day 16 | 5–8h | 通常无需 | 复核签字 | 结果/路径/哈希不一致 |

Day 12–13 的 30 GPU 小时超过两个普通工作日的 16 小时，因此必须尽早启动后台串行任务，并确保 AutoDL 实例不会因平台自动关机策略提前停止。

## 13. 最终口述准备

完成后需要能够清楚回答：

1. 为什么这是三组单变量实验，而不是 27 次全因子搜索。
2. 为什么有九次运行却只有七个唯一超参数组合。
3. 固定 seed 为什么仍可能出现 CUDA 非确定性和调度波动。
4. Rank 实验为什么固定 alpha，以及 `alpha/rank` 变化带来的解释限制。
5. 为什么 Final Loss 是最后一个有效 step，而最近 100 步均值只是辅助。
6. 为什么 loss 不能直接决定模型好坏。
7. 20 题怎样冻结、怎样证明没有训练集泄漏。
8. 自动评测与双人盲评分别承担什么职责。
9. 为什么自动结果不进入选模排序。
10. 为什么只合并一个最优 adapter。
11. OpenCompass 为什么不参与选模，只验证公开基准泛化。
12. 如果最优 SFT 在 CEval/CMMLU 低于基座，怎样如实解释。
13. Week 4 DPO 将从哪个精确模型路径开始。

## 14. 本周最终检查

- [ ] 九行实验矩阵、三组假设和配置哈希已冻结。
- [ ] 三个组内只有目标变量变化；输出路径差异已排除在学习变量外。
- [ ] Rank 组固定 alpha 的限制已写入计划和报告结构。
- [ ] 九次正式 run 均有明确状态和完整原始日志。
- [ ] 每个成功 run 有 Final Loss、最近 100 步均值、耗时、采样峰值显存和 adapter 清单。
- [ ] 三次重复基线单独报告波动，未冒充新超参数。
- [ ] 20 题在模型推理前冻结，分布为 7/7/6，并完成训练集污染检查。
- [ ] 基座和九个 adapter 使用同一推理条件。
- [ ] 自动评测只作辅助，没有进入选模排序键。
- [ ] 两位真实评分者均完成盲评，其中一位为同事或导师。
- [ ] 五维权重为 30%/25%/20%/15%/10%，解盲过程有哈希证据。
- [ ] Day 14 最优 SFT 已按人工规则冻结。
- [ ] 只合并最优 adapter，合并模型可独立加载。
- [ ] OpenCompass 0.5.3 的 CEval、CMMLU 对两个模型均退出码为 0 并出分。
- [ ] OpenCompass 结果未反向改变 Day 14 选模。
- [ ] 第 3 周报告、验收矩阵、雷达图、分数表和最佳路径说明均完成。
- [ ] 完整权重留在 AutoDL，Git 中只有小文件、清单、哈希和报告。
- [ ] Week 4 DPO 输入模型形态、绝对路径和 SHA-256 已明确。

## 15. 官方参考

执行时只引用与固定版本相符的一手来源：

- [OpenCompass 0.5.3 Quick Start](https://opencompass.readthedocs.io/en/stable/get_started/quick_start.html)
- [OpenCompass Prepare Models](https://opencompass.readthedocs.io/en/latest/user_guides/models.html)
- [OpenCompass Task Execution and Monitoring](https://opencompass.readthedocs.io/en/latest/user_guides/experimentation.html)
- LLaMA-Factory 0.9.3 本地安装源码及 Week 2 已归档官方 YAML。
- Qwen2.5-7B-Instruct 本地 `config.json`、Tokenizer 与模型文件哈希。

如果在线文档与当前 0.5.3 安装结构不同，以本地 `tools/list_configs.py`、实际模块路径和 dry-run 结果为准，并把差异写进 Day 15 README。
