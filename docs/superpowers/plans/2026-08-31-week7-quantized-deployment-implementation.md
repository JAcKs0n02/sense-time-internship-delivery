# Week 7 Quantized Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Week 4 最终 DPO 模型完成 AWQ/GPTQ 量化、同口径基准、vLLM OpenAI 服务和 Gradio 文字/图片交互，并形成 Day 34–Day 39 的工程归档与教师提交。

**Architecture:** `deliverables/week7/source/week7_deployment/` 保存确定性数据冻结、模型 manifest、指标计算、OpenAI 流适配和路由等共享逻辑；Day 34–Day 39 目录只保存当天配置、脚本、证据和说明。文字请求路由到 Week 4 量化文本 vLLM，含图请求路由到 Week 5 基座 VLM vLLM，两套模型在 RTX 3090 上按需切换。

**Tech Stack:** Python 3.10、pytest、LLaMA-Factory 0.9.3、AutoAWQ、GPTQModel、Transformers、PyTorch、vLLM 0.6.4.post1、OpenAI Python SDK、Gradio、JSON/JSONL/CSV、Bash。

**Spec:** `docs/superpowers/specs/2026-08-31-week7-quantized-deployment-design.md`

## Global Constraints

- 文本量化输入固定为 Week 4 `reward_corrective_40step_merged`，merged manifest SHA-256 为 `aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c`。
- 若重建 merged model，只能使用 Week 4 adapter SHA-256 `d7a932ee4f28c8950db289126381f5d4dd30a037b238852e87ad8e5a241f9e52` 与既有精确基座。
- VLM 固定为 revision `eed13092ef92e448dd6875b2a00151bd3f7db0ac`，files digest `575d62b8abe01fda09b28a02ff8e6cd04e8df0d15fd03e1642d3a22bbe7eb812`。
- LLaMA-Factory 0.9.3 只负责 GPTQ 导出；AWQ 使用 AutoAWQ，并保存版本兼容性收据。
- AWQ 与 GPTQ 使用同一个 128 条校准集；PPL 使用与校准集互斥的 256 条冻结数据。
- 显存下降门禁以同口径进程实际峰值为准，至少一个量化模型相对 BF16 降低 `30%`。
- 文字和视觉 vLLM 默认不同时常驻；服务脚本发现冲突时退出，不自动杀死进程。
- 模型大权重、checkpoint、缓存、凭据和私有远端信息不进入 Git 或教师提交。
- 没有真实 AutoDL 日志、退出码和结果时，相关状态保持“待远端执行”，不得写成完成或 PASS。
- 老师的 `实习需求.docx`、`实习需求.pdf` 和其他用户未提交修改不纳入本轮提交。

---

### Task 1: 冻结校准与评测数据

**Files:**
- Create: `deliverables/week7/source/week7_deployment/__init__.py`
- Create: `deliverables/week7/source/week7_deployment/hashing.py`
- Create: `deliverables/week7/source/week7_deployment/datasets.py`
- Create: `deliverables/week7/day34/source/scripts/build_quantization_datasets.py`
- Create: `deliverables/week7/day34/source/data/calibration_128.jsonl`
- Create: `deliverables/week7/day35/source/data/perplexity_eval_256.jsonl`
- Create: `deliverables/week7/day35/source/data/generation_eval_20.json`
- Create: `deliverables/week7/day34/source/results/dataset_manifest.json`
- Test: `deliverables/week7/tests/test_datasets.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str` and `stable_record_sha256(record: Mapping[str, object]) -> str`.
- Produces: `freeze_rows(rows: Sequence[Mapping[str, object]], calibration_count: int, perplexity_count: int, seed: int) -> FrozenDatasets` for unit-level deterministic selection.
- Produces: `freeze_quantization_datasets(source: Path, calibration_count: int, perplexity_count: int, seed: int) -> FrozenDatasets`.
- Produces: deterministic calibration/PPL JSONL files plus a manifest consumed by both quantizers.

- [ ] **Step 1: Write deterministic-split tests**

```python
def test_split_is_stable_when_source_order_changes(tmp_path):
    rows = sample_sharegpt_rows(400)
    first = freeze_rows(rows, calibration_count=128, perplexity_count=256, seed=42)
    second = freeze_rows(list(reversed(rows)), calibration_count=128, perplexity_count=256, seed=42)
    assert first.calibration_ids == second.calibration_ids
    assert first.perplexity_ids == second.perplexity_ids

def test_calibration_and_perplexity_sets_are_disjoint():
    frozen = freeze_rows(sample_sharegpt_rows(400), 128, 256, 42)
    assert len(frozen.calibration_ids) == 128
    assert len(frozen.perplexity_ids) == 256
    assert set(frozen.calibration_ids).isdisjoint(frozen.perplexity_ids)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest deliverables/week7/tests/test_datasets.py -q`

Expected: collection fails because `week7_deployment.datasets` does not exist.

- [ ] **Step 3: Implement canonical hashing and deterministic selection**

```python
def stable_record_sha256(record: Mapping[str, object]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def freeze_rows(rows, calibration_count=128, perplexity_count=256, seed=42):
    unique = {stable_record_sha256(row): row for row in rows}
    ordered = sorted(unique.items())
    ranked = sorted(ordered, key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest())
    return FrozenDatasets(ranked[:calibration_count], ranked[calibration_count:calibration_count + perplexity_count])
```

Normalize each source row to a stable `id`, `messages`, `source_sha256` record. Reject malformed conversations, duplicate canonical hashes and source sets smaller than 384 unique rows.

- [ ] **Step 4: Run RED-to-GREEN tests**

Run: `python3 -m pytest deliverables/week7/tests/test_datasets.py -q`

Expected: all deterministic, count, disjointness and manifest-hash tests pass.

- [ ] **Step 5: Build formal frozen files**

Run:

```bash
PYTHONPATH=deliverables/week7/source python3 deliverables/week7/day34/source/scripts/build_quantization_datasets.py \
  --source deliverables/week2/day7/data/week2_clean_sharegpt.jsonl \
  --calibration-output deliverables/week7/day34/source/data/calibration_128.jsonl \
  --perplexity-output deliverables/week7/day35/source/data/perplexity_eval_256.jsonl \
  --manifest-output deliverables/week7/day34/source/results/dataset_manifest.json \
  --seed 42
```

Expected: exactly 128 and 256 rows, zero overlap, stable SHA-256 values and no mutation of Week 2 data.

- [ ] **Step 6: Freeze 20 generation prompts**

Build `generation_eval_20.json` from the existing Week 4 safety/business prompt sources: 10 safety and 10 business/general tasks. Store source path, source ID, prompt, category and source file SHA-256; do not copy blind reviewer identity mappings.

- [ ] **Step 7: Commit Task 1**

```bash
git add deliverables/week7/source deliverables/week7/day34/source/data deliverables/week7/day34/source/results deliverables/week7/day35/source/data deliverables/week7/tests/test_datasets.py
git commit -m "feat(week7): freeze quantization evaluation data"
```

### Task 2: 模型血缘、量化配置和结果门禁

**Files:**
- Create: `deliverables/week7/source/week7_deployment/manifests.py`
- Create: `deliverables/week7/source/week7_deployment/acceptance.py`
- Create: `deliverables/week7/day34/configs/awq_config.json`
- Create: `deliverables/week7/day34/source/scripts/check_llamafactory_awq_compatibility.py`
- Create: `deliverables/week7/day34/source/scripts/preflight_week4_model.py`
- Create: `deliverables/week7/day34/source/scripts/quantize_awq.py`
- Create: `deliverables/week7/day35/configs/week4_dpo_gptq.yaml`
- Create: `deliverables/week7/day35/source/scripts/validate_quantized_model.py`
- Test: `deliverables/week7/tests/test_manifests_and_acceptance.py`

**Interfaces:**
- Produces: `validate_model_manifest(manifest: Mapping[str, object], expected: ModelIdentity) -> list[str]`.
- Produces: `memory_reduction(bf16_peak_mib: float, quantized_peak_mib: float) -> float`.
- Produces: `quantization_passes(receipt: QuantizationReceipt) -> bool` with finite-metric and `>= 0.30` gates.

- [ ] **Step 1: Write failing lineage and acceptance tests**

```python
def test_week4_manifest_must_match_frozen_identity():
    errors = validate_model_manifest(valid_week4_manifest(), WEEK4_DPO_IDENTITY)
    assert errors == []

def test_wrong_manifest_hash_is_rejected():
    manifest = valid_week4_manifest() | {"manifest_sha256": "0" * 64}
    assert "manifest_sha256" in validate_model_manifest(manifest, WEEK4_DPO_IDENTITY)

def test_memory_gate_requires_thirty_percent():
    assert memory_reduction(16000.0, 11200.0) == pytest.approx(0.30)
    assert quantization_passes(receipt(peak=11200.0, smoke=True, ppl=8.2))
    assert not quantization_passes(receipt(peak=11300.0, smoke=True, ppl=8.2))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest deliverables/week7/tests/test_manifests_and_acceptance.py -q`

Expected: FAIL because manifest and acceptance modules are missing.

- [ ] **Step 3: Implement strict schemas and finite metric checks**

```python
def memory_reduction(bf16_peak_mib: float, quantized_peak_mib: float) -> float:
    if not math.isfinite(bf16_peak_mib) or not math.isfinite(quantized_peak_mib) or bf16_peak_mib <= 0:
        raise ValueError("finite positive memory values required")
    return 1.0 - quantized_peak_mib / bf16_peak_mib
```

Reject absent files, non-64-character hashes, negative byte counts, non-finite PPL/NLL and receipts without a successful load/generation smoke.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest deliverables/week7/tests/test_manifests_and_acceptance.py -q`

Expected: all lineage, schema, finite-value and 30% boundary tests pass.

- [ ] **Step 5: Add auditable quantization entrypoints**

`quantize_awq.py` must read `awq_config.json`, verify the frozen dataset hash, call AutoAWQ with 4-bit/group-128/zero-point/GEMM, then write a manifest only after files exist. `week4_dpo_gptq.yaml` must set `export_quantization_bit: 4`, point to the same calibration JSONL and never set `quantization_method: awq`.

- [ ] **Step 6: Validate local configs without importing GPU libraries**

Run:

```bash
PYTHONPATH=deliverables/week7/source python3 deliverables/week7/day34/source/scripts/check_llamafactory_awq_compatibility.py --static-only
PYTHONPATH=deliverables/week7/source python3 deliverables/week7/day34/source/scripts/preflight_week4_model.py --validate-config-only
PYTHONPATH=deliverables/week7/source python3 deliverables/week7/day35/source/scripts/validate_quantized_model.py --validate-config-only
```

Expected: compatibility receipt identifies LLaMA-Factory 0.9.3 as GPTQ-export-only; both quantizer configs pass static validation.

- [ ] **Step 7: Commit Task 2**

```bash
git add deliverables/week7/source/week7_deployment deliverables/week7/day34 deliverables/week7/day35/configs deliverables/week7/day35/source/scripts deliverables/week7/tests/test_manifests_and_acceptance.py
git commit -m "feat(week7): add audited AWQ and GPTQ workflows"
```

### Task 3: BF16/AWQ/GPTQ 基准工具

**Files:**
- Create: `deliverables/week7/source/week7_deployment/benchmark.py`
- Create: `deliverables/week7/day35/configs/benchmark_config.json`
- Create: `deliverables/week7/day35/source/scripts/benchmark_model.py`
- Create: `deliverables/week7/day35/source/scripts/build_comparison.py`
- Create: `deliverables/week7/day35/source/results/README.md`
- Test: `deliverables/week7/tests/test_benchmark.py`

**Interfaces:**
- Produces: `summarize_timings(output_tokens: Sequence[int], durations: Sequence[float]) -> TimingSummary`.
- Produces: `aggregate_model_results(records: Sequence[Mapping[str, object]]) -> dict[str, object]`.
- Consumes: three raw result JSON files named `bf16.json`, `awq.json`, `gptq.json`.

- [ ] **Step 1: Write failing metric tests**

```python
def test_tokens_per_second_uses_total_tokens_over_total_time():
    summary = summarize_timings([100, 120], [2.0, 3.0])
    assert summary.tokens_per_second == pytest.approx(44.0)

def test_nonfinite_ppl_fails_comparison():
    rows = [result("bf16", ppl=8.0), result("awq", ppl=float("nan")), result("gptq", ppl=8.5)]
    with pytest.raises(ValueError, match="finite"):
        aggregate_model_results(rows)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest deliverables/week7/tests/test_benchmark.py -q`

Expected: FAIL because benchmark functions are absent.

- [ ] **Step 3: Implement deterministic aggregation**

Implement total-token throughput, median/P10/P90 latency, TTFT, peak memory, weight bytes, NLL/PPL and BF16-relative deltas. Verify all model names are exactly `bf16`, `awq`, `gptq` and all raw records reference the same protocol SHA-256.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest deliverables/week7/tests/test_benchmark.py -q`

Expected: all timing, percentile, protocol-match, finite-metric and acceptance tests pass.

- [ ] **Step 5: Add remote benchmark CLI**

The CLI records pre/post `nvidia-smi`, `torch.cuda.memory_allocated`, peak allocation, three warmups, ten measured generations, 256-window PPL and the full environment. It writes raw JSON atomically and never edits the comparison CSV directly.

- [ ] **Step 6: Add comparison builder**

`build_comparison.py` reads the three raw JSON files, verifies protocol hashes, emits `quantization_comparison.csv`, `quantization_comparison.json` and a machine-readable acceptance result. Missing raw results must return non-zero and leave Day 35 status incomplete.

- [ ] **Step 7: Commit Task 3**

```bash
git add deliverables/week7/day35 deliverables/week7/source/week7_deployment/benchmark.py deliverables/week7/tests/test_benchmark.py
git commit -m "feat(week7): add quantization benchmark protocol"
```

### Task 4: OpenAI 流式客户端、消息转换和路由

**Files:**
- Create: `deliverables/week7/source/week7_deployment/config.py`
- Create: `deliverables/week7/source/week7_deployment/messages.py`
- Create: `deliverables/week7/source/week7_deployment/openai_stream.py`
- Create: `deliverables/week7/source/week7_deployment/routing.py`
- Create: `deliverables/week7/source/week7_deployment/validation.py`
- Test: `deliverables/week7/tests/test_messages.py`
- Test: `deliverables/week7/tests/test_openai_stream.py`
- Test: `deliverables/week7/tests/test_routing.py`

**Interfaces:**
- Produces: `Settings.from_env(env: Mapping[str, str]) -> Settings`.
- Produces: `to_text_messages(history: Sequence[Turn], prompt: str) -> list[dict[str, object]]`.
- Produces: `to_vision_messages(history: Sequence[Turn], prompt: str, image: ImageInput) -> list[dict[str, object]]`.
- Produces: `iter_text_deltas(chunks: Iterable[object]) -> Iterator[str]`.
- Produces: `choose_backend(image: ImageInput | None) -> Literal["text", "vision"]`.

- [ ] **Step 1: Write failing message and route tests**

```python
def test_no_image_routes_to_text():
    assert choose_backend(None) == "text"

def test_image_routes_to_vision_and_uses_content_parts(valid_png):
    assert choose_backend(valid_png) == "vision"
    messages = to_vision_messages([], "读出标题", valid_png)
    assert [part["type"] for part in messages[-1]["content"]] == ["text", "image_url"]

def test_invalid_or_large_image_is_rejected(tmp_path):
    path = tmp_path / "large.png"
    path.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    with pytest.raises(InputValidationError):
        validate_image(path)
```

- [ ] **Step 2: Write failing stream tests**

```python
def test_stream_ignores_empty_deltas_and_yields_cumulative_text():
    chunks = [chunk(None), chunk("你"), chunk("好"), chunk("")]
    assert list(iter_text_deltas(chunks)) == ["你", "你好"]

def test_missing_required_endpoint_fails_fast():
    with pytest.raises(SettingsError, match="WEEK7_TEXT_BASE_URL"):
        Settings.from_env({})
```

- [ ] **Step 3: Run tests and verify RED**

Run: `python3 -m pytest deliverables/week7/tests/test_messages.py deliverables/week7/tests/test_openai_stream.py deliverables/week7/tests/test_routing.py -q`

Expected: FAIL because the shared modules do not exist.

- [ ] **Step 4: Implement minimal pure-Python core**

Use dataclasses and mappings so local tests do not require Gradio, OpenAI SDK, PIL or vLLM. Validate environment variables, role order, image extension/MIME/size, cumulative deltas and backend route. Keep SDK imports inside runtime client functions.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python3 -m pytest deliverables/week7/tests/test_messages.py deliverables/week7/tests/test_openai_stream.py deliverables/week7/tests/test_routing.py -q`

Expected: all environment, history, multimodal payload, image boundary and stream tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add deliverables/week7/source/week7_deployment deliverables/week7/tests/test_messages.py deliverables/week7/tests/test_openai_stream.py deliverables/week7/tests/test_routing.py
git commit -m "feat(week7): add streaming client and multimodal routing core"
```

### Task 5: vLLM 服务脚本和 Gradio 应用

**Files:**
- Create: `deliverables/week7/day36/source/scripts/start_text_server.sh`
- Create: `deliverables/week7/day36/source/scripts/start_vlm_server.sh`
- Create: `deliverables/week7/day36/source/scripts/stop_week7_server.sh`
- Create: `deliverables/week7/day36/source/scripts/check_service.py`
- Create: `deliverables/week7/day36/source/scripts/chat_client.py`
- Create: `deliverables/week7/day37/app.py`
- Create: `deliverables/week7/day37/requirements.txt`
- Create: `deliverables/week7/day38/source/scripts/multimodal_client.py`
- Test: `deliverables/week7/tests/test_service_scripts.py`
- Test: `deliverables/week7/tests/test_gradio_app.py`

**Interfaces:**
- Service scripts expose `week4-dpo-quantized` or `week5-qwen2-vl-base` at an environment-provided loopback URL.
- `app.stream_chat(message, history, image, temperature, top_p, max_tokens, client=None)` yields cumulative response strings; the optional client is dependency injection for tests.
- Clients consume the shared settings/messages/stream interfaces from Task 4.

- [ ] **Step 1: Write failing script contract tests**

```python
def test_text_script_has_served_model_name_and_no_embedded_secret():
    text = TEXT_SERVER.read_text()
    assert "--served-model-name week4-dpo-quantized" in text
    assert "WEEK7_TEXT_MODEL_PATH" in text
    assert "sk-" not in text

def test_vlm_script_uses_distinct_model_name():
    text = VLM_SERVER.read_text()
    assert "--served-model-name week5-qwen2-vl-base" in text
```

- [ ] **Step 2: Write failing Gradio behavior tests**

```python
def test_text_chat_forwards_generation_parameters(fake_client):
    output = list(stream_chat("你好", [], None, 0.4, 0.8, 128, client=fake_client))
    assert output[-1] == "你好！"
    assert fake_client.last_request["temperature"] == 0.4
    assert fake_client.last_request["top_p"] == 0.8
    assert fake_client.last_request["max_tokens"] == 128

def test_image_chat_selects_vision_model(fake_client, valid_png):
    list(stream_chat("图中是什么", [], valid_png, 0.2, 0.9, 256, client=fake_client))
    assert fake_client.last_request["model"] == "week5-qwen2-vl-base"
```

- [ ] **Step 3: Run tests and verify RED**

Run: `python3 -m pytest deliverables/week7/tests/test_service_scripts.py deliverables/week7/tests/test_gradio_app.py -q`

Expected: FAIL because service scripts and `app.py` are absent.

- [ ] **Step 4: Implement fail-fast service scripts**

Use `set -euo pipefail`, quote every path, validate required variables, refuse a conflicting Week 7 PID file, bind to `127.0.0.1`, set explicit max model length and write logs/PIDs under a configurable runtime directory. `stop_week7_server.sh` may terminate only the validated PID from that runtime directory.

- [ ] **Step 5: Implement Gradio application**

Build a single-image ChatInterface/Blocks app with history, temperature/top-p/max-token controls, route status, clear and stop controls. `stream_chat` must remain separately importable and dependency-injectable for tests; heavy Gradio initialization runs only under `main()`.

- [ ] **Step 6: Run tests and static shell checks**

Run:

```bash
python3 -m pytest deliverables/week7/tests/test_service_scripts.py deliverables/week7/tests/test_gradio_app.py -q
bash -n deliverables/week7/day36/source/scripts/start_text_server.sh
bash -n deliverables/week7/day36/source/scripts/start_vlm_server.sh
bash -n deliverables/week7/day36/source/scripts/stop_week7_server.sh
```

Expected: all application tests pass and all shell scripts parse cleanly.

- [ ] **Step 7: Commit Task 5**

```bash
git add deliverables/week7/day36 deliverables/week7/day37 deliverables/week7/day38/source/scripts deliverables/week7/tests/test_service_scripts.py deliverables/week7/tests/test_gradio_app.py
git commit -m "feat(week7): add vLLM services and streaming Gradio app"
```

### Task 6: Day 34–Day 39 文档、周报和状态门禁

**Files:**
- Create: `docs/week7_execution_plan.md`
- Create: `deliverables/week7/README.md`
- Create: `deliverables/week7/day34/README.md`
- Create: `deliverables/week7/day35/README.md`
- Create: `deliverables/week7/day36/README.md`
- Create: `deliverables/week7/day37/README.md`
- Create: `deliverables/week7/day38/README.md`
- Create: `deliverables/week7/day38/OPTIMIZATION_NOTES.md`
- Create: `deliverables/week7/day39/README.md`
- Create: `deliverables/week7/day39/REPORT.md`
- Create: `deliverables/week7/day39/LOCAL_DEPLOYMENT_GUIDE.md`
- Create: `deliverables/week7/day39/source/scripts/validate_week7.py`
- Create: `deliverables/week7/day39/source/results/week7_acceptance_matrix.csv`
- Test: `deliverables/week7/tests/test_documentation.py`

**Interfaces:**
- `validate_week7.py` consumes daily receipts and emits a final validation JSON without inventing missing metrics.
- Documentation links only to existing repository-relative paths and distinguishes planned, locally verified and remotely verified facts.

- [ ] **Step 1: Write failing documentation contract tests**

```python
def test_each_day_readme_maps_teacher_requirement():
    for day in range(34, 40):
        text = (WEEK7 / f"day{day}" / "README.md").read_text()
        assert "老师要求" in text
        assert "交付物" in text
        assert "验收" in text
        assert "当前状态" in text

def test_report_cannot_claim_pass_without_receipts():
    result = validate_week7(WEEK7, require_remote=True)
    assert result["status"] != "PASS"
    assert result["missing_remote_evidence"]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest deliverables/week7/tests/test_documentation.py -q`

Expected: FAIL because Week 7 documentation and validator do not exist.

- [ ] **Step 3: Write execution and daily documentation**

Map each teacher requirement, prerequisites, exact commands, expected outputs, evidence schema, failure handling and current status. Status is calculated from existing receipts; local scaffolding alone is not completion.

- [ ] **Step 4: Write deployment guide and evidence-bound report**

The guide contains environment isolation, model paths, `.env.example`, text/VLM start order, client commands, Gradio launch, switching procedure, troubleshooting and shutdown. The report includes only machine-readable values that exist; before remote execution it explicitly labels GPU result sections as pending rather than inserting zeros or estimates.

- [ ] **Step 5: Implement and run local-status validator**

Run:

```bash
PYTHONPATH=deliverables/week7/source python3 deliverables/week7/day39/source/scripts/validate_week7.py --allow-pending-remote
python3 -m pytest deliverables/week7/tests/test_documentation.py -q
```

Expected: local structure/config/code gates pass; remote-only items are listed as pending until Task 7 provides receipts.

- [ ] **Step 6: Commit Task 6**

```bash
git add docs/week7_execution_plan.md deliverables/week7
git commit -m "docs(week7): map deployment requirements and evidence gates"
```

### Task 7: AutoDL 正式量化、基准和端到端验收

**Files:**
- Populate: `deliverables/week7/day34/source/results/`
- Populate: `deliverables/week7/day35/source/results/`
- Populate: `deliverables/week7/day36/source/results/`
- Populate: `deliverables/week7/day37/source/results/`
- Populate: `deliverables/week7/day38/source/results/`
- Populate: `deliverables/week7/day38/source/demo/`
- Update: `deliverables/week7/day34/README.md`
- Update: `deliverables/week7/day35/README.md`
- Update: `deliverables/week7/day36/README.md`
- Update: `deliverables/week7/day37/README.md`
- Update: `deliverables/week7/day38/README.md`
- Update: `deliverables/week7/day39/REPORT.md`
- Update: `deliverables/week7/day39/source/results/week7_acceptance_matrix.csv`

**Interfaces:**
- Consumes: frozen data, quantizer configs, benchmark protocol, vLLM scripts and Gradio app from Tasks 1–6.
- Produces: real remote receipts, model manifests, metrics, service traces, screenshots/recording metadata and final acceptance status.

- [ ] **Step 1: Verify remote model and environments**

Run Week 4 model preflight first. Create isolated AWQ/GPTQ/serving environments, run import/version smokes, and save `environment.json` plus command receipts. If the merged model is absent, reconstruct it only from the frozen Week 3 base and Week 4 adapter lineage in the spec.

- [ ] **Step 2: Run LLaMA-Factory compatibility receipt and AWQ export**

Execute the non-static compatibility checker, then AutoAWQ export. Save stdout/stderr, exit status, wall time, GPU snapshot, calibration SHA, quantization config, output manifest and fixed-prompt generation smoke.

- [ ] **Step 3: Run GPTQ export**

Execute the validated LLaMA-Factory YAML using the same calibration SHA. Save the effective config, command, GPTQModel version, logs, exit status, output manifest and matching smoke.

- [ ] **Step 4: Benchmark BF16/AWQ/GPTQ**

Run the formal benchmark once per model under the same protocol. Build the comparison only after all three raw JSON files validate. Preserve actual memory reduction even if below 30%; do not rerun selectively to optimize the number.

- [ ] **Step 5: Validate text vLLM and client**

Start the selected passing quantized model, run health/model-list/synchronous/streaming requests, save sanitized request/response traces and server logs, then stop the validated PID.

- [ ] **Step 6: Validate Gradio text and VLM paths**

Run the Gradio text flow, record progressive chunks and UI screenshot. Stop text vLLM, start the Week 5 base VLM, upload two frozen Week 5 images, preserve multimodal request/response logs and UI screenshot.

- [ ] **Step 7: Record and validate the demo**

Record the 2–4 minute Day 38 demonstration, transcode to 720p H.264 and compute duration, byte size and SHA-256. Commit the MP4 only if it is at most 25 MiB; otherwise commit the recording manifest and document the controlled external delivery location.

- [ ] **Step 8: Rebuild final report and acceptance**

Run:

```bash
PYTHONPATH=deliverables/week7/source python3 deliverables/week7/day39/source/scripts/validate_week7.py --require-remote
python3 -m pytest deliverables/week7/tests -q
```

Expected: all teacher gates PASS. If any gate fails, preserve FAIL with the exact metric and stop before claiming Week 7 complete.

- [ ] **Step 9: Commit verified remote evidence**

```bash
git add deliverables/week7/day34 deliverables/week7/day35 deliverables/week7/day36 deliverables/week7/day37 deliverables/week7/day38 deliverables/week7/day39
git commit -m "feat(week7): collect quantized deployment evidence"
```

### Task 8: 教师提交投影和全局入口更新

**Files:**
- Create: `deliverables/week7/source/week7_deployment/submission.py`
- Create: `deliverables/week7/day39/source/scripts/build_submission.py`
- Create: `Submission/Week7/README.md`
- Create: `Submission/Week7/SHA256SUMS.txt`
- Create: `Submission/Week7/Day34_AWQ_Quantization/`
- Create: `Submission/Week7/Day35_GPTQ_and_Benchmark/`
- Create: `Submission/Week7/Day36_vLLM_Service/`
- Create: `Submission/Week7/Day37_Gradio_Application/`
- Create: `Submission/Week7/Day38_Multimodal_Optimization/`
- Create: `Submission/Week7/Day39_Weekly_Report_and_Deployment_Guide/`
- Modify: `README.md`
- Modify: `Submission/README.md`
- Modify: `Submission/SHA256SUMS.txt`
- Modify: `.gitignore`
- Test: `deliverables/week7/tests/test_submission_layout.py`
- Test: `deliverables/week7/tests/test_global_readmes.py`

**Interfaces:**
- Produces: `build_submission(repo_root: Path, mapping: Mapping[str, Sequence[str]]) -> list[SubmissionRecord]`.
- Each Day has a `Submission_Map.json` mapping repository sources to byte-identical teacher copies.
- Week 7 and global checksum files cover every eligible submitted file except themselves.

- [ ] **Step 1: Write failing teacher-layout tests**

```python
def test_week7_has_six_shallow_day_directories():
    assert {p.name for p in WEEK7_SUBMISSION.iterdir() if p.is_dir()} == {
        "Day34_AWQ_Quantization",
        "Day35_GPTQ_and_Benchmark",
        "Day36_vLLM_Service",
        "Day37_Gradio_Application",
        "Day38_Multimodal_Optimization",
        "Day39_Weekly_Report_and_Deployment_Guide",
    }

def test_submission_maps_are_byte_identical():
    for record in all_submission_records():
        assert sha256_file(record.source) == sha256_file(record.submitted)
```

- [ ] **Step 2: Write failing global README tests**

```python
def test_root_readme_no_longer_says_week6_not_started():
    text = ROOT_README.read_text()
    assert "第 6 周 Day 28–Day 33 的 Agent 智能体开发要求已经完成规划，当前尚未开始实验" not in text
    assert "docs/week7_execution_plan.md" in text

def test_submission_index_lists_week6_and_week7():
    text = SUBMISSION_README.read_text()
    assert "Week6/README.md" in text
    assert "Week7/README.md" in text
    assert "本次 Week 4 交付请直接提交" not in text
```

- [ ] **Step 3: Run tests and verify RED**

Run: `python3 -m pytest deliverables/week7/tests/test_submission_layout.py deliverables/week7/tests/test_global_readmes.py -q`

Expected: FAIL because Week 7 submission and global updates are missing.

- [ ] **Step 4: Implement deterministic teacher projection**

Copy only approved small artifacts, reject absolute/parent paths, model weight suffixes, credentials, caches and unverified result files. Generate per-day mappings and Week 7 SHA-256 after copying.

- [ ] **Step 5: Update global indexes accurately**

Root README must describe actual Week 6 metrics, Week 5 quality FAIL, and Week 7’s verified or pending status. Submission README lists Week 1–Week 7 and removes the obsolete Week 4-only instruction. Rebuild the global checksum over all Week submission files except the checksum file itself.

- [ ] **Step 6: Run layout, link, checksum and leakage verification**

Run:

```bash
python3 -m pytest deliverables/week7/tests -q
shasum -a 256 -c Submission/Week7/SHA256SUMS.txt
shasum -a 256 -c Submission/SHA256SUMS.txt
git diff --check
```

Expected: all tests pass, both checksum scopes validate, no whitespace errors, no large weights or secrets are tracked.

- [ ] **Step 7: Commit final Week 7 repository update**

```bash
git add .gitignore README.md Submission deliverables/week7 docs/week7_execution_plan.md
git commit -m "feat(week7): publish quantized deployment deliverables"
```

### Task 9: Final verification and handoff

**Files:**
- Verify: all files touched by Tasks 1–8

**Interfaces:**
- Produces: a fresh, evidence-backed completion summary with exact test/checksum counts and any remaining external blocker.

- [ ] **Step 1: Run full Week 7 test suite**

Run: `python3 -m pytest deliverables/week7/tests -q`

Expected: zero failures.

- [ ] **Step 2: Run historical submission regression**

Run: `python3 -m pytest deliverables/week6/tests/test_submission_layout.py -q`

Expected: Week 6 teacher package remains valid.

- [ ] **Step 3: Revalidate checksums and tracked-file safety**

Run:

```bash
shasum -a 256 -c Submission/Week7/SHA256SUMS.txt
shasum -a 256 -c Submission/SHA256SUMS.txt
git ls-files -z | xargs -0 shasum -a 256 >/dev/null
git diff --check
git status --short
```

Expected: both checksum manifests pass; all tracked files are readable; only explicitly preserved user changes remain outside Week 7 commits.

- [ ] **Step 4: Re-read teacher acceptance matrix**

Confirm each of the four teacher Week 7 acceptance requirements points to a real validated artifact. If AutoDL integration is unavailable or any result fails, report the exact incomplete items and do not use completion language.

- [ ] **Step 5: Prepare final handoff**

Report the implemented architecture, actual metrics, tests/checksums, commit IDs, file entrypoints, preserved user changes and the shortest next action if an external login or GPU run remains.
