# Week 6 Agent 智能体 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现三个受限工具、基于 Week 4 DPO 模型的 ReAct Agent、100 条工具调用 SFT、冻结评测与错误分析，并形成 Week 6 教师交付。

**Architecture:** 三个工具和 Agent 工厂位于共享的 `deliverables/week6/source/week6_agent/` 包，逐日目录只保存当天配置、数据、结果、测试和说明。所有轨迹使用统一 schema，Day 31 与 Day 32 复用 Day 28–Day 30 的工具定义，Day 33 只从验证后的机器可读结果生成报告。

**Tech Stack:** Python 3.10、LangChain、LangGraph、Transformers/vLLM 兼容本地模型接口、Pydantic、pytest、LLaMA-Factory、PEFT、JSON/JSONL、pandas。

**Spec:** `docs/superpowers/specs/2026-08-24-week6-agent-deliverables-design.md`

## Global Constraints

- Day 29/31 的起点必须是 Week 4 `reward_corrective_40step_merged`，manifest SHA-256 为 `aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c`。
- 老师要求的 `create_react_agent` 必须通过真实导入和调用证据验证；记录兼容 LangChain/LangGraph 精确版本。
- Calculator 只能解释 AST 白名单节点，禁止 `eval`/`exec`/shell。
- KnowledgeRetrieval 只能读取固定本地 JSON，不联网、不接受任意文件路径。
- CodeExecutor 只做 AST 解析和风险检查，永不执行代码。
- Agent 必须设置最大步骤、超时、重复调用检测和结构化错误。
- Day 31 正好 100 条训练数据；额外评测集不计入 100 条且不得用于训练或调参。
- Day 32 的 baseline/optimized 必须使用同一冻结评测、模型和生成参数。
- 规划值不等于事实；只有原始日志、manifest 和验证器通过后才能修改状态。
- 模型权重、checkpoint、缓存和凭据不进入 Git 或教师提交。

---

### Task 1: Day 28 两个基础工具

**Files:**
- Create: `deliverables/week6/source/week6_agent/tools/calculator.py`
- Create: `deliverables/week6/source/week6_agent/tools/knowledge_retrieval.py`
- Create: `deliverables/week6/source/week6_agent/limits.py`
- Create: `deliverables/week6/day28/source/data/knowledge_base.json`
- Create: `deliverables/week6/day28/source/tests/test_calculator.py`
- Create: `deliverables/week6/day28/source/tests/test_knowledge_retrieval.py`
- Create: `deliverables/week6/day28/source/scripts/validate_day28.py`

**Interfaces:**
- Produces: `CalculatorTool.run(expression: str) -> ToolResult`
- Produces: `KnowledgeRetrievalTool.search(query: str) -> ToolResult`
- Produces: JSON-serializable `ToolResult(status, data, error_code)` used by all later days.

- [ ] **Step 1: Write failing Calculator tests**

```python
def test_calculator_accepts_arithmetic(calculator):
    assert calculator.run("123 * 456").data["value"] == 56088

def test_calculator_rejects_calls(calculator):
    result = calculator.run("__import__('os').system('id')")
    assert result.status == "rejected"
```

- [ ] **Step 2: Run Calculator tests and confirm failure**

Run: `pytest deliverables/week6/day28/source/tests/test_calculator.py -q`
Expected: FAIL because the package does not exist.

- [ ] **Step 3: Implement AST-whitelist arithmetic**

Implement constants, unary plus/minus and explicit binary operators with limits from the spec. Return structured errors for syntax, forbidden nodes, division by zero, non-finite values and limit breaches.

- [ ] **Step 4: Write and pass KnowledgeRetrieval tests**

```python
def test_exact_alias_returns_price_and_shipping(tool):
    result = tool.search("星云键盘")
    assert result.data["price_cny"] >= 0
    assert result.data["shipping_cny"] >= 0

def test_unknown_product_is_not_fabricated(tool):
    assert tool.search("不存在的产品").status == "not_found"
```

- [ ] **Step 5: Validate data and tools**

Run: `python deliverables/week6/day28/source/scripts/validate_day28.py`
Expected: two tool classes import, all knowledge entries pass schema/SHA checks, and unsafe Calculator cases are rejected.

- [ ] **Step 6: Commit Day 28**

```bash
git add deliverables/week6/source deliverables/week6/day28
git commit -m "feat(week6): implement safe calculator and knowledge tools"
```

### Task 2: Day 29 ReAct Agent

**Files:**
- Create: `deliverables/week6/source/week6_agent/model_adapter.py`
- Create: `deliverables/week6/source/week6_agent/agent_factory.py`
- Create: `deliverables/week6/source/week6_agent/trace_schema.py`
- Create: `deliverables/week6/day29/source/scripts/run_single_turn.py`
- Create: `deliverables/week6/day29/source/scripts/validate_day29.py`
- Create: `deliverables/week6/day29/source/tests/test_agent_factory.py`

**Interfaces:**
- Consumes: Day 28 `CalculatorTool` and `KnowledgeRetrievalTool`.
- Produces: `build_react_agent(model, tools, limits) -> CompiledAgent`.
- Produces: `run_agent_case(agent, case) -> AgentTrace`.

- [ ] **Step 1: Freeze runtime and model preflight**

Record Python/LangChain/LangGraph versions, prove `from langgraph.prebuilt import create_react_agent`, verify the Week 4 model manifest and run `local_files_only=True` load smoke. Missing DPO weights must stop the task.

- [ ] **Step 2: Write failing factory tests**

```python
def test_factory_binds_exact_two_tools(fake_model):
    agent = build_react_agent(fake_model, basic_tools(), AgentLimits(max_steps=4))
    assert agent.tool_names == {"calculator", "knowledge_retrieval"}
```

- [ ] **Step 3: Implement model adapter and Agent factory**

Bind only the two Day 28 tools, version the System Prompt, cap steps/time, and emit the trace schema. Keep the model server on loopback and never log credentials.

- [ ] **Step 4: Run the required single-turn test**

Run the frozen prompt `计算 123 * 456` once after smoke. Preserve raw model message, parsed action, `{"expression":"123 * 456"}`, observation `56088`, final answer, latency, versions and hashes.

- [ ] **Step 5: Validate and commit Day 29**

Run: `python deliverables/week6/day29/source/scripts/validate_day29.py`
Expected: DPO lineage PASS, exactly one successful Calculator call, correct final value and bounded termination.

### Task 3: Day 30 CodeExecutor and multi-step reasoning

**Files:**
- Create: `deliverables/week6/source/week6_agent/tools/code_executor.py`
- Create: `deliverables/week6/day30/source/scripts/run_multistep.py`
- Create: `deliverables/week6/day30/source/scripts/validate_day30.py`
- Create: `deliverables/week6/day30/source/tests/test_code_executor.py`
- Create: `deliverables/week6/day30/source/tests/test_multistep_trace.py`

**Interfaces:**
- Produces: `CodeExecutor.inspect(source: str) -> ToolResult`.
- Consumes: all three tools and the Day 29 trace runner.

- [ ] **Step 1: Prove CodeExecutor does not execute**

```python
def test_code_executor_reports_syntax_without_execution(tmp_path, tool):
    marker = tmp_path / "must_not_exist"
    result = tool.inspect(f"open({str(marker)!r}, 'w').write('x')")
    assert result.status == "rejected"
    assert not marker.exists()
```

- [ ] **Step 2: Implement syntax/risk inspection and pass tests**

Return syntax location, node counts and forbidden-node labels. Do not call the compiled object and do not expose file/network/process access.

- [ ] **Step 3: Freeze complex cases before running**

Include the teacher product-price example, at least two variants, one CodeExecutor routing case and error-recovery cases. Each case defines expected tool sequence, key arguments and completion criteria.

- [ ] **Step 4: Run and validate the three-step trace**

The required trace must contain Knowledge retrieval, Calculator total, and final budget comparison. Verify no repeated identical action/input and no step exceeds the cap.

- [ ] **Step 5: Commit Day 30**

```bash
git add deliverables/week6/source/week6_agent/tools/code_executor.py deliverables/week6/day30
git commit -m "feat(week6): add AST checker and bounded multistep agent"
```

### Task 4: Day 31 tool-call SFT

**Files:**
- Create: `deliverables/week6/day31/configs/dataset_info.json`
- Create: `deliverables/week6/day31/configs/week6_tool_sft_lora.yaml`
- Create: `deliverables/week6/day31/source/scripts/build_tool_sft_dataset.py`
- Create: `deliverables/week6/day31/source/scripts/validate_tool_sft_dataset.py`
- Create: `deliverables/week6/day31/source/tests/test_tool_sft_dataset.py`

**Interfaces:**
- Consumes: exact tool names/schema from Tasks 1–3.
- Produces: 100-row ShareGPT tool-call training set, separate frozen evaluation set, training summary and adapter manifest.

- [ ] **Step 1: Write failing dataset contract tests**

```python
def test_train_has_exactly_100_unique_rows(train_rows):
    assert len(train_rows) == 100
    assert len({row["id"] for row in train_rows}) == 100

def test_function_calls_have_matching_observations(train_rows):
    assert all(valid_tool_sequence(row["conversations"]) for row in train_rows)
```

- [ ] **Step 2: Build deterministic train/eval data**

Use the exact 20/20/20/25/15 distribution from the spec. Store concise routing rationale in the audit form and export official `function_call`/`observation` messages plus tool descriptions for training.

- [ ] **Step 3: Run contamination, schema and tokenizer gates**

Require unique IDs, valid JSON arguments, registered tool names, reproducible observations, zero train/eval prompt overlap, finite token lengths and successful LLaMA-Factory parser smoke.

- [ ] **Step 4: Run one-step GPU smoke, then formal LoRA**

Start from the verified Week 4 DPO merged model. Preserve effective YAML, command, exit status, per-step loss, trainable parameter names, adapter files, sizes and SHA-256.

- [ ] **Step 5: Commit small evidence only**

Keep adapter/checkpoint/model weights on AutoDL. Commit dataset, config, logs/metrics, manifest and validation; then confirm AutoDL shutdown without placing shutdown evidence in `Submission/Week6`.

### Task 5: Day 32 error analysis and optimization

**Files:**
- Create: `deliverables/week6/day32/source/data/frozen_agent_eval.json`
- Create: `deliverables/week6/day32/source/scripts/evaluate_agent.py`
- Create: `deliverables/week6/day32/source/scripts/analyze_failures.py`
- Create: `deliverables/week6/day32/source/results/AGENT_ERROR_ANALYSIS.md`
- Create: `deliverables/week6/day32/source/tests/test_error_taxonomy.py`

**Interfaces:**
- Consumes: pre-SFT model, Day 31 SFT adapter, fixed tool code and one frozen evaluation set.
- Produces: row-level baseline/optimized traces, failure labels and before/after summary.

- [ ] **Step 1: Freeze evaluation and taxonomy before inference**

Each case records expected tools, arguments, minimum/maximum steps and success predicate. Freeze hashes before any baseline result is viewed.

- [ ] **Step 2: Evaluate baseline and SFT model identically**

Use identical prompts, tools, limits and generation parameters. Score tool choice, argument extraction, completion, loops and hallucinated observations.

- [ ] **Step 3: Change only the prompt/tool-description layer**

Version the optimized System Prompt or descriptions, explain the targeted failure modes, then rerun the same frozen cases without selecting only successful outputs.

- [ ] **Step 4: Generate and validate the report**

Report all failure categories, counts, representative trace IDs, root causes, changes and deltas. A non-improving metric remains visible and is not renamed PASS.

### Task 6: Day 33 report and teacher projection

**Files:**
- Create: `deliverables/week6/day33/REPORT.md`
- Create: `deliverables/week6/day33/source/results/week6_acceptance_matrix.csv`
- Create: `deliverables/week6/day33/source/scripts/validate_day33.py`
- Create: `Submission/Week6/README.md`
- Create: `Submission/Week6/SHA256SUMS.txt`

**Interfaces:**
- Consumes: verified Day 28–Day 32 evidence only.
- Produces: teacher-facing Week 6 package and four-row acceptance matrix.

- [ ] **Step 1: Run all daily validators and freeze report inputs**

Stop if a required source file, count, hash, model lineage or semantic gate fails. Record actual failures instead of substituting plan values.

- [ ] **Step 2: Write the Week 6 report**

Cover environment, three tools, single-turn Agent, three-step task, 100-row SFT, pre/post evaluation, error optimization, security boundaries, limitations and model archive.

- [ ] **Step 3: Build the four-row teacher acceptance matrix**

Rows are: three tools callable; at least three-step complex task; deep error report; weekly report submitted. Each row includes status, actual metric and evidence path.

- [ ] **Step 4: Project only teacher-required artifacts**

Exclude model weights, checkpoints, caches, credentials, private mappings and AutoDL shutdown evidence. Generate SHA-256 over every submitted file.

- [ ] **Step 5: Final validation and commit**

```bash
python deliverables/week6/day33/source/scripts/validate_day33.py
cd Submission/Week6 && shasum -a 256 -c SHA256SUMS.txt
git add deliverables/week6 Submission/Week6 README.md Submission/README.md
git commit -m "feat(week6): complete agent development report"
```
