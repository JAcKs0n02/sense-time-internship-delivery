# Week6 Agent Quality V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the official Week6 training and evaluation pipeline with self-contained data, semantic scoring, controlled model selection, and a teacher-ready final submission.

**Architecture:** Deterministic builders produce versioned train/dev/test artifacts and manifests. A pure scoring module evaluates raw ReAct traces, while an optional policy layer evaluates deployed Agent reliability without changing raw-model metrics. GPU work runs only after all local contracts pass.

**Tech Stack:** Python 3.10/3.12, pytest, LangChain 1.3.13, LangGraph 1.2.9, LLaMA-Factory 0.9.3, Transformers 4.52.4, PEFT 0.15.2, Torch 2.1.2+cu121, QLoRA.

**Spec:** `docs/superpowers/specs/2026-08-30-week6-agent-quality-v2-design.md`

## Global Constraints

- Preserve all currently committed result artifacts as historical evidence until the new formal run is collected.
- Teacher-facing prose presents the formal main experiment and prompt ablations, without process-remediation wording.
- Every new behavior begins with a failing test and a verified red-green cycle.
- The final test split is not used for checkpoint, prompt, or hyperparameter selection.
- Raw-model and guarded-system metrics are reported separately.
- All AutoDL work uses the canonical 320 instance and the instance is stopped after artifact synchronization.

---

### Task 1: Versioned self-contained data contracts

**Files:**
- Modify: `deliverables/week6/day31/source/tests/test_tool_sft_dataset.py`
- Modify: `deliverables/week6/day31/source/scripts/build_tool_sft_dataset.py`
- Modify: `deliverables/week6/day31/source/scripts/validate_tool_sft_dataset.py`
- Modify: `deliverables/week6/day31/configs/dataset_info.json`
- Create: `deliverables/week6/day31/source/data/tool_sft_train_core_100.json`
- Create: `deliverables/week6/day31/source/data/tool_sft_train_expanded.json`
- Create: `deliverables/week6/day31/source/data/tool_sft_dev_v2.json`
- Create: `deliverables/week6/day31/source/data/tool_sft_test_v2.json`

**Interfaces:**
- Produces: deterministic `build_dataset_bundle()` output containing core, expanded, dev, test, and manifest payloads.
- Produces: validator checks `self_contained_inputs`, `natural_prompt_families`, `split_semantic_disjoint`, and `final_test_minimum_100`.

- [ ] Add tests asserting every direct tool input is present in the user Prompt and every derived calculator input is grounded in the preceding Observation.
- [ ] Run the targeted dataset tests and confirm they fail because the current recovery prompts hide payloads.
- [ ] Add tests for 100 core rows, expanded training rows, separate dev/test splits, prompt-family diversity, and at least 100 final cases.
- [ ] Run the targeted tests and confirm the new bundle API and files are absent.
- [ ] Implement deterministic diversified builders and manifest records.
- [ ] Run dataset tests and the validator until all checks pass.

### Task 2: Semantic scoring contract

**Files:**
- Modify: `deliverables/week6/day31/source/tests/test_day31_formal_harness.py`
- Modify: `deliverables/week6/day31/source/scripts/day31_harness.py`
- Modify: `deliverables/week6/day31/source/scripts/evaluate_frozen_cases.py`
- Modify: `deliverables/week6/day31/source/scripts/build_pre_post_eval_summary.py`

**Interfaces:**
- Produces: `score_case(row, result)` checks schema, exact arguments, observation integrity, final grounding, semantic correctness, safe termination, and strict success.
- Produces: aggregate metrics with explicit applicable counts and split identity hashes.

- [ ] Add a failing test where the trace contains a real but wrong calculator argument and the final answer contradicts that Observation.
- [ ] Add a failing test where the final prose is nonempty but omits required budget fields.
- [ ] Add a failing test for truncated tool-call JSON being classified as format failure.
- [ ] Implement structured semantic predicates and renamed observation-integrity metrics.
- [ ] Run all Day31 harness tests and confirm the new cases pass without weakening existing integrity checks.

### Task 3: Agent policy layer and parser recovery

**Files:**
- Modify: `deliverables/week6/day29/source/tests/test_agent_factory.py`
- Modify: `deliverables/week6/day30/source/tests/test_multistep_trace.py`
- Modify: `deliverables/week6/source/week6_agent/model_adapter.py`
- Modify: `deliverables/week6/source/week6_agent/agent_factory.py`
- Modify: `deliverables/week6/source/week6_agent/day30_agent.py`
- Create: `deliverables/week6/source/week6_agent/policy.py`

**Interfaces:**
- Produces: `validate_tool_transition(prompt, trace, proposed_call)` returning an allow/retry/terminate decision.
- Produces: raw mode and guarded mode in the compiled Agent result.

- [ ] Add failing tests for zero-shipping calculator skipping, Observation-derived amount mismatch, source-code drift, error-state continuation, and malformed tool JSON.
- [ ] Implement the smallest general policy functions required by each failing case.
- [ ] Integrate guarded mode without changing raw evaluation behavior.
- [ ] Run Day29–Day30 tests and inspect full trace serialization.

### Task 4: Formal configuration and local preflight

**Files:**
- Modify: `deliverables/week6/day31/configs/week6_tool_sft_lora.yaml`
- Create: `deliverables/week6/day31/configs/week6_tool_sft_lora_v2.yaml`
- Modify: `deliverables/week6/day31/source/scripts/formal_preflight.py`
- Modify: `deliverables/week6/day31/source/tests/test_day31_formal_harness.py`

**Interfaces:**
- Consumes: versioned training/dev files from Task 1.
- Produces: one environment/version contract for both trainer and evaluator.

- [ ] Add failing config tests for fixed package versions, learning rate 5e-5, two epochs, dropout 0.05, evaluation cadence, and checkpoint retention.
- [ ] Implement the v2 config and preflight version checks.
- [ ] Run formal preflight tests locally with fake model fixtures.

### Task 5: Day32 single-variable prompt experiments

**Files:**
- Modify: `deliverables/week6/day32/source/tests/test_day32_agent.py`
- Modify: `deliverables/week6/day32/source/tests/test_error_analysis.py`
- Modify: `deliverables/week6/source/week6_agent/day32_agent.py`
- Modify: `deliverables/week6/day32/source/scripts/evaluate_agent.py`
- Modify: `deliverables/week6/day32/source/scripts/analyze_failures.py`
- Create: `deliverables/week6/day32/source/configs/system_prompt_main.txt`
- Create: `deliverables/week6/day32/source/configs/system_prompt_ablation_input_fidelity.txt`

**Interfaces:**
- Produces: prompt manifest that proves exactly one rule change per ablation.
- Consumes: development split only during prompt selection.

- [ ] Add failing tests that reject multi-rule prompt comparisons and final-test use during prompt selection.
- [ ] Implement prompt manifest diff metadata and dev-only evaluation gates.
- [ ] Run Day32 analysis and validation tests.

### Task 6: Local release gate before GPU work

**Files:**
- Modify: `deliverables/week6/tests/test_submission_layout.py`
- Modify: `deliverables/week6/day33/source/tests/test_validate_day33.py`

**Interfaces:**
- Produces: a local readiness receipt containing dataset hashes, test counts, config hashes, and explicit GPU-pending fields.

- [ ] Add failing tests for missing v2 artifacts and ambiguous raw/guarded metric names.
- [ ] Generate all deterministic datasets and manifests.
- [ ] Run the complete Week6 pytest suite.
- [ ] Run every local Day31/Day32 CLI validator and record the commands and exit codes.

### Task 7: Canonical AutoDL model selection and formal training

**Files:**
- Collect: `deliverables/week6/day31/source/results/v2_checkpoint_selection.json`
- Collect: `deliverables/week6/day31/source/results/v2_training_log.txt`
- Collect: `deliverables/week6/day31/source/results/v2_training_summary.json`
- Collect: `deliverables/week6/day31/source/results/v2_adapter_manifest.json`
- Collect: `deliverables/week6/day31/source/results/v2_dev_eval.json`
- Collect: `deliverables/week6/day31/source/results/v2_final_test_eval.json`

**Interfaces:**
- Consumes: canonical 320 model archive, versioned data/config, and local readiness receipt.
- Produces: selected checkpoint and one frozen final-test result.

- [ ] Start the canonical 320 instance and verify disk capacity, model hash, adapter hash, repository revision, and exact package versions.
- [ ] Evaluate existing checkpoint-25, checkpoint-39, and final adapter on the v2 development split with identical generation parameters.
- [ ] If no existing checkpoint meets the acceptance gate, run the formal v2 QLoRA configuration once and evaluate every saved checkpoint on the development split.
- [ ] Freeze the selected checkpoint, prompt, generation config, tools, and hashes.
- [ ] Run the final test split once in raw mode and guarded mode.
- [ ] Synchronize logs, manifests, and result JSON to the local worktree.
- [ ] Stop the AutoDL instance and verify its stopped state.

### Task 8: Teacher submission projection

**Files:**
- Modify: `deliverables/week6/day31/README.md`
- Modify: `deliverables/week6/day32/README.md`
- Modify: `deliverables/week6/day32/source/results/AGENT_ERROR_ANALYSIS.md`
- Modify: `deliverables/week6/day33/REPORT.md`
- Modify: `deliverables/week6/day33/source/scripts/build_day33.py`
- Modify: `deliverables/week6/day33/source/scripts/build_week6_submission.py`
- Modify: `Submission/Week6/**`

**Interfaces:**
- Consumes: verified v2 model and evaluation artifacts.
- Produces: shallow teacher package, submission maps, acceptance matrix, final archive, and root checksum manifest.

- [ ] Update teacher-facing text with the formal main experiment, selected prompt, raw/guarded metrics, limitations, and reproducibility commands.
- [ ] Build Day33 machine-readable artifacts from verified inputs.
- [ ] Rebuild `Submission/Week6` from formal sources.
- [ ] Scan teacher-facing prose for prohibited process-remediation wording and credentials/operations details.
- [ ] Verify every Submission map and SHA-256 entry.

### Task 9: Final verification and integration

**Files:**
- Verify all modified and generated files.

**Interfaces:**
- Produces: a clean, merged `main` branch matching `origin/main` after push.

- [ ] Run the complete Week6 pytest suite in the feature worktree.
- [ ] Run dataset, Day31, Day32, Day33, submission-layout, and checksum validators.
- [ ] Review `git diff --check`, repository status, artifact sizes, and prohibited-file scan.
- [ ] Commit the verified feature branch.
- [ ] Merge into `main`, rerun the complete verification suite, and push `main`.
