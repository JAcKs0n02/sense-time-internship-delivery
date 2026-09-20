# Week5 Day27 Remediation and Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete one bounded Day26 remediation cycle, evaluate it without reusing an exposed final set, and produce the evidence-backed Day27 Week5 report and teacher submission.

**Architecture:** Reuse the audited v7 train images and the previously unopened v7 final set as a new development set, build a deterministic balanced corrective training export, and preregister one main continuation candidate plus at most one fallback candidate. Only a development-pass candidate may trigger acquisition of a new final set and a one-time final blind evaluation; Day27 then reports PASS or FAIL exactly as observed.

**Tech Stack:** Python 3, pytest, JSON/JSONL/CSV, LLaMA-Factory 0.9.3, Qwen2-VL-7B-Instruct, LoRA, AutoDL RTX 3090, Git.

**Spec:** `deliverables/week5/day27/README.md`; `docs/superpowers/specs/2026-08-21-week5-day26-v7-repair-addendum.md`

## Global Constraints

- Teacher requirements remain: complete unfinished experiments, write the Week5 multimodal report, and substantiate obvious VLM improvement.
- Use at most one main candidate and one fallback candidate; stop if both fail development gates.
- Never train on dev/final images, prompts, answers, or model outputs.
- The unopened v7 final set may become v8 dev; it is no longer a final set after that declaration.
- A new v8 final set is acquired and annotated only after a candidate passes v8 dev.
- Freeze ViT and projection layers; train only language-model LoRA parameters.
- Development gate: mean gain `>= 0.35/5`, direct gain `>= 0`, hallucination rate not worse, opportunity win rate `>= 0.50`.
- Report raw strict wins even though it is not the executable gate when Base ceiling makes it impossible.
- Final is run exactly once after adapter, scale, generation config, scoring config, and hashes are frozen.
- Teacher submission excludes model weights, checkpoints, caches, credentials, private A/B keys, blind seeds, and AutoDL shutdown evidence.
- Shut down AutoDL immediately after remote artifacts are synchronized and verified; do not poll periodically while training.

---

### Task 1: Opportunity-aware v8 scoring contract

**Files:**
- Modify: `deliverables/week5/day26/source/scripts/score_base_lora_v7.py`
- Modify: `deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py`
- Create: `deliverables/week5/day26/configs/evaluation_scoring_v8_dev.json`
- Create: `deliverables/week5/day26/configs/evaluation_scoring_v8_final.json`

**Interfaces:**
- Consumes: revealed 20-row Base/LoRA score records.
- Produces: `score_comparison(...)->dict` with `opportunity_count`, `opportunity_wins`, `opportunity_win_rate`, and a configurable opportunity gate.

- [ ] **Step 1: Write the failing scoring tests**

Add assertions proving that opportunities are rows where Base weighted score is below 5, that wins count only strict LoRA improvements inside that subset, and that `opportunity_win_rate_min` replaces the impossible raw-win gate while raw `lora_wins` remains reported.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `../../../../.venv/bin/python -m pytest deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py -k opportunity -q`

Expected: FAIL because the opportunity fields do not exist.

- [ ] **Step 3: Implement the minimal scoring change**

Compute opportunity indices from `base_scores[index] < 5.0`, add the three metrics, and add a check only when `opportunity_win_rate_min` exists. Preserve the legacy `lora_wins_min` behavior for v6/v7 configs.

- [ ] **Step 4: Add frozen v8 configs**

Dev thresholds: mean gain `0.35`, direct gain `0.0`, opportunity win rate `0.50`, hallucination not worse. Final thresholds use the same values so the criterion is not tightened after seeing dev.

- [ ] **Step 5: Run all Day26 tests and commit**

Run: `../../../../.venv/bin/python -m pytest deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py deliverables/week5/day26/source/tests/test_day26_pipeline.py -q`

Commit: `test(day26): add opportunity-aware v8 quality gate`

### Task 2: Deterministic v8 corrective train and dev exports

**Files:**
- Create: `deliverables/week5/day26/source/scripts/build_v8_corrective_dataset.py`
- Create: `deliverables/week5/day26/source/data/week5_vlm_train_v8_corrective.json`
- Create: `deliverables/week5/day26/source/data/week5_vlm_dev_v8.json`
- Create: `deliverables/week5/day26/source/data/week5_vlm_dev_v8_internal.json`
- Create: `deliverables/week5/day26/source/results/v8/data_manifest_v8.json`
- Modify: `deliverables/week5/day26/configs/dataset_info.json`
- Modify: `deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py`

**Interfaces:**
- Consumes: aligned v7 public/internal 200-row train exports and unopened v7 final public/internal 20-row exports.
- Produces: a balanced 400-row train-only LLaMA-Factory export and a 20-row v8 dev set relabeled from the unopened v7 final set.

- [ ] **Step 1: Write failing builder tests**

Require 400 rows, 80 per category, 200 original prompts plus 200 deterministic paraphrases, one image token, no dev/final media in train, and no prompt collision with v8 dev. Require v8 dev to contain exactly 20 records, 4 per category, 10 direct and 10 verification, and source images formerly labeled v7 final.

- [ ] **Step 2: Run the builder tests and verify RED**

Run: `../../../../.venv/bin/python -m pytest deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py -k v8_dataset -q`

Expected: FAIL because the builder is absent.

- [ ] **Step 3: Implement the builder**

Copy all 200 audited rows once, then add one category/mode-specific paraphrase for every row without changing the image or target answer. Relabel the unopened 20-row v7 final records to v8 dev, preserve their source hashes, and reject execution if any v7-final inference artifact is found in the repository.

- [ ] **Step 4: Generate exports and manifest**

The manifest records parent hashes, row/category/mode counts, prompt collision sets, image SHA intersections, output hashes, seed, and the declaration that the former v7 final is now v8 dev.

- [ ] **Step 5: Run tests and commit**

Run the full Day26 test pair and commit as `feat(day26): build bounded v8 corrective dataset`.

### Task 3: Preregister two bounded GPU candidates

**Files:**
- Create: `deliverables/week5/day26/configs/qwen2vl_lora_v8_e_balanced_continue.yaml`
- Create: `deliverables/week5/day26/configs/qwen2vl_lora_v8_f_balanced_restart.yaml`
- Create: `deliverables/week5/day26/source/results/v8/preregistration.json`
- Modify: `deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py`

**Interfaces:**
- Candidate E consumes the selected v7-B adapter and v8 corrective data; LR `2e-6`, `0.5` epoch, scale `1.0`.
- Candidate F is only allowed after E fails and trains from the frozen Base on the same data; LR `1e-5`, `1.0` epoch, scale `1.0`.

- [ ] **Step 1: Write failing config and preregistration tests**

Assert Base revision, dataset names, eval dataset, freeze flags, LoRA targets, seeds, output directories, candidate order, maximum candidate count, thresholds, and stop rule.

- [ ] **Step 2: Verify RED, add configs and preregistration, verify GREEN**

Run the focused config tests before and after implementation, then the complete Day26 suite.

- [ ] **Step 3: Commit**

Commit as `test(day26): preregister bounded v8 remediation`.

### Task 4: Execute remote training and v8 development blind evaluation

**Files:**
- Create after synchronization: `deliverables/week5/day26/source/results/v8/candidate_e_training_summary.json`
- Create conditionally: `deliverables/week5/day26/source/results/v8/candidate_f_training_summary.json`
- Create: `deliverables/week5/day26/source/results/v8/dev_blind_scores.json`
- Create: `deliverables/week5/day26/source/results/v8/dev_revealed_scores.json`
- Create: `deliverables/week5/day26/source/results/v8/dev_score_summary.json`
- Create: `deliverables/week5/day26/source/results/v8/candidate_selection.json`

**Interfaces:**
- Consumes: v8 datasets/configs, Qwen2-VL base manifest, and the exact v7-B parent adapter hash.
- Produces: one selected dev-pass candidate or a frozen `NO_PASSING_CANDIDATE` result.

- [ ] **Step 1: Start AutoDL and run preflight plus one-step smoke**

Verify GPU, environment, dataset hashes, base model hashes, parent adapter hash, trainable parameter names, and frozen visual modules before formal training.

- [ ] **Step 2: Run candidate E without periodic polling**

Launch formal training and check once near its expected completion. Save logs, metrics, adapter config/weight SHA, and loading smoke.

- [ ] **Step 3: Run Base/LoRA v8 dev inference and blind scoring**

Generate 40 raw responses with immutable run-spec binding, score A/B before downloading the private key, reveal, and compute all four gates.

- [ ] **Step 4: Conditionally run candidate F**

Only if E fails. Repeat the same preflight, training, blind evaluation, and gate calculation. Do not create a third candidate.

- [ ] **Step 5: Freeze selection and synchronize artifacts**

Select the highest mean-gain passing candidate; otherwise record no passing candidate and skip final evaluation.

### Task 5: Conditional new final set and one-time final evaluation

**Files:**
- Create only after dev PASS: `deliverables/week5/day26/source/data/source_image_specs_v8_final.json`
- Create only after dev PASS: `deliverables/week5/day26/source/data/source_image_facts_v8_final.json`
- Create only after dev PASS: `deliverables/week5/day26/source/data/week5_vlm_final_v8_internal.json`
- Create only after dev PASS: final evaluation artifacts under `deliverables/week5/day26/source/results/v8/`

**Interfaces:**
- Consumes: frozen passing adapter/scale/config and 10 newly sourced, double-checked images.
- Produces: one 20-case final comparison, or no final artifacts when dev fails.

- [ ] **Step 1: Acquire and visually verify two real-source images per category**

Use pinned repository revisions or dated source pages, record author/license/URLs/SHA, and inspect every image before annotation.

- [ ] **Step 2: Write and validate 20 final cases**

Use distinct prompt wording, double-check OCR/formula transcription, verify zero image/prompt/answer overlap, and freeze hashes before inference.

- [ ] **Step 3: Run final once, blind score, reveal, and stop**

Do not retrain or alter scale after the final result. Record PASS only when all four frozen gates pass.

### Task 6: Day27 report, model archive, teacher projection, and shutdown

**Files:**
- Modify: `deliverables/week5/day27/README.md`
- Create: `deliverables/week5/day27/REPORT.md`
- Create: `deliverables/week5/day27/source/scripts/validate_day27.py`
- Create: `deliverables/week5/day27/source/tests/test_week5_acceptance.py`
- Create: `deliverables/week5/day27/source/results/report_input_manifest.json`
- Create: `deliverables/week5/day27/source/results/week5_acceptance_matrix.csv`
- Create: `deliverables/week5/day27/source/results/final_vlm_model_manifest.json`
- Create: `deliverables/week5/day27/source/results/day27_validation.json`
- Create/Modify: `Submission/Week5/Day27_Weekly_Report/*`, `Submission/Week5/README.md`, `Submission/Week5/SHA256SUMS.txt`
- Modify: `deliverables/week5/README.md`, `docs/week5_execution_plan.md`, `README.md`

**Interfaces:**
- Consumes: validated Day22-Day26 evidence and the frozen v8 terminal outcome.
- Produces: a traceable Week5 report, acceptance matrix, model manifest, validator result, and minimal teacher submission.

- [ ] **Step 1: Write failing Day27 validator tests**

Test exact Day22-Day25 PASS evidence, Day26 terminal status, report numeric traceability, acceptance rows, model manifest hashes, submission privacy boundaries, checksum completeness, and no claim of PASS when the quality gate failed.

- [ ] **Step 2: Verify RED and implement report asset generation/validation**

Build the input manifest and acceptance matrix from source files, then render report tables from those frozen values.

- [ ] **Step 3: Write the report and final model archive**

Include teacher requirements, methods, all daily results, failed candidates, limitations, terminal quality conclusion, Base+adapter lineage, and reproduction entry points.

- [ ] **Step 4: Create minimal teacher projection**

Include the report, acceptance matrix, compact model archive, and direct links to existing Day22-Day26 teacher deliverables. Exclude all private and operational artifacts.

- [ ] **Step 5: Run complete validation and checksum verification**

Run Day22-Day27 validators, all Week5 tests, JSON/CSV parsing, recursive sensitive-string scan, link checks, and SHA256 recomputation.

- [ ] **Step 6: Commit, push, and verify AutoDL is off**

Commit only Week5/Day27 work, push `codex/week5-day27`, and verify every AutoDL instance is stopped. Do not place shutdown evidence in Submission.
