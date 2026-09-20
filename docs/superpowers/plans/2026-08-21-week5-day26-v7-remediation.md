# Week5 Day26 V7 Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Day26 with audited labels, balanced multimodal tasks, isolated development/final evaluation, and a genuinely passing LoRA result before replacing the teacher submission.

**Architecture:** A versioned v7 pipeline converts manually reviewed image fact cards into exactly 200 unique training records and two sealed 20-case evaluation sets. Hash-bound manifests connect source bytes, facts, records, configs, adapters, and evaluation ledgers. Candidate selection uses development data only; the final packet is run once after all settings are frozen.

**Tech Stack:** Python 3, pytest, LLaMA-Factory 0.9.3, Qwen2-VL-7B-Instruct, PEFT LoRA, JSON/JSONL, SHA256

**Spec:** `docs/superpowers/specs/2026-08-21-week5-day26-v7-remediation-design.md`

---

## Task 1: Lock the V7 fact-card and dataset contracts

**Files:**

- Create: `deliverables/week5/day26/source/scripts/build_v7_dataset.py`
- Create: `deliverables/week5/day26/source/data/source_image_facts_v7.json`
- Create: `deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py`

- [ ] Write tests that reject missing provenance, missing SHA256, fewer than three observable facts, unsupported categories, ambiguous OCR/formula gold labels, missing boolean `claim_supported`, or a non-reviewed annotation.
- [ ] Write tests requiring 50 train facts, 10 dev facts, 10 final facts and category balance of 10/2/2 images.
- [ ] Write tests requiring exactly four train modes and two evaluation modes per image.
- [ ] Run `.venv/bin/python -m pytest deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py -q` and confirm the new tests fail because the v7 builder does not exist.
- [ ] Implement the smallest fact validator and deterministic record builder that satisfies the tests.
- [ ] Run the focused test file again and confirm it passes.

## Task 2: Acquire and freeze new source images

**Files:**

- Create: `deliverables/week5/day26/source/data/source_image_specs_v7.json`
- Create: `deliverables/week5/day26/source/data/source_image_manifest_v7.json`
- Create: `deliverables/week5/day26/source/data/source_image_manifest_v7.csv`
- Create: `deliverables/week5/day26/source/scripts/acquire_source_images_v7.py`
- Modify: `deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py`

- [ ] Add acquisition tests that require pinned source revisions, direct source URLs, license metadata, byte count, and SHA256.
- [ ] Add tests that reject every prior v6 dev/final image SHA256 and any overlap across v7 train/dev/final; old train bytes are allowed only after their labels are discarded and independently reviewed again.
- [ ] Run the focused tests and observe the acquisition/overlap tests fail.
- [ ] Implement deterministic download and manifest generation without overwriting existing non-matching bytes.
- [ ] Download candidate images only from the recorded official source URLs.
- [ ] Render or inspect every new image, exclude illegible or ambiguous evaluation candidates, and record the accepted byte hashes.
- [ ] Run the focused tests and confirm provenance and split-isolation checks pass.

## Task 3: Build reviewed V7 records and contamination gates

**Files:**

- Modify: `deliverables/week5/day26/source/data/source_image_facts_v7.json`
- Create: `deliverables/week5/day26/source/data/week5_vlm_train_v7.json`
- Create: `deliverables/week5/day26/source/data/week5_vlm_dev_v7_internal.json`
- Create: `deliverables/week5/day26/source/data/week5_vlm_final_v7_internal.json`
- Create: `deliverables/week5/day26/source/data/data_manifest_v7.json`
- Create: `deliverables/week5/day26/source/scripts/validate_v7_dataset.py`
- Modify: `deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py`

- [ ] Add tests for exactly 200 unique train records and 20 unique records in each evaluation split.
- [ ] Add tests requiring supported/unsupported verification balance within every category and split.
- [ ] Add tests rejecting exact normalized prompt overlap across train/dev/final, answer-label contradictions, image leakage, duplicate records, and copied weighting records.
- [ ] Add tests that recompute all image, facts, dataset, and manifest hashes rather than trusting recorded values.
- [ ] Run focused tests and confirm the new validation cases fail.
- [ ] Complete the manually reviewed fact cards using only visible evidence and explicit uncertainty notes.
- [ ] Generate v7 datasets and manifests deterministically.
- [ ] Implement the contamination and hash validator, then run it against the real v7 artifacts.
- [ ] Run all Day26 tests and confirm both legacy and v7 suites pass.

## Task 4: Freeze candidate training configurations and training lineage

**Files:**

- Create: `deliverables/week5/day26/source/configs/qwen2vl_lora_v7_a_lr1e5_e1.yaml`
- Create: `deliverables/week5/day26/source/configs/qwen2vl_lora_v7_b_lr2e5_e1.yaml`
- Create: `deliverables/week5/day26/source/configs/qwen2vl_lora_v7_c_lr1e5_e15.yaml`
- Create: `deliverables/week5/day26/source/scripts/prepare_v7_run.py`
- Modify: `deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py`

- [ ] Add tests proving the three configs vary only in the preregistered learning-rate/epoch dimensions and all reference the same 200-record training data.
- [ ] Add tests requiring frozen vision tower/projector, base revision, seed, precision, max pixels, LoRA targets, and output path.
- [ ] Add tests requiring the run manifest to hash the exact train dataset, fact cards, source manifest, config, base revision, and eventual `adapter_model.safetensors`.
- [ ] Run the focused tests and observe failures for the absent configs/preparer.
- [ ] Implement configs and pre-run/post-run manifest generation.
- [ ] Run focused and legacy tests to confirm the configuration and lineage gates pass.

## Task 5: Make evaluation ledgers immutable and add development selection

**Files:**

- Create: `deliverables/week5/day26/source/scripts/run_base_lora_eval_v7.py`
- Create: `deliverables/week5/day26/source/scripts/select_v7_candidate.py`
- Modify: `deliverables/week5/day26/source/tests/test_day26_v7_pipeline.py`

- [ ] Add tests requiring every evaluation ledger to bind the base revision, adapter weights SHA256, case SHA256, generation config SHA256, scoring config SHA256, stage, scale, and run ID.
- [ ] Add tests proving `--resume` rejects any changed fingerprint and never merges runs from different adapters, cases, stages, scales, or configs.
- [ ] Add tests for the development gate: mean gain `>= 0.35`, direct gain `>= 0`, wins `>= 12/20`, and hallucination non-increase.
- [ ] Add tests proving final cases cannot be loaded by the candidate selector.
- [ ] Run the focused tests and observe failures for the absent evaluation/selection implementation.
- [ ] Implement immutable evaluation ledgers and deterministic dev candidate selection.
- [ ] Run focused and legacy tests and confirm they pass.

## Task 6: Train and select candidates on AutoDL

**Files:**

- Create: `deliverables/week5/day26/source/results/v7/training_summary_<candidate>.json` for each executed candidate
- Create: `deliverables/week5/day26/source/results/v7/base_lora_paired_dev_<candidate>_<scale>.json` for each dev evaluation
- Create: `deliverables/week5/day26/source/results/v7/candidate_selection.json`
- Create: `deliverables/week5/day26/source/results/v7/selected_adapter_sha256.txt`

- [ ] Synchronize only the v7 inputs, scripts, tests, and fixed base revision to AutoDL.
- [ ] Run dataset validation and a one-step smoke train before any full candidate.
- [ ] Train candidate A and verify the actual adapter weights hash plus frozen/trainable parameter evidence.
- [ ] Evaluate A at scales 0.75, 0.85, and 1.0 on dev only.
- [ ] Train B only if A fails the development gate or the preregistered comparison is needed; train C only if A is demonstrably underfit.
- [ ] Select exactly one candidate using the frozen development gate and write the selection evidence.
- [ ] If no candidate passes, return to Tasks 2–4 without opening the final packet.

## Task 7: Run the sealed final evaluation once

**Files:**

- Create: `deliverables/week5/day26/source/results/v7/final_run_spec.json`
- Create: `deliverables/week5/day26/source/results/v7/base_lora_paired_final.json`
- Create: `deliverables/week5/day26/source/results/v7/blind_scores_final.json`
- Create: `deliverables/week5/day26/source/results/v7/revealed_scores_final.json`
- Create: `deliverables/week5/day26/source/results/v7/final_eval_summary.json`

- [ ] Freeze the selected adapter SHA256, LoRA scale, final case hash, generation config, scoring rubric, blind seed, and pass thresholds into `final_run_spec.json`.
- [ ] Verify the final image/case hashes were never present in a prior run ledger.
- [ ] Generate Base and LoRA outputs exactly once under the frozen specification.
- [ ] Score the anonymized outputs before revealing the A/B mapping.
- [ ] Recompute mean gain, direct-mode gain, wins, hallucination rate, category means, and all pass gates from raw rows.
- [ ] Accept `PASS` only when mean gain is `>= 0.50`, wins are `>= 12/20`, hallucination does not increase, and every engineering gate passes.
- [ ] If final fails, archive it honestly and do not modify the candidate using these final outputs.

## Task 8: Replace the teacher submission only after PASS

**Files:**

- Modify: `Submission/Week5/Day26_VLM_LoRA/README.md`
- Modify: `Submission/Week5/Day26_VLM_LoRA/Model_Training_Summary.md`
- Modify: `Submission/Week5/Day26_VLM_LoRA/Base_LoRA_Comparison.md`
- Modify: `Submission/Week5/Day26_VLM_LoRA/Hallucination_Evaluation.md`
- Modify: `Submission/Week5/Day26_VLM_LoRA/SHA256SUMS`
- Modify: `Submission/Week5/SHA256SUMS`
- Modify: `Submission/SHA256SUMS`

- [ ] Confirm the v7 final summary says `PASS` and recomputes from raw evidence before copying any result into Submission.
- [ ] Replace failed-v6 teacher-facing claims with v7 metrics, explicit limitations, source/permission records, and reproducibility instructions.
- [ ] Include the actual adapter archive or a teacher-accessible artifact reference with its SHA256; exclude AutoDL shutdown proof, credentials, private blind mapping, and unrelated repository files.
- [ ] Run all Day26 tests, the v7 real-artifact validator, Submission checksum verification, archive inspection, credential scanning, and `git diff --check`.
- [ ] Shut down every AutoDL instance used by this round and verify each reports powered off.
- [ ] Stage only Day26 v7 and intended Submission changes, preserving the user's unrelated Word/PDF edits.
- [ ] Commit on `codex/week5-day26`, push to `origin/codex/week5-day26`, and report the commit hash plus verification evidence.
