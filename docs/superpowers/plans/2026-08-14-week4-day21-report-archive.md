# Week 4 Day 21 Report and Final DPO Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate audited DPO Rewards curves, write the Week 4 DPO preference-alignment report, and archive the retained final DPO model without placing weights in Git.

**Architecture:** Freeze the unpromoted remediation result first, then transform the immutable Day 19 metric CSV into a canonical Day 21 table and plot. A cross-day validator builds report/archive facts from Day 17–Day 20 evidence, and only validated teacher-facing artifacts are projected into `Submission`.

**Tech Stack:** Python 3.10, pytest, csv/json, matplotlib, Pillow, Markdown, SHA-256, Git.

## Global Constraints

- The retained final model is Day 20 `sft_dpo_v1_merged`; no remediation candidate is promoted.
- Rewards are read from the frozen Day 19 240-step CSV; report values are not manually invented.
- Teacher safety refusal uses score >= 1; enhanced safe completion uses score == 2.
- The report states business quality did not improve on the five frozen prompts.
- Model weights, checkpoints, private mappings, credentials, symlinks, caches and shutdown evidence are excluded from `Submission`.
- No GPU is required unless existing evidence is unexpectedly incomplete.

---

### Task 1: Freeze remediation outcome

**Files:**
- Create: `deliverables/week4/day20/remediation/source/results/dev/*`
- Create: `deliverables/week4/day20/remediation/source/results/candidate_selection.json`
- Modify: `deliverables/week4/day20/remediation/README.md`

- [ ] Verify the downloaded remote bundle and replace its invalid self-referential checksum entry with a manifest that excludes itself.
- [ ] Copy the four complete 35-prompt response files into the canonical development result directory.
- [ ] Record 120 safety judgments and 20 blinded business judgments with reasons.
- [ ] Combine training trends and development summaries, run the frozen selector, and require `selected_candidate=null`.
- [ ] Document every failed gate and commit the remediation result independently.

### Task 2: Day 21 metric normalization and plot via TDD

**Files:**
- Create: `deliverables/week4/day21/source/tests/test_plot_dpo_metrics.py`
- Create: `deliverables/week4/day21/source/scripts/plot_dpo_metrics.py`
- Create: `deliverables/week4/day21/source/results/dpo_training_metrics.csv`
- Create: `deliverables/week4/day21/source/results/dpo_rewards_curves.png`
- Create: `deliverables/week4/day21/source/results/dpo_plot_manifest.json`

- [ ] Write failing tests for 240 unique steps, finite metrics, exact margin identity, source hash, moving averages and PNG creation.
- [ ] Run the focused test and confirm failure because the plot module is absent.
- [ ] Implement normalization and the fixed 2×2 plot using the non-interactive Agg backend.
- [ ] Run tests, render the PNG, inspect it visually and record dimensions and hashes.

### Task 3: Cross-day validator and final model archive via TDD

**Files:**
- Create: `deliverables/week4/day21/source/tests/test_validate_week4.py`
- Create: `deliverables/week4/day21/source/scripts/validate_week4.py`
- Create: `deliverables/week4/day21/source/results/final_dpo_model_path.json`
- Create: `deliverables/week4/day21/source/results/week4_acceptance_matrix.md`
- Create: `deliverables/week4/day21/source/results/week4_validation.json`

- [ ] Write failing tests for Day 17 categories, Day 18 counts, Day 19 steps, Day 20 denominators/results and final model lineage.
- [ ] Run focused tests and confirm the validator is missing.
- [ ] Implement evidence readers and deterministic archive/acceptance outputs.
- [ ] Require all cross-day checks to pass before report generation.

### Task 4: Week 4 report and engineering archive

**Files:**
- Create: `deliverables/week4/day21/REPORT.md`
- Create: `deliverables/week4/day21/README.md`
- Create: `deliverables/week4/day21/source/results/week4_file_manifest.csv`
- Create: `deliverables/week4/day21/source/results/week4_files.sha256`
- Modify: `docs/week4_execution_plan.md`
- Modify: `deliverables/week4/README.md`
- Modify: `README.md`

- [ ] Write the report from validated facts using the approved section order.
- [ ] Insert the Rewards image and preserve the three separate safety/business conclusions.
- [ ] Explain the failed explicit-reference attempt and unpromoted remediation without overstating quality.
- [ ] Generate a small-file manifest and validate every internal link and hash.

### Task 5: Teacher projection

**Files:**
- Create: `Submission/Week4/Day21_Weekly_Report_and_Final_Model_Archive/README.md`
- Create: `Submission/Week4/Day21_Weekly_Report_and_Final_Model_Archive/Week4_DPO_Preference_Alignment_Report.md`
- Create: `Submission/Week4/Day21_Weekly_Report_and_Final_Model_Archive/DPO_Rewards_Curves.png`
- Create: `Submission/Week4/Day21_Weekly_Report_and_Final_Model_Archive/DPO_Training_Metrics.csv`
- Create: `Submission/Week4/Day21_Weekly_Report_and_Final_Model_Archive/Week4_Acceptance_Matrix.md`
- Create: `Submission/Week4/Day21_Weekly_Report_and_Final_Model_Archive/Final_DPO_Model_Archive.json`
- Modify: `Submission/Week4/README.md`
- Modify: `Submission/README.md`
- Modify: `Submission/SHA256SUMS.txt`

- [ ] Project only validated teacher artifacts from the engineering archive.
- [ ] Update Week 4 and root submission indexes from actual final facts.
- [ ] Rebuild the complete Submission SHA-256 manifest excluding itself.
- [ ] Verify no forbidden artifacts, large files or symlinks exist.

### Task 6: Final verification and Git delivery

- [ ] Run Day 21 tests, all Week 4 tests and all locally executable repository tests.
- [ ] Parse every JSON, JSONL and CSV; compile Python; check shell syntax where applicable.
- [ ] Run the Week 4 validator, Submission validator, checksum verification, link check, secret/large-file scan and `git diff --check`.
- [ ] Commit engineering results and teacher projection separately.
- [ ] Push `codex/week4-day20-remediation` and verify local/remote commit equality.

