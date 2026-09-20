# Day 6 Data Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect deterministic 2k/2k/1k subsets from the three approved Chinese instruction datasets, convert the same 5,000 samples to Alpaca and ShareGPT, and produce an auditable archive manifest.

**Architecture:** Three small command-line programs separate collection, pure format conversion, and read-only validation. Collection pins Hugging Face revisions and writes immutable raw subsets; conversion writes model-visible data separately from provenance; validation recomputes counts, schemas, and hashes without trusting the producer.

**Tech Stack:** Python 3.11, standard library, `huggingface_hub`, `datasets`, `pyarrow`, `pytest`.

## Global Constraints

- Dataset counts must be exactly 2,000 Alpaca-GPT4-zh, 2,000 COIG-PC, and 1,000 ShareGPT-zh.
- Random seed is exactly `42`; sampling is based on stable SHA-256 priority and upstream row index.
- Dataset revisions are copied verbatim from the approved design.
- Day 6 must not perform HTML cleaning, control-character cleaning, token truncation, or deduplication.
- Hugging Face credentials must never be written into source files, logs, manifests, shell history, or Git.
- Existing user changes in the dirty worktree must be preserved.
- Do not create a Git commit or push without explicit user confirmation.

---

### Task 1: Pure format mapping with TDD

**Files:**
- Create: `deliverables/week2/day6/source/tests/test_day6_pipeline.py`
- Create: `deliverables/week2/day6/source/scripts/convert_formats.py`

**Interfaces:**
- Produces: `alpaca_to_alpaca(record) -> dict`, `alpaca_to_sharegpt(record) -> dict`, `sharegpt_to_alpaca(record) -> dict`, `normalize_sharegpt(record) -> dict`, and a CLI that reads the three raw JSONL files.

- [ ] **Step 1: Write failing tests**

  Tests use literal fixtures to assert: empty and non-empty Alpaca input mapping; multi-turn ShareGPT preservation; multi-turn ShareGPT-to-Alpaca history serialization; rejection of malformed role order; and absence of provenance fields from model-visible outputs.

- [ ] **Step 2: Verify RED**

  Run:

  ```bash
  /Users/yifanren/anaconda3/bin/python -m pytest \
    deliverables/week2/day6/source/tests/test_day6_pipeline.py -v
  ```

  Expected: collection fails because `convert_formats.py` does not yet exist.

- [ ] **Step 3: Implement minimal converter**

  Add pure mapping functions, JSONL readers/writers, stable `sample_id` generation, provenance output, and CLI arguments `--raw-dir`, `--formatted-dir`, and `--provenance-file`.

- [ ] **Step 4: Verify GREEN**

  Run the same pytest command. Expected: mapping tests pass.

---

### Task 2: Deterministic collection with TDD

**Files:**
- Modify: `deliverables/week2/day6/source/tests/test_day6_pipeline.py`
- Create: `deliverables/week2/day6/source/scripts/collect_day6.py`

**Interfaces:**
- Produces: `stable_sample(records, sample_count, seed, source_name) -> list`, source-specific iterators, three raw JSONL subsets, and preliminary manifest metadata.
- Consumes: fixed dataset IDs, revisions, filenames, and target counts from constants in the collector.

- [ ] **Step 1: Add failing tests**

  Use literal indexed fixtures to verify exact sample count, stable order, repeatability, seed sensitivity, insufficient-data failure, preservation of original fields, and `_provenance.source_index`.

- [ ] **Step 2: Verify RED**

  Run pytest and confirm failures are caused by missing collection behavior.

- [ ] **Step 3: Implement minimal collector**

  Use `huggingface_hub` for fixed-revision file retrieval, `datasets`/`pyarrow` or JSON parsing for local source files, a bounded heap for SHA-256 sampling, and JSONL output. Accept `--cache-dir`, `--raw-dir`, `--manifest-json`, and `--seed 42`.

- [ ] **Step 4: Verify GREEN**

  Run pytest. Expected: collection and conversion unit tests pass without network access.

---

### Task 3: Independent validation and manifest generation with TDD

**Files:**
- Modify: `deliverables/week2/day6/source/tests/test_day6_pipeline.py`
- Create: `deliverables/week2/day6/source/scripts/validate_day6.py`

**Interfaces:**
- Produces: `validate_delivery(root) -> dict`, nonzero CLI exit on any violation, `day6_validation.json`, `raw_data_manifest.csv`, `raw_data_manifest.md`, and `day6_files.sha256`.
- Consumes: raw JSONL, formatted JSONL, provenance JSONL, and collector metadata.

- [ ] **Step 1: Add failing tests**

  Create temporary delivery fixtures and assert that the validator catches wrong counts, malformed Alpaca fields, invalid ShareGPT role order, duplicate sample IDs, missing manifest fields, and wrong file SHA-256.

- [ ] **Step 2: Verify RED**

  Run pytest and confirm the new validation tests fail because the validator is missing.

- [ ] **Step 3: Implement minimal validator**

  Parse every JSONL line, enforce exact schemas and counts, recompute SHA-256 and file sizes, verify provenance alignment, generate machine-readable validation results, and return exit code 1 on failure.

- [ ] **Step 4: Verify GREEN**

  Run pytest. Expected: all Day 6 tests pass.

---

### Task 4: Authenticate and collect the real fixed-revision data

**Files:**
- Create: `deliverables/week2/day6/source/data/raw/*.jsonl`
- Create: `deliverables/week2/day6/source/results/day6_collection.txt`
- Create: collector metadata under `deliverables/week2/day6/source/manifests/`

**Interfaces:**
- Consumes: an authenticated Hugging Face session for gated `BAAI/COIG-PC`.
- Produces: immutable 2k/2k/1k source subsets and source metadata.

- [ ] **Step 1: Complete user-controlled authentication**

  The user logs in manually in the visible Hugging Face page and accepts the COIG-PC conditions. No password or token is copied into the conversation.

- [ ] **Step 2: Verify access without downloading the full corpus**

  Query fixed-revision repository metadata and confirm the selected `Top200PerTask` file exists and is accessible.

- [ ] **Step 3: Run collection**

  ```bash
  /Users/yifanren/anaconda3/bin/python \
    deliverables/week2/day6/source/scripts/collect_day6.py \
    --raw-dir deliverables/week2/day6/source/data/raw \
    --cache-dir .cache/day6-huggingface \
    --manifest-json deliverables/week2/day6/source/manifests/collection_metadata.json \
    --seed 42
  ```

  Capture stdout and stderr in `day6_collection.txt` with the real exit code.

- [ ] **Step 4: Independently count raw subsets**

  Parse all three files and confirm exact counts 2,000, 2,000, and 1,000.

---

### Task 5: Convert, validate, and document

**Files:**
- Create: `deliverables/week2/day6/source/data/interim/day6_provenance.jsonl`
- Create: `deliverables/week2/day6/source/data/formatted/week2_5k_alpaca.jsonl`
- Create: `deliverables/week2/day6/source/data/formatted/week2_5k_sharegpt.jsonl`
- Create: `deliverables/week2/day6/source/manifests/raw_data_manifest.csv`
- Create: `deliverables/week2/day6/source/manifests/raw_data_manifest.md`
- Create: `deliverables/week2/day6/source/results/day6_validation.json`
- Create: `deliverables/week2/day6/source/results/day6_files.sha256`
- Create: `deliverables/week2/day6/README.md`
- Modify: `README.md`
- Modify: `docs/week2_execution_plan.md`

**Interfaces:**
- Consumes: all real raw subsets and collector metadata.
- Produces: final Day 6 delivery and updated project navigation/status.

- [ ] **Step 1: Run conversion**

  ```bash
  /Users/yifanren/anaconda3/bin/python \
    deliverables/week2/day6/source/scripts/convert_formats.py \
    --raw-dir deliverables/week2/day6/source/data/raw \
    --formatted-dir deliverables/week2/day6/source/data/formatted \
    --provenance-file deliverables/week2/day6/source/data/interim/day6_provenance.jsonl
  ```

- [ ] **Step 2: Run independent validation**

  ```bash
  /Users/yifanren/anaconda3/bin/python \
    deliverables/week2/day6/source/scripts/validate_day6.py \
    --root deliverables/week2/day6 \
    --output deliverables/week2/day6/source/results/day6_validation.json
  ```

  Expected: exit code 0 with all checks true.

- [ ] **Step 3: Write actual-result documentation**

  Document real revisions, counts, file sizes, SHA-256 values, mapping decisions, COIG-PC terms, reproducibility commands, and the explicit Day 6/Day 7 boundary. Update README progress only after validation succeeds.

- [ ] **Step 4: Run final verification**

  Re-run all tests, the standalone validator, JSONL counts, a secrets scan, link/path checks, and `git diff --check`. Inspect `git status` to confirm no unrelated file was modified.

## Plan Self-Review

- Spec coverage: all six Day 6 teacher requirements map to Tasks 2, 4, and 5.
- Placeholder scan: no `TBD`, `TODO`, “implement later”, or undefined behavior remains.
- Type consistency: the raw collector, converter, provenance, validator, and manifest paths agree across all tasks.
- Execution choice: the user explicitly requested inline execution in this conversation; use `superpowers:executing-plans`, without subagents.
