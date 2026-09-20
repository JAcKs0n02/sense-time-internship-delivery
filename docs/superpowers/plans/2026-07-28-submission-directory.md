# Unified Submission Directory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one portable, English-named `Submission/` directory containing the existing curated Week 1 submission and only the teacher-required Day 6 and Day 7 Week 2 artifacts.

**Architecture:** Move the existing root `Week1/` directory into `Submission/Week1/` and rename its Chinese path components in English. Copy a minimal, teacher-facing subset from the canonical `deliverables/week2/` archives into `Submission/Week2/`, then add concise entry documents and a submission-relative SHA-256 manifest.

**Tech Stack:** Git file moves, POSIX filesystem operations for binary/data copies, Markdown, JSON/JSONL validation with Python standard library, `shasum`.

## Global Constraints

- Every path component under `Submission/` must use ASCII English names, digits, underscores, hyphens or dots.
- Chinese text may remain inside Markdown, logs, notebooks and datasets.
- Do not create a compressed archive.
- Do not use symlinks.
- Keep `deliverables/week2/day6/` and `deliverables/week2/day7/` unchanged.
- Week 2 includes only the teacher-required core artifacts plus minimal numeric validation evidence.
- Do not include `__pycache__`, `.pytest_cache`, temporary downloads or process logs.
- Hash paths in `Submission/SHA256SUMS.txt` are relative to `Submission/`.

---

### Task 1: Move and normalize the Week 1 submission

**Files:**

- Move: `Week1/` → `Submission/Week1/`
- Rename: `Submission/Week1/提交说明.md` → `Submission/Week1/README.md`
- Rename: `Submission/Week1/Day1_环境初始化/` → `Submission/Week1/Day1_Environment_Setup/`
- Rename: `Submission/Week1/Day2_模型下载与原生推理/` → `Submission/Week1/Day2_Model_Download_and_Inference/`
- Rename: `Submission/Week1/Day3_架构分析/` → `Submission/Week1/Day3_Architecture_Analysis/`
- Rename: `Submission/Week1/Day4_Tokenizer实验/` → `Submission/Week1/Day4_Tokenizer_Experiments/`
- Rename: `Submission/Week1/Day5_LLaMA-Factory与周报/` → `Submission/Week1/Day5_LLaMA_Factory_and_Weekly_Report/`
- Rename: `Qwen2.5架构分析报告.md` → `Qwen2.5_Architecture_Report.md`
- Rename: `第1周总结报告.md` → `Week1_Report.md`
- Remove: the two tracked `__pycache__/` directories and their `.pyc` files.
- Modify: `Submission/Week1/README.md`

**Interfaces:**

- Consumes: the existing curated `Week1/` tree.
- Produces: one cache-free, fully English-named Week 1 submission tree.

- [ ] **Step 1: Inventory exact tracked Week 1 paths**

Run:

```bash
git ls-tree -r --name-only HEAD Week1
```

Expected: all files match the design inventory; only two `__pycache__` files are non-submission cache artifacts.

- [ ] **Step 2: Move Week 1 into the unified directory**

Run exact `git mv` operations for the directory and each Chinese path component.

Expected: `Week1/` no longer exists at repository root and
`Submission/Week1/` contains all non-cache Week 1 artifacts.

- [ ] **Step 3: Remove cache artifacts**

Delete only:

```text
Submission/Week1/Day2_Model_Download_and_Inference/__pycache__/inference.cpython-311.pyc
Submission/Week1/Day3_Architecture_Analysis/__pycache__/analyze_params.cpython-311.pyc
```

Expected: neither `__pycache__` directory remains.

- [ ] **Step 4: Update Week 1 README links**

Replace every old Chinese relative path with its renamed English path and
rename the two report links.

Expected: every relative link in `Submission/Week1/README.md` resolves.

- [ ] **Step 5: Verify naming**

Run a path scan that fails when any path component under
`Submission/Week1/` contains a non-ASCII character.

Expected: zero violations.

### Task 2: Create the minimal Week 2 Day 6 submission

**Files:**

- Create: `Submission/Week2/Day6_Data_Collection_and_Formatting/README.md`
- Copy:
  - `Formatted_Alpaca_5K.jsonl`
  - `Formatted_ShareGPT_5K.jsonl`
  - `Data_Provenance.jsonl`
  - `Raw_Data_Manifest.md`
  - `Raw_Data_Manifest.csv`
  - `Collection_Metadata.json`

**Interfaces:**

- Consumes: canonical Day 6 files under `deliverables/week2/day6/`.
- Produces: the formatted data and archive manifest explicitly needed to
  demonstrate Day 6.

- [ ] **Step 1: Verify Day 6 source hashes before copying**

Run:

```bash
(cd deliverables/week2/day6 && \
  shasum -a 256 -c source/results/day6_files.sha256)
```

Expected: all Day 6 canonical files report `OK`.

- [ ] **Step 2: Copy the six approved Day 6 files**

Use direct, explicit source and target paths. Do not copy raw subsets, tests,
scripts, caches or process logs.

Expected: exactly six data/manifest files plus `README.md` exist under the
Day 6 submission directory.

- [ ] **Step 3: Write the Day 6 README**

Document:

- 2,000 Alpaca-GPT4-zh records;
- 2,000 COIG-PC records;
- 1,000 ShareGPT-zh records;
- two equivalent 5,000-row output formats;
- the purpose of the provenance and manifest files;
- the complete audit location at `deliverables/week2/day6/`.

- [ ] **Step 4: Validate Day 6 submission data**

Parse all JSONL rows and assert:

```text
Formatted_Alpaca_5K.jsonl   = 5000
Formatted_ShareGPT_5K.jsonl = 5000
Data_Provenance.jsonl       = 5000
```

Also assert the manifest source counts are `2000`, `2000`, `1000`.

### Task 3: Create the minimal Week 2 Day 7 submission

**Files:**

- Create: `Submission/Week2/Day7_Cleaning_Pipeline/README.md`
- Copy:
  - `clean_pipeline.py`
  - `Cleaned_Alpaca_4999.jsonl`
  - `Cleaned_ShareGPT_4999.jsonl`
  - `Length_Distribution.png`
  - `Cleaning_Counts.png`
  - `Cleaning_Stats.json`
  - `Cleaning_Stats.csv`
  - `Duplicate_Pairs.csv`
  - `SimHash_Calibration.csv`
  - `Day7_Validation.json`

**Interfaces:**

- Consumes: canonical Day 7 files under `deliverables/week2/day7/`.
- Produces: the standalone pipeline, cleaned dataset, requested figures and
  minimum deduplication/validation evidence.

- [ ] **Step 1: Verify Day 7 source hashes before copying**

Run:

```bash
shasum -a 256 -c deliverables/week2/day7/source/results/day7_files.sha256
```

Expected: all Day 7 canonical files report `OK`.

- [ ] **Step 2: Copy the ten approved Day 7 files**

Use direct, explicit source and target paths. Do not copy per-record audit,
cleaned provenance, tests, caches, determinism logs or process logs.

Expected: exactly ten artifacts plus `README.md` exist under the Day 7
submission directory.

- [ ] **Step 3: Write the Day 7 README**

Map the teacher requirements to exact files and record:

```text
input records              = 5000
final records              = 4999
HTML-modified records      = 72
control-modified records   = 27
truncated records          = 276
exact duplicates removed   = 1
fuzzy duplicates removed   = 0
maximum final token length = 2048
```

Explain that SimHash candidate generation was executed and that no fuzzy
candidate passed the precision-preserving Jaccard confirmation gates.

- [ ] **Step 4: Validate Day 7 submission data**

Parse both JSONL files, assert both contain 4,999 valid rows, read
`Day7_Validation.json`, and assert:

```json
{"valid": true, "final_count": 4999, "max_final_tokens": 2048}
```

### Task 4: Create unified entry documents and integrity manifest

**Files:**

- Create: `Submission/README.md`
- Create: `Submission/Week2/README.md`
- Create: `Submission/SHA256SUMS.txt`
- Modify: repository `README.md`

**Interfaces:**

- Consumes: the completed Week 1 and Week 2 submission trees.
- Produces: one teacher-facing navigation entry and one complete integrity
  manifest.

- [ ] **Step 1: Write `Submission/Week2/README.md`**

Create a compact Day 6/Day 7 requirement-to-file table and link to both day
READMEs.

- [ ] **Step 2: Write `Submission/README.md`**

Link Week 1 and Week 2, identify the AutoDL execution environment, and explain
that `deliverables/` retains complete engineering evidence.

- [ ] **Step 3: Update repository README**

Add `Submission/README.md` as the first teacher-facing entry and change any
statement that still treats root `Week1/` as the submission path.

- [ ] **Step 4: Generate the relative hash inventory**

Compute SHA-256 for every regular file under `Submission/` except
`SHA256SUMS.txt`, sort by relative path, and add those exact values to
`Submission/SHA256SUMS.txt`.

- [ ] **Step 5: Verify the hash inventory**

Run from `Submission/`:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

Expected: every file reports `OK`.

### Task 5: Final validation, commit and push

**Files:**

- Verify: all files under `Submission/`

**Interfaces:**

- Consumes: the completed unified submission.
- Produces: a clean, pushed Git commit ready for the user to compress.

- [ ] **Step 1: Validate path hygiene**

Assert:

- every path under `Submission/` is ASCII;
- no symlink exists;
- no cache or temporary path exists;
- no file exceeds GitHub's 100 MB single-file limit.

- [ ] **Step 2: Validate README links**

Parse local Markdown links in all `Submission/**/*.md` files and assert every
relative target exists.

- [ ] **Step 3: Re-run data and hash validation**

Repeat the Day 6 counts, Day 7 counts/schema, validation JSON and all checksum
checks.

- [ ] **Step 4: Review the Git diff**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only the approved move, curated submission copies, README updates
and planning document changes are present.

- [ ] **Step 5: Commit and push**

Commit with:

```bash
git commit -m "docs: organize unified teacher submission"
```

Push the current branch and verify local `HEAD` matches its origin tracking
reference.
