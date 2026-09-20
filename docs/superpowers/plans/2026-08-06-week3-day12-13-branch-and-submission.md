# Week 3 Day 12–13 Branch and Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish an independent Day 12–13 branch whose READMEs reflect the completed nine-run experiment and whose `Submission/Week3` contains only the teacher-required Day 11 and Day 12–13 deliverables.

**Architecture:** Keep the complete engineering archive under `deliverables/week3`, then build a deliberately smaller teacher-facing projection under `Submission/Week3`. All displayed metrics and copied logs come from the frozen Day 12–13 result bundle; machine checks enforce run-set, field, hash, forbidden-artifact, link, branch, and `main` invariants.

**Tech Stack:** Git, Markdown, JSON, CSV, POSIX shell tools, Python 3 standard library, pytest, SHA-256 (`shasum`).

## Global Constraints

- Work only on `codex/week3-day12-13-experiments`.
- Do not switch, merge, commit to, or push `main`.
- `Submission/Week3` contains completed Day 11 and Day 12–13 only; do not create Day 14–16 placeholders.
- Do not copy scripts, tests, training data, adapters, model weights, checkpoints, optimizer state, caches, or `__pycache__` into `Submission/Week3`.
- Teacher-facing metrics must be copied from `deliverables/week3/day12_13/source/results/experiment_summary.csv` and `.json`, not manually recomputed.
- Preserve exactly nine run IDs: `rank-r8`, `rank-r32`, `rank-r64`, `lr-1e-4`, `lr-2e-4`, `lr-5e-5`, `epoch-e2`, `epoch-e3`, `epoch-e5`.
- Preserve all 108 allowlisted raw-log files and verify their content hashes.
- Push the feature branch to `origin`; leave `main` and `origin/main` at `9dd3470`.

---

### Task 1: Update the engineering README hierarchy

**Files:**
- Modify: `README.md`
- Create: `deliverables/week3/README.md`
- Modify: `deliverables/week3/day12_13/README.md`
- Create: `deliverables/week3/day12_13/source/results/README.md`

**Interfaces:**
- Consumes: frozen summary files under `deliverables/week3/day12_13/source/results/`
- Produces: repository and engineering-archive navigation used by the teacher-facing index

- [ ] **Step 1: Run the pre-change documentation audit and verify the expected gaps**

```bash
python - <<'PY'
from pathlib import Path
root = Path('.')
assert 'Day 12' not in (root / 'README.md').read_text()
assert not (root / 'deliverables/week3/README.md').exists()
assert not (root / 'deliverables/week3/day12_13/source/results/README.md').exists()
print('expected README gaps confirmed')
PY
```

Expected: prints `expected README gaps confirmed`.

- [ ] **Step 2: Update the root README**

Use `apply_patch` to make these exact semantic changes:

- opening status: Week 3 Day 11–13 complete;
- Week 3 progress table: add a Day 12–13 row linked to `deliverables/week3/day12_13/README.md`;
- core results: add 9/9 completed, exit codes 0, Final Loss/runtime/VRAM captured, and 108 hashes verified;
- repository tree: add `day12_13/` and `Submission/Week3/`;
- learning links: add the Day 12–13 README and result summary.

- [ ] **Step 3: Create `deliverables/week3/README.md`**

The file must contain:

```markdown
# Week 3 Engineering Archive

| Stage | Status | Engineering entry | Teacher deliverable |
|---|---|---|---|
| Day 11 | Completed | `day11/README.md` | Experiment plan with hypotheses |
| Day 12–13 | Completed | `day12_13/README.md` | Nine-run raw logs and metric summary |
```

Also explain that Day 14–16 are added only after completion and that `deliverables/` retains scripts/tests/audit details while `Submission/Week3/` is the reduced teacher view.

- [ ] **Step 4: Convert the Day 12–13 README from target-state wording to actual results**

Add this exact result table from `experiment_summary.json`:

| Run ID | Final Loss | Last-100 Mean | Train Time | Peak VRAM MiB | Steps | Status |
|---|---:|---:|---:|---:|---:|---|
| rank-r8 | 0.9714 | 1.061565 | 03:11:36 | 21096 | 3747 | completed |
| rank-r32 | 0.9925 | 1.089147 | 03:12:11 | 21836 | 3747 | completed |
| rank-r64 | 0.9968 | 1.101392 | 03:13:46 | 22682 | 3747 | completed |
| lr-1e-4 | 0.9701 | 1.063174 | 03:10:58 | 21096 | 3747 | completed |
| lr-2e-4 | 0.9056 | 0.876906 | 03:11:27 | 21096 | 3747 | completed |
| lr-5e-5 | 1.0299 | 1.166453 | 03:11:22 | 21096 | 3747 | completed |
| epoch-e2 | 1.3368 | 1.250650 | 02:07:24 | 21096 | 2498 | completed |
| epoch-e3 | 0.9744 | 1.061932 | 03:11:11 | 21096 | 3747 | completed |
| epoch-e5 | 0.9867 | 0.627596 | 05:18:57 | 21096 | 6245 | completed |

State that lower training loss does not select the best model; Day 14 blind human scoring remains the selection gate. Record the completed 9/9 gate and 108/108 raw-log hashes.

- [ ] **Step 5: Create the result-directory README**

Document `experiment_summary.csv`, `experiment_summary.json`, `baseline_variability.json`, `completed_adapters.json`, `raw_logs.sha256`, and `raw_logs/<run-id>/attempt-001/`. Include the local verification command:

```bash
cd deliverables/week3/day12_13/source/results
shasum -a 256 -c raw_logs.sha256
```

Explicitly prohibit `.safetensors`, `.bin`, `checkpoint-*`, `optimizer.pt`, training data, and symlinks.

- [ ] **Step 6: Verify README facts and links, then commit**

```bash
python - <<'PY'
from pathlib import Path
for path in [
    Path('README.md'),
    Path('deliverables/week3/README.md'),
    Path('deliverables/week3/day12_13/README.md'),
    Path('deliverables/week3/day12_13/source/results/README.md'),
]:
    text = path.read_text()
    assert 'Day 12' in text or 'Day 12–13' in text
print('README content gate passed')
PY
git diff --check
git add README.md deliverables/week3/README.md \
  deliverables/week3/day12_13/README.md \
  deliverables/week3/day12_13/source/results/README.md
git commit -m "docs: record week 3 day 12-13 results"
```

Expected: content gate and `git diff --check` pass; commit succeeds.

### Task 2: Build the minimal Week 3 teacher submission

**Files:**
- Create: `Submission/Week3/README.md`
- Create: `Submission/Week3/Day11_Experiment_Design/README.md`
- Create: `Submission/Week3/Day12_13_Batch_Experiments/README.md`
- Copy: `Submission/Week3/Day12_13_Batch_Experiments/Results/Experiment_Summary.csv`
- Copy: `Submission/Week3/Day12_13_Batch_Experiments/Results/Experiment_Summary.json`
- Copy: `Submission/Week3/Day12_13_Batch_Experiments/Raw_Logs/**`
- Create: `Submission/Week3/Day12_13_Batch_Experiments/Raw_Logs.sha256`

**Interfaces:**
- Consumes: Day 11 experiment table and Day 12–13 frozen summary/raw logs
- Produces: teacher-facing Week 3 entry without engineering-only material

- [ ] **Step 1: Verify the teacher submission is absent before creation**

```bash
test ! -e Submission/Week3
```

Expected: exit code 0.

- [ ] **Step 2: Copy only the frozen machine-readable results and raw logs**

```bash
mkdir -p Submission/Week3/Day12_13_Batch_Experiments/Results
mkdir -p Submission/Week3/Day12_13_Batch_Experiments/Raw_Logs
cp deliverables/week3/day12_13/source/results/experiment_summary.csv \
  Submission/Week3/Day12_13_Batch_Experiments/Results/Experiment_Summary.csv
cp deliverables/week3/day12_13/source/results/experiment_summary.json \
  Submission/Week3/Day12_13_Batch_Experiments/Results/Experiment_Summary.json
cp -R deliverables/week3/day12_13/source/results/raw_logs/. \
  Submission/Week3/Day12_13_Batch_Experiments/Raw_Logs/
```

- [ ] **Step 3: Create the three teacher-facing READMEs with `apply_patch`**

`Submission/Week3/README.md` must list only Day 11 and Day 12–13, link their entry files, summarize 9/9 completion, and link back to `deliverables/week3/README.md` for engineering evidence.

`Day11_Experiment_Design/README.md` must contain the nine-row plan table and hypotheses, baseline `rank=8 / lr=1e-4 / epochs=3 / alpha=16 / seed=42`, the `alpha/rank` limitation, and the purpose of three baseline repeats.

`Day12_13_Batch_Experiments/README.md` must contain the nine-row actual result table from Task 1, define Final Loss and sampled peak VRAM, link `Results/Experiment_Summary.csv`, `Results/Experiment_Summary.json`, `Raw_Logs/`, and `Raw_Logs.sha256`, and state that model ranking waits for Day 14 human blind scoring.

- [ ] **Step 4: Generate the teacher-submission raw-log checksum manifest**

```bash
cd Submission/Week3/Day12_13_Batch_Experiments
find Raw_Logs -type f -print0 | LC_ALL=C sort -z | \
  xargs -0 shasum -a 256 > Raw_Logs.sha256
shasum -a 256 -c Raw_Logs.sha256
```

Expected: 108 `OK` lines.

- [ ] **Step 5: Cross-check copied content against the engineering archive**

```bash
python - <<'PY'
from hashlib import sha256
from pathlib import Path
src = Path('deliverables/week3/day12_13/source/results/raw_logs')
dst = Path('Submission/Week3/Day12_13_Batch_Experiments/Raw_Logs')
src_files = sorted(p.relative_to(src) for p in src.rglob('*') if p.is_file())
dst_files = sorted(p.relative_to(dst) for p in dst.rglob('*') if p.is_file())
assert src_files == dst_files
assert len(src_files) == 108
for rel in src_files:
    assert sha256((src / rel).read_bytes()).digest() == sha256((dst / rel).read_bytes()).digest()
print('raw log mirror gate passed: 108/108')
PY
```

- [ ] **Step 6: Reject forbidden or premature submission content**

```bash
python - <<'PY'
from pathlib import Path
root = Path('Submission/Week3')
forbidden_names = {'optimizer.pt', '__pycache__', 'tests', 'scripts', 'data'}
for p in root.rglob('*'):
    assert p.name not in forbidden_names, p
    assert p.suffix not in {'.safetensors', '.bin', '.pyc'}, p
    assert not p.name.startswith('checkpoint-'), p
for day in ('Day14', 'Day15', 'Day16'):
    assert not any(day in str(p) for p in root.rglob('*'))
print('submission scope gate passed')
PY
```

Expected: prints `submission scope gate passed`.

- [ ] **Step 7: Commit the Week 3 teacher projection**

```bash
git add Submission/Week3
find Submission/Week3/Day12_13_Batch_Experiments/Raw_Logs \
  -type f -name train.log -print0 | xargs -0 git add -f
git diff --cached --check -- 'Submission/Week3/**/*.md'
git commit -m "docs: add week 3 day 11-13 teacher submission"
```

Expected: commit succeeds with no forbidden artifact. `train.log` requires the explicit `-f` because the repository intentionally ignores ordinary runtime logs; these nine files are teacher-required frozen evidence. Do not normalize copied CSV or raw-log whitespace because byte equality with the engineering archive is part of the evidence gate.

### Task 3: Update the submission index and global checksums

**Files:**
- Modify: `Submission/README.md`
- Regenerate: `Submission/SHA256SUMS.txt`

**Interfaces:**
- Consumes: final `Submission/Week3` tree
- Produces: repository-wide teacher-submission entry and integrity manifest

- [ ] **Step 1: Add the Week 3 row to `Submission/README.md`**

Use `apply_patch` to add:

| Week 3 | Day 11 experiment design and Day 12–13 nine-run raw logs with Final Loss, runtime and VRAM summary | [Week 3 Submission](Week3/README.md) |

Update the directory policy to state that Week 3 currently contains completed Day 11–13 only.

- [ ] **Step 2: Regenerate `Submission/SHA256SUMS.txt` deterministically**

```bash
cd Submission
find . -type f ! -name SHA256SUMS.txt ! -name .DS_Store -print0 | LC_ALL=C sort -z | \
  xargs -0 shasum -a 256 | sed 's#  \./#  #' > SHA256SUMS.txt
shasum -a 256 -c SHA256SUMS.txt
```

Expected: every listed file reports `OK`, each path occurs exactly once, and `SHA256SUMS.txt` does not list itself.

- [ ] **Step 3: Verify summary and raw-log manifests**

```bash
python - <<'PY'
from pathlib import Path
lines = Path('Submission/SHA256SUMS.txt').read_text().splitlines()
paths = [line.split('  ', 1)[1] for line in lines]
assert len(paths) == len(set(paths))
assert 'SHA256SUMS.txt' not in paths
assert 'Week3/README.md' in paths
assert sum('/Raw_Logs/' in p for p in paths) == 108
print('submission checksum index gate passed')
PY
```

- [ ] **Step 4: Commit the index and checksum update**

```bash
git add Submission/README.md Submission/SHA256SUMS.txt
git diff --cached --check
git commit -m "docs: index week 3 teacher submission"
```

### Task 4: Run final repository verification and publish the branch

**Files:**
- Verify only; no expected file changes

**Interfaces:**
- Consumes: Tasks 1–3 commits
- Produces: verified remote feature branch

- [ ] **Step 1: Run Week 3 tests**

```bash
/Users/yifanren/anaconda3/bin/python -m pytest -q \
  deliverables/week3/day11/source/tests \
  deliverables/week3/day12_13/source/tests
```

Expected: 39 tests pass.

- [ ] **Step 2: Run repository regression tests**

```bash
/Users/yifanren/anaconda3/bin/python -m pytest -q deliverables \
  --ignore=deliverables/week1/day2/source/tests/test_day2_scripts.py
```

Expected baseline: 116 passed, 5 skipped. The ignored historical test requires local PyTorch and is unrelated to documentation/submission changes.

- [ ] **Step 3: Verify Markdown relative links**

Run a Python standard-library scanner over every changed README. Ignore external URLs and anchors; resolve each relative Markdown target from the containing file and assert it exists.

- [ ] **Step 4: Run final data and repository gates**

```bash
cd deliverables/week3/day12_13/source/results
shasum -a 256 -c raw_logs.sha256
cd ../../../../..
cd Submission/Week3/Day12_13_Batch_Experiments
shasum -a 256 -c Raw_Logs.sha256
cd ../..
git diff --check
git status --short --branch
git log -5 --oneline --decorate
test "$(git rev-parse main)" = "9dd3470daa77b0e70f617173dae7d7d6ca8dc5b0"
test "$(git rev-parse origin/main)" = "9dd3470daa77b0e70f617173dae7d7d6ca8dc5b0"
```

Expected: both 108-file manifests pass; working tree clean; `main` and `origin/main` unchanged.

- [ ] **Step 5: Push and verify the new branch**

```bash
git push -u origin codex/week3-day12-13-experiments
git status --short --branch
git ls-remote --heads origin codex/week3-day12-13-experiments
```

Expected: upstream is `origin/codex/week3-day12-13-experiments`, local branch is clean, and the remote head equals local HEAD.
