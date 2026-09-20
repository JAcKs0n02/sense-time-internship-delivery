# Day 7 Cleaning Pipeline Implementation Plan

> **Execution:** Follow this plan in order on branch
> `codex/week2-day7-cleaning`. All tests and full-data processing run in the
> AutoDL `llm_exp` environment. Use strict RED-GREEN-REFACTOR for production
> behavior and fresh verification before every completion claim.

**Goal:** Deliver an independently runnable cleaning pipeline that converts
the 5,000 Day 6 semantic samples into aligned Alpaca and ShareGPT datasets,
removing HTML, unwanted controls, invalid/empty records and near duplicates,
and enforcing a Qwen-template length of at most 2,048 tokens.

**Architecture:** Parse the Day 6 ShareGPT representation and its provenance
into one canonical conversation model. Clean, truncate and deduplicate that
model once, then render both output schemas from the same retained records.
Use standard-library HTML/Unicode handling, the real local Qwen tokenizer,
self-implemented 64-bit SimHash with banded candidate generation, and explicit
JSON/CSV audit artifacts.

**Runtime:** Python 3.10, Transformers 4.50.0, Matplotlib 3.10.9, pytest,
standard library (`argparse`, `csv`, `hashlib`, `html.parser`, `json`,
`statistics`, `unicodedata`).

---

## Task 1: Freeze the design and prepare the AutoDL workspace

**Files:**

- Verify: `docs/superpowers/specs/2026-07-28-day7-cleaning-pipeline-design.md`
- Create remotely: `/root/autodl-tmp/qwen25-week2`

**Steps:**

1. Confirm the local branch is clean and based on the committed Day 6 data.
2. Push `codex/week2-day7-cleaning` to the existing GitHub origin.
3. Connect to the authorized AutoDL instance without printing credentials.
4. Clone the branch into `/root/autodl-tmp/qwen25-week2`.
5. Run Day 6 validation and compare the recorded SHA-256 manifest.
6. Count raw sources as 2,000/2,000/1,000 and both formatted files as 5,000.
7. Record the AutoDL Python, Transformers, Matplotlib and tokenizer paths.

**Verification:**

```bash
git status --short --branch
python deliverables/week2/day6/source/scripts/validate_day6.py \
  --deliverable-dir deliverables/week2/day6
```

Expected: clean branch, Day 6 validator exits 0, counts match the committed
manifest.

## Task 2: Scaffold the deliverable and prove the first cleaning test fails

**Files:**

- Create: `deliverables/week2/day7/source/tests/test_clean_pipeline.py`
- Create later: `deliverables/week2/day7/clean_pipeline.py`

**RED:**

1. Add a test importing `strip_html` and asserting:
   - `<p>甲&amp;乙<br>第二行</p>` becomes `甲&乙\n第二行`;
   - `<script>bad()</script><b>正文</b>` becomes `正文`;
   - `a < b and vector<string>` stays unchanged.
2. Sync the test only to AutoDL.
3. Run pytest and confirm it fails because the production module/function is
   missing, not because of a test typo.

**GREEN:**

4. Create the minimum importable `clean_pipeline.py`.
5. Implement an `HTMLParser` subclass with a known-tag whitelist, block
   boundaries and script/style suppression.
6. Sync to AutoDL and rerun the focused test.
7. Refactor only after it passes.

**Verification:**

```bash
pytest -q deliverables/week2/day7/source/tests/test_clean_pipeline.py \
  -k html
```

## Task 3: Add Unicode, whitespace and empty-value behavior

**Files:**

- Modify: `deliverables/week2/day7/source/tests/test_clean_pipeline.py`
- Modify: `deliverables/week2/day7/clean_pipeline.py`

**RED-GREEN cycles:**

1. Add a failing test for CR/LF normalization, C0/C1 removal, U+200B/U+200C
   and BOM removal, U+200D/emoji/combining mark preservation.
2. Implement `clean_controls`.
3. Add a failing test for Tab expansion, trailing-space removal, internal code
   indentation and excessive blank-line handling.
4. Implement `normalize_whitespace`.
5. Add failing Alpaca validation tests for empty `instruction`, empty
   `output`, legal empty `input`, null and wrong types.
6. Add failing ShareGPT validation tests for empty messages, wrong role order,
   incomplete pair and valid multi-turn input.
7. Implement canonical parsing and structured rejection reasons.

**Verification:**

```bash
pytest -q deliverables/week2/day7/source/tests/test_clean_pipeline.py \
  -k "control or whitespace or alpaca or sharegpt"
```

## Task 4: Implement real Qwen template length and structural truncation

**Files:**

- Modify: `deliverables/week2/day7/source/tests/test_clean_pipeline.py`
- Modify: `deliverables/week2/day7/clean_pipeline.py`

**RED:**

1. Load the real local Qwen2.5 tokenizer in AutoDL integration fixtures.
2. Construct deterministic conversations whose template lengths are exactly
   2,048, 2,049 and far above 2,048.
3. Add failing tests that:
   - keep the 2,048-token case unchanged;
   - truncate the 2,049-token case;
   - drop oldest complete pairs first;
   - retain the final user-assistant pair;
   - keep both final messages nonempty;
   - re-encode to no more than 2,048.
4. Run and confirm failure is due to missing truncation behavior.

**GREEN:**

5. Implement `chat_token_length`.
6. Implement complete-pair removal.
7. Implement token-boundary longest-message reduction with binary search.
8. Record original/final lengths, removed pairs, affected roles and removed
   token counts.
9. Rerun focused and full tests.

**Verification:**

```bash
MODEL_DIR=/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct \
pytest -q deliverables/week2/day7/source/tests/test_clean_pipeline.py \
  -k "token or trunc"
```

## Task 5: Implement exact deduplication and self-contained SimHash

**Files:**

- Modify: `deliverables/week2/day7/source/tests/test_clean_pipeline.py`
- Modify: `deliverables/week2/day7/clean_pipeline.py`

**RED-GREEN cycles:**

1. Add a failing exact-duplicate test with different provenance.
2. Implement canonical SHA-256 deduplication.
3. Add failing literal tests for deterministic 64-bit SimHash, identical text
   distance 0, and small punctuation/whitespace changes.
4. Implement character 3-grams, stable 64-bit SHA-derived feature hashes,
   weighted bit voting and Hamming distance.
5. Add a failing test proving four 16-bit bands discover all pairs with
   Hamming distance at most 3.
6. Implement banded candidate generation.
7. Add failing cluster tests for short-text protection, near duplicate
   removal, topic-similar/nonduplicate retention and deterministic
   representative selection.
8. Implement union-find clustering and evidence records.
9. Rerun the entire unit suite twice.

**Verification:**

```bash
pytest -q deliverables/week2/day7/source/tests/test_clean_pipeline.py \
  -k "exact or simhash or hamming or dedup"
```

## Task 6: Implement CLI orchestration, statistics and plots

**Files:**

- Modify: `deliverables/week2/day7/source/tests/test_clean_pipeline.py`
- Modify: `deliverables/week2/day7/clean_pipeline.py`
- Create: `deliverables/week2/day7/README.md`

**RED-GREEN cycles:**

1. Add a subprocess test for `--help`.
2. Add failing temporary-file integration tests for valid JSONL, malformed
   JSON, clear nonzero exit codes and all required output artifacts.
3. Implement argparse options:
   `--input`, `--provenance`, `--output-alpaca`, `--output-sharegpt`,
   `--tokenizer`, `--max-tokens`, `--simhash-threshold`, `--min-jaccard`,
   `--min-role-jaccard`, `--seed`, `--results-dir`, `--evidence-dir`.
4. Implement deterministic reading, canonical cleaning and aligned dual
   rendering.
5. Implement audit JSONL, provenance JSONL, statistics JSON/CSV, duplicate
   pair CSV and cluster JSONL.
6. Implement Matplotlib length and count plots using noninteractive `Agg`.
7. Add count-conservation and artifact-schema assertions.
8. Document the exact CLI, fixed order, algorithms and reproduction commands.

**Verification:**

```bash
python deliverables/week2/day7/clean_pipeline.py --help
pytest -q deliverables/week2/day7/source/tests/test_clean_pipeline.py
```

## Task 7: Calibrate SimHash on real candidates

**Files:**

- Generate:
  `deliverables/week2/day7/source/results/simhash_calibration.csv`
- Modify threshold documentation in:
  `deliverables/week2/day7/README.md`

**Steps:**

1. Run preprocessing through SimHash candidate generation without deleting
   fuzzy matches.
2. Export all candidates at distance 0–6 with sources and bounded text
   previews.
3. Combine real boundary candidates with controlled punctuation, whitespace
   and wording perturbations.
4. Label duplicate/nonduplicate based on semantic equivalence.
5. Compare thresholds 0–6 and select the highest threshold that preserves
   high precision; do not tune for the 1,500-row floor.
6. Save labels, distance, decision, selected threshold and notes.
7. Run focused false-positive tests using observed boundary examples.

**Verification:** Calibration CSV has explicit decisions and the selected
threshold matches the formal full-run command.

## Task 8: Run the complete 5,000-sample pipeline on AutoDL

**Files generated under:** `deliverables/week2/day7/`

**Steps:**

1. Start from a clean output directory while preserving Day 6 inputs.
2. Run the formal CLI with the real tokenizer, 2,048 limit, seed 42 and the
   calibrated threshold.
3. Tee stdout/stderr to `day7_clean_pipeline.txt`.
4. Run the final validator against both output formats and all audit files.
5. Verify:
   - final semantic count is at least 1,500;
   - two output files have identical counts and aligned IDs;
   - every template length is at most 2,048;
   - count equations balance;
   - duplicate mappings resolve to retained samples;
   - charts exist, are nonempty and have readable dimensions.
6. Generate `day7_files.sha256`.

**Formal command:**

```bash
conda run -n llm_exp python \
  deliverables/week2/day7/clean_pipeline.py \
  --input deliverables/week2/day6/source/data/formatted/week2_5k_sharegpt.jsonl \
  --provenance deliverables/week2/day6/source/data/interim/day6_provenance.jsonl \
  --output-alpaca deliverables/week2/day7/data/week2_clean_alpaca.jsonl \
  --output-sharegpt deliverables/week2/day7/data/week2_clean_sharegpt.jsonl \
  --tokenizer /root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct \
  --max-tokens 2048 \
  --simhash-threshold 3 \
  --min-jaccard 0.85 \
  --min-role-jaccard 0.90 \
  --seed 42 \
  --results-dir deliverables/week2/day7/source/results \
  --evidence-dir deliverables/week2/day7/evidence
```

## Task 9: Prove determinism and inspect visual evidence

**Steps:**

1. Copy first-run hashes to a temporary evidence file outside the deliverable.
2. Rerun the complete pipeline into a separate temporary output directory.
3. Compare cleaned JSONL, audit, statistics, duplicate evidence and hashes.
4. Explain any intentionally nondeterministic log timestamp; all semantic
   artifacts must be byte-identical.
5. Open both PNG files and visually inspect labels, legends, 2,048 marker and
   clipping.
6. If a visual issue exists, add a plot behavior check where practical,
   adjust plotting code and rerun.

**Verification:**

```bash
sha256sum -c deliverables/week2/day7/source/results/day7_files.sha256
pytest -q deliverables/week2/day7/source/tests/test_clean_pipeline.py
```

## Task 10: Synchronize, document and commit

**Files:**

- Modify: `README.md`
- Modify: `docs/week2_execution_plan.md`
- Verify all files under: `deliverables/week2/day7/`

**Steps:**

1. Download/synchronize the AutoDL Day 7 directory into the local branch.
2. Verify local SHA-256 values and rerun schema/statistics validation where it
   does not require a missing tokenizer.
3. Update root README Day 7 status only after fresh validation.
4. Mark each Day 7 checklist item with evidence paths and actual counts.
5. Review `git diff`, ensure no credentials, caches, model files or temporary
   data are included.
6. Run `git diff --check`, tests and final validation again.
7. Commit the complete Day 7 deliverable.
8. Push `codex/week2-day7-cleaning`.
9. Confirm the remote branch points to the final commit.
10. Shut down the AutoDL instance only after the push and local hashes are
    verified.

**Final evidence commands:**

```bash
git status --short --branch
git diff --check
pytest -q deliverables/week2/day7/source/tests/test_clean_pipeline.py
sha256sum -c deliverables/week2/day7/source/results/day7_files.sha256
```
