# Submission Integrity Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the teacher-facing `Submission/` package and add a repeatable gate that rejects hidden metadata, duplicate-copy paths, missing Week 3 deliverables, forbidden artifacts, broken links, and an incomplete or stale SHA-256 manifest.

**Architecture:** Keep teacher deliverables under `Submission/` and engineering validation under `deliverables/week3/day16/`. Extend the existing Week 3 validator with a Submission contract, then regenerate the root manifest deterministically after removing local metadata. The validator reads the package without modifying it and reports all violations in the existing JSON check format.

**Tech Stack:** Python 3.10+, `pathlib`, `hashlib`, `csv`, `json`, pytest, POSIX `shasum`.

## Global Constraints

- `Submission/` contains only teacher-facing deliverables and its SHA-256 manifest.
- Do not add model weights, checkpoints, optimizer state, caches, bytecode, symlinks, private blind mappings, or engineering tests to `Submission/`.
- Exclude `.DS_Store` and `SHA256SUMS.txt` itself from the manifest.
- Reject path components ending in copy suffixes such as ` 2`, ` (1)`, ` copy`, or `副本`.
- Preserve all formal Week 3 results and raw logs byte-for-byte unless a teacher-facing relative path must be adjusted.

---

### Task 1: Add the Submission contract test

**Files:**
- Modify: `deliverables/week3/day16/source/tests/test_validate_week3.py`
- Test: `deliverables/week3/day16/source/tests/test_validate_week3.py`

**Interfaces:**
- Consumes: a temporary `Submission/` fixture.
- Produces: failing tests for `validate_submission(submission_root: Path) -> list[str]`.

- [x] Add a repository-level integration assertion covering every required Week 3 teacher deliverable and the deterministic `SHA256SUMS.txt`.
- [x] Assert that the current formal package returns no errors.
- [x] Add `.DS_Store`, a `Day14_Model_Evaluation 2/` directory, a symlink, a forbidden weight file, and a stale hash; assert that each violation is reported.
- [x] Run `pytest -q deliverables/week3/day16/source/tests/test_validate_week3.py` and confirm failure because `validate_submission` does not yet exist.

### Task 2: Implement the Submission gate

**Files:**
- Modify: `deliverables/week3/day16/source/scripts/validate_week3.py`
- Test: `deliverables/week3/day16/source/tests/test_validate_week3.py`

**Interfaces:**
- Consumes: `Submission/` and `Submission/SHA256SUMS.txt`.
- Produces: `validate_submission(submission_root: Path) -> list[str]` and a `submission_package` check in the existing validator output.

- [x] Validate required Week 3 paths, ASCII path components, duplicate-copy suffixes, forbidden names/extensions, symlinks, manifest ordering, one-to-one manifest coverage, and every SHA-256 digest.
- [x] Add the gate to `main()` as `submission_package`.
- [x] Re-run the focused test and confirm it passes.

### Task 3: Repair the teacher-facing package

**Files:**
- Remove locally: `Submission/.DS_Store`
- Regenerate: `Submission/SHA256SUMS.txt`

**Interfaces:**
- Consumes: the final regular-file tree under `Submission/`.
- Produces: a sorted manifest containing every regular file exactly once except itself and `.DS_Store`.

- [x] Move `Submission/.DS_Store` to a recoverable Trash location.
- [x] Regenerate the manifest with `LC_ALL=C` path ordering and SHA-256.
- [x] Run `shasum -a 256 -c SHA256SUMS.txt` from `Submission/` and confirm all entries pass.

### Task 4: Run final acceptance and publish the repair

**Files:**
- Verify: `Submission/**`
- Verify: `deliverables/week3/day16/source/scripts/validate_week3.py`
- Verify: `deliverables/week3/day16/source/tests/test_validate_week3.py`

**Interfaces:**
- Consumes: repaired Submission and Week 3 evidence.
- Produces: clean tests, clean Git diff, committed and pushed correction.

- [x] Run the complete Week 3 pytest suite.
- [x] Run `validate_week3.py` and confirm all checks, including `submission_package`, pass.
- [x] Re-scan the repository for duplicate-copy suffixes and forbidden Submission artifacts.
- [x] Verify `main` tracking state, commit only the audit-related files, and push `main`.
