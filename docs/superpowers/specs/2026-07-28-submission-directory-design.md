# Unified Submission Directory Design

## 1. Goal

Create one teacher-facing directory at the repository root:

```text
Submission/
├── README.md
├── SHA256SUMS.txt
├── Week1/
└── Week2/
```

The existing root-level `Week1/` submission will move into `Submission/Week1/`.
Week 2 will be curated from the complete experiment archives under
`deliverables/week2/day6/` and `deliverables/week2/day7/`.

The submission directory must be directly usable without symlinks or a
compressed archive. The user will create any archive separately.

## 2. Naming Rules

Every directory and filename under `Submission/` must use ASCII English names,
digits, underscores, hyphens or dots. Chinese text may remain inside Markdown
reports, logs and data records.

The existing Week 1 submission paths will be renamed:

```text
Submission/Week1/
├── README.md
├── Day1_Environment_Setup/
├── Day2_Model_Download_and_Inference/
├── Day3_Architecture_Analysis/
├── Day4_Tokenizer_Experiments/
└── Day5_LLaMA_Factory_and_Weekly_Report/
```

Chinese Week 1 report filenames will also receive clear English names:

- `Qwen2.5_Architecture_Report.md`
- `Week1_Report.md`

Existing already-English artifact filenames may remain unchanged.

## 3. Week 2 Submission Scope

Week 2 contains only Day 6 and Day 7 because later days have not been
performed. It will use this structure:

```text
Submission/Week2/
├── README.md
├── Day6_Data_Collection_and_Formatting/
│   ├── README.md
│   ├── Data/
│   │   ├── Formatted_Alpaca_5K.jsonl
│   │   ├── Formatted_ShareGPT_5K.jsonl
│   │   └── Data_Provenance.jsonl
│   └── Manifest/
│       ├── Raw_Data_Manifest.md
│       ├── Raw_Data_Manifest.csv
│       └── Collection_Metadata.json
└── Day7_Cleaning_Pipeline/
    ├── README.md
    ├── clean_pipeline.py
    ├── Data/
    │   ├── Cleaned_Alpaca_4999.jsonl
    │   └── Cleaned_ShareGPT_4999.jsonl
    ├── Figures/
    │   ├── Length_Distribution.png
    │   └── Cleaning_Counts.png
    └── Results/
        ├── Cleaning_Stats.json
        ├── Cleaning_Stats.csv
        ├── Duplicate_Pairs.csv
        ├── SimHash_Calibration.csv
        └── Day7_Validation.json
```

### Day 6 rationale

The teacher explicitly requires:

1. 2K Alpaca-GPT4-zh, 2K COIG-PC and 1K ShareGPT-zh samples;
2. unified Alpaca and ShareGPT formats;
3. an original-data archive manifest.

The submission therefore includes the two formatted 5K datasets, their
provenance mapping, and the human-readable and machine-readable archive
manifest. The larger raw subsets and collection scripts remain in
`deliverables/week2/day6/`; their exact revisions, counts, sizes and hashes are
recorded in the submitted manifest.

### Day 7 rationale

The teacher explicitly requires:

1. independently runnable `clean_pipeline.py`;
2. cleaned data containing at least 1,500 records;
3. HTML/control-character/empty-value cleaning and 2,048-token truncation;
4. SimHash or Jaccard fuzzy deduplication;
5. before/after counts and Matplotlib length figures.

The submission therefore includes the standalone script, both cleaned
formats, both figures, the numeric statistics, duplicate evidence, SimHash
calibration and final validation. Per-record audit files, test source,
determinism logs and intermediate process logs remain in the complete
`deliverables/` archive because they are not part of the teacher's requested
submission.

## 4. README Responsibilities

`Submission/README.md` is the single entry point. It will:

- explain that Week 1 is complete and Week 2 currently covers Day 6–Day 7;
- link to each week's README;
- state that all GPU work ran on AutoDL;
- state that `deliverables/` contains the complete engineering archive.

`Submission/Week2/README.md` will map every Day 6 and Day 7 teacher requirement
to an exact submitted file and summarize the verified counts:

- Day 6: 2,000 + 2,000 + 1,000 = 5,000;
- Day 7: 5,000 input, 4,999 retained;
- 276 truncated records;
- one exact duplicate removed;
- no fuzzy candidate passed the precision-preserving confirmation gates;
- maximum final Qwen chat-template length of 2,048 tokens.

Each day README will describe only the files present in the submission copy and
point to the complete `deliverables/` directory for optional audit material.

## 5. Integrity and Validation

`Submission/SHA256SUMS.txt` will contain hashes for every submitted file except
itself, using paths relative to `Submission/`.

Final validation must confirm:

1. no non-ASCII path component exists under `Submission/`;
2. no `__pycache__`, `.pytest_cache`, temporary file or symlink is present;
3. every README link resolves;
4. Day 6 formatted files each contain 5,000 valid records;
5. Day 6 manifest reports source counts 2,000/2,000/1,000;
6. Day 7 cleaned files each contain 4,999 valid records;
7. Day 7 validation is `valid: true` with maximum length 2,048;
8. all SHA-256 entries verify;
9. the complete originals under `deliverables/week2/` remain unchanged;
10. `git diff --check` passes.

## 6. Git Handling

The existing root `Week1/` will be moved, not copied, so there is only one
teacher-facing Week 1 folder. Week 2 submission files are curated copies;
`deliverables/week2/` remains the canonical complete archive.

The design document is committed before implementation. The completed
submission reorganization is committed separately and pushed to the current
branch.
