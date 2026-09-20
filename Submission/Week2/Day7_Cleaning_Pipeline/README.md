# Week 2 Day 7: Cleaning Pipeline

Day 7 applies an independently runnable cleaning pipeline to the 5,000 Day 6
semantic samples. It removes known HTML markup and unwanted control
characters, filters invalid or empty records, performs Qwen chat-template
length truncation, and implements exact plus SimHash-based fuzzy
deduplication.

## Submitted Files

| Teacher requirement | Submitted file |
|---|---|
| Independently runnable cleaning script | [clean_pipeline.py](clean_pipeline.py) |
| Cleaned Alpaca training data | [Cleaned_Alpaca_4999.jsonl](Data/Cleaned_Alpaca_4999.jsonl) |
| Cleaned ShareGPT training data | [Cleaned_ShareGPT_4999.jsonl](Data/Cleaned_ShareGPT_4999.jsonl) |
| Before/after length distribution | [Length_Distribution.png](Figures/Length_Distribution.png) |
| Cleaning-stage count chart | [Cleaning_Counts.png](Figures/Cleaning_Counts.png) |
| Machine-readable cleaning statistics | [Cleaning_Stats.json](Results/Cleaning_Stats.json) |
| Tabular cleaning statistics | [Cleaning_Stats.csv](Results/Cleaning_Stats.csv) |
| Removed duplicate mapping | [Duplicate_Pairs.csv](Results/Duplicate_Pairs.csv) |
| SimHash threshold calibration evidence | [SimHash_Calibration.csv](Results/SimHash_Calibration.csv) |
| Final count, alignment and length validation | [Day7_Validation.json](Results/Day7_Validation.json) |

## Verified Results

| Metric | Result |
|---|---:|
| Input records | 5,000 |
| Final retained records | 4,999 |
| HTML/entity-modified records | 72 |
| Control-character-modified records | 27 |
| Whitespace-normalized records | 1,127 |
| Records over 2,048 tokens | 276 |
| Structurally truncated records | 276 |
| Exact duplicates removed | 1 |
| Fuzzy duplicates removed | 0 |
| Maximum final Qwen chat-template length | 2,048 |

The fuzzy-duplicate count of zero does not mean the algorithm was skipped.
The pipeline computes a 64-bit SimHash from character 3-grams, recalls
candidates using four 16-bit bands, and evaluates Hamming distance. Seventy-
seven real distance-0-to-6 candidate pairs were exported. None simultaneously
passed the selected Hamming threshold and the overall plus user/assistant
Jaccard confirmation gates, so they were retained to avoid false deletion.

Both cleaned files represent the same 4,999 semantic samples in different
formats. Only one representation should be registered for a single training
run.

The complete per-record audit, cleaned provenance, unit tests, determinism
report and execution logs remain in the repository engineering archive at
`deliverables/week2/day7/`. They are intentionally omitted from this compact
teacher-facing directory.
