# Week 2 Acceptance Matrix

| ID | Acceptance criterion | Status | Measured result | Primary evidence |
|---:|---|---|---|---|
| 1 | `clean_pipeline.py` runs independently | PASS | CLI implementation; 20 Day 7 tests passed; 12 semantic rerun artifacts matched byte-for-byte | [`day7/clean_pipeline.py`](../../../day7/clean_pipeline.py), [`day7_validation.json`](../../../day7/source/results/day7_validation.json), [`determinism_check.txt`](../../../day7/source/results/determinism_check.txt) |
| 2 | Cleaned training set has at least 1,500 correctly formatted records | PASS | 4,999 retained records in aligned Alpaca and ShareGPT representations; maximum Qwen template length 2,048 | [`cleaning_stats.json`](../../../day7/source/results/cleaning_stats.json), [`day7_validation.json`](../../../day7/source/results/day7_validation.json) |
| 3 | SFT loss decreases stably | PASS | 3,747/3,747 sequential finite records; 100-step mean 1.607461 → 1.061506; linear slope -0.00013193245 | [`loss_summary.json`](../../../day9/source/results/loss_summary.json), [`train_loss_by_step.csv`](../../../day9/source/results/train_loss_by_step.csv), [`TensorBoard`](../../../day9/evidence/tensorboard_train_loss.png) |
| 4 | Merged model conversation quality exceeds the base model | **FAIL** | Frozen three-question blind score: base 25/30, merged 22/30; merge/load validation itself passed | [`base_vs_merged_comparison.json`](../../../day9/source/results/base_vs_merged_comparison.json), [`merged_model_validation.json`](../../../day9/source/results/merged_model_validation.json) |
| 5 | Week 2 report is submitted | PASS | Report and FAQ are present and link to the evidence above | [`REPORT.md`](../../REPORT.md), [`FAQ.md`](../../FAQ.md) |

Status is based on the evidence available at the end of Day 10. A successful export or a
decreasing training loss is not used as a substitute for criterion 4.
