# Week 2 Day 9: Training Monitoring, LoRA Merge, and Evaluation

Day 9 reuses the completed Day 8 SFT run on the established AutoDL RTX 3090
instance. It verifies every recorded loss step, opens the original event file
in TensorBoard, merges the final LoRA adapter into Qwen2.5-7B-Instruct with
LLaMA-Factory 0.9.3, validates the merged model folder, and runs three frozen
questions against both the base and merged models.

## Teacher Requirements

| Requirement | Result |
|---|---|
| Monitor loss in TensorBoard and preserve every step | The original event file contains the formal run; 3,747/3,747 finite loss records were exported to CSV and a TensorBoard screenshot was captured |
| Merge LoRA with `llamafactory-cli export` | Exit code 0; four full `safetensors` shards were written |
| Load the merged model | Successful BF16 CUDA load and generation for all three questions |
| Test with three new questions | Three questions were frozen before inference and checked against all 4,999 training prompts for novelty |
| Preserve the conversation record | Base and merged JSONL responses, manifests, rubric scores, and comparison report are archived |

## Loss Evidence

- Optimizer steps: 3,747.
- First loss: 1.4029; last loss: 0.9737.
- First 100-step moving average: 1.607461.
- Last 100-step moving average: 1.061506.
- Linear slope: -0.00013193245.
- Non-finite losses: 0.
- Overall downward trend: yes.

The step-level evidence is in
`source/results/train_loss_by_step.csv`; aggregate calculations are in
`source/results/loss_summary.json`. The screenshot is the TensorBoard UI
reading the original Day 8 event file, not a replacement chart.

## Merge Evidence

The version-matched official export example was archived before writing
`configs/qwen25_7b_week2_export.yaml`. The formal command was:

```text
llamafactory-cli export configs/qwen25_7b_week2_export.yaml
```

The export finished with code 0. The merged directory is:

```text
/root/autodl-tmp/qwen25-week2/merged/qwen25-7b-week2-sft-merged
```

Validation found 14 files, including four model-weight shards totaling
15,231,271,872 bytes. `config.json`, tokenizer files, chat template and the
weight index are present; no `adapter_model` file remains, confirming that
this is a full merged model rather than an adapter-only folder. Exact file
sizes and SHA-256 hashes are in the inventory files.

## Three-Question Evaluation

The question file and deterministic generation configuration were frozen
before inference (`do_sample: false`, seed 42, BF16, maximum 512 new tokens).
A Jaccard novelty check found maximum similarities of 0.146789, 0.116959 and
0.104478, all far below the 0.80 exclusion threshold.

Blind rubric scoring produced:

| Model | Total | Generation-quality regression |
|---|---:|---|
| Model A (base) | 25/30 | No |
| Model B (merged) | 22/30 | No |

The merged model was substantially better on Q3, producing a concise
executable instruction with an explicit acceptance standard. It regressed on
Q2 by emitting four points in sections explicitly limited to three, and both
models exceeded Q1's 150-character limit. Therefore this controlled
three-question check does **not** demonstrate that the merged model is
globally better than the base model. This result is reported as measured;
the Week 2 acceptance criterion “merged conversation quality is better than
base” remains unmet and should be addressed with more targeted,
constraint-following data plus a separate held-out validation set.

## Engineering Archive

- `configs/`: validated LLaMA-Factory export configuration.
- `evidence/`: TensorBoard screenshot.
- `source/results/`: step loss, novelty check, manifests, model inventory,
  hashes, raw responses, blind scores and comparison report.
- `source/scripts/`: metric export, novelty check, configuration validation,
  merge validation, deterministic inference and report generation.
- `source/tests/`: unit tests for the Day 9 scripts.
