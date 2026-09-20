# Week 2 Day 8: Annotated Configuration and First SFT

Day 8 uses the 4,999-record Alpaca representation produced by the Day 7
cleaning pipeline to run QLoRA SFT on Qwen2.5-7B-Instruct. All GPU work was
performed on the established AutoDL RTX 3090 instance with LLaMA-Factory
0.9.3.

## Teacher Requirements

| Requirement | Implementation |
|---|---|
| Copy the official LoRA SFT YAML | Preserved byte-for-byte as `configs/qwen_lora.official.yaml` |
| Comment every parameter | The formal YAML contains 45 active parameters; every parameter has `作用`, `当前值` and `调整影响` comments |
| Explain six named parameters | `model_name_or_path`, `template`, `finetuning_type`, `lora_rank`, `lora_alpha` and `learning_rate` receive detailed comments in context |
| Use cleaned data for first SFT | Registered only the cleaned Alpaca representation and launched a 3-epoch run on all 4,999 records |
| Submit annotated YAML | `configs/qwen25_7b_week2_lora_annotated.yaml` |

## Official Configuration Source

LLaMA-Factory 0.9.3 does not contain a pure-text example literally named
`qwen_lora.yaml`. The package was searched before making any substitute.
The archived official baseline is the same-version generic text LoRA SFT
example:

```text
/root/autodl-tmp/qwen25-week1/week1/third_party/LlamaFactory/
examples/train_lora/llama3_lora_sft.yaml
```

It was copied without modification to `qwen_lora.official.yaml`; the source
and archived copy share SHA-256
`ecd8efc4cdf29d52d8a0be25028443cc813f6d82db957a90f124d7dee5095fba`.
The Qwen-specific model path, template and verified Week 1 QLoRA settings
were then introduced only in the separately annotated formal configuration.

## Six Core Parameters

- `model_name_or_path` selects the local Qwen2.5-7B-Instruct architecture,
  tokenizer and frozen base weights. The same base is required again when
  the LoRA adapter is merged on Day 9.
- `template: qwen` converts Alpaca fields into Qwen role tokens and determines
  which assistant tokens receive labels. A real sample preview verified a
  `system → user → assistant` sequence, masked all 46 prompt labels, and
  supervised all 122 response labels.
- `finetuning_type: lora` freezes the base model and trains low-rank adapter
  matrices. It is independent of `stage: sft`, which selects the supervised
  training objective.
- `lora_rank: 8` sets adapter capacity. With `lora_target: all`, the actual
  run has 20,185,088 trainable parameters, 0.2643% of 7,635,801,600 total
  parameters.
- `lora_alpha: 16` scales the LoRA update; with rank 8, the standard scaling
  factor is 2. Alpha changes update strength, not the number of parameters
  and not the optimizer learning rate.
- `learning_rate: 0.0001` controls optimizer update size. It matches the
  verified Week 1 LoRA baseline and is combined with cosine scheduling,
  10% warmup and an effective batch size of 4.

## Data Registration and Template Verification

`source/data/dataset_info.json` maps `instruction`, `input` and `output` to
LLaMA-Factory's prompt, query and response roles. The equivalent ShareGPT
copy is intentionally not registered, avoiding duplicate semantic training
examples.

Preflight validation established:

- 4,999 exact-format records and no empty instruction or output;
- data SHA-256
  `600c01cb1d2ca41bb1fbe7fca3973336568062ead29ee0e6b6033733a86b2743`;
- a 20-record schema sample check and a 2,048-token maximum inherited from
  Day 7;
- an actual Qwen template preview with no provenance field in model input;
- prompt labels masked with `-100` and assistant labels supervised;
- custom duplicate-key/comment validation and LLaMA-Factory's own argument
  parser both exiting with code 0.

## Training Configuration

| Item | Formal value |
|---|---:|
| Training records | 4,999 |
| Epochs | 3 |
| Optimization steps | 3,747 |
| Micro batch / accumulation | 1 / 4 |
| Effective batch size | 4 |
| Cutoff length | 2,048 |
| Base loading | bitsandbytes NF4 4-bit with double quantization |
| Compute precision | bf16; LayerNorm and trainable parameters upcast for stability |
| Loss logging | Every step |
| TensorBoard | Enabled |
| Checkpoint interval | 500 steps, retaining the latest 2 |
| Random seed | 42 |

The formal run started at `2026-07-29T18:09:59+08:00` on an NVIDIA GeForce
RTX 3090. A separate one-step smoke test completed first with exit code 0,
loss 2.715, and a valid 80,792,096-byte LoRA adapter. The formal run uses:

```text
/root/autodl-tmp/qwen25-week2/day8/
saves/qwen25-7b-week2-qlora-sft-run1
```

The run finished naturally with exit code 0 after 3:12:12. All 3,747 optimizer
steps have finite loss records in exact sequence. The first-100-step mean loss
was 1.6075 and the last-100-step mean was 1.0615; the full linear slope was
-0.00013193, so the overall trend is downward despite expected per-step
fluctuation and an epoch-2 plateau. The final 80,792,096-byte adapter has
SHA-256 `39eef2d0647d08cc1119ff2f80aa90224229cc6078d52982ad28a7b09b3d94fa`.

Exact final metrics are in `results/first_sft_summary.json`. The complete
engineering archive includes launch scripts, validation tests and the exact
template preview; the compact `Submission/` copy contains only the
teacher-facing annotated YAML, result summary and this README.
