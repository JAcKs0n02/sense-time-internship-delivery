# Merged Model Folder

The full merged Qwen2.5-7B-Instruct model is stored on AutoDL at:

```text
/root/autodl-tmp/qwen25-week2/merged/qwen25-7b-week2-sft-merged
```

The model was produced with LLaMA-Factory 0.9.3 by merging the final Day 8
LoRA adapter into the exact base model used for SFT. Formal export exit code
was 0.

## Validation

- File count: 14.
- Full weight shards: 4.
- Weight bytes: 15,231,271,872.
- Total directory bytes: 15,247,180,180.
- Model type: `qwen2`.
- Tokenizer chat template: present.
- Adapter-only weight file: absent.
- BF16 CUDA load and three-question generation: successful.

`Model_Inventory.txt` records every relative path and byte size.
`Model_SHA256.txt` records a SHA-256 digest for every file. These two files
make the external 15 GB AutoDL folder auditable without placing the model
weights in Git.
