#!/usr/bin/env python3
"""Fail closed when trainable LoRA tensors touch the visual stack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORBIDDEN_TOKENS = ("visual", "vision", "merger", "projector", "mm_projector")
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def audit_trainable_names(names: list[str]) -> dict:
    forbidden = [name for name in names if any(token in name.casefold() for token in FORBIDDEN_TOKENS)]
    non_lora = [name for name in names if "lora_" not in name.casefold()]
    return {
        "passed": not forbidden and not non_lora and bool(names),
        "trainable_count": len(names),
        "forbidden_names": forbidden,
        "non_lora_names": non_lora,
    }


def _tensor_sha256(tensor) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def inspect_model(model_dir: Path) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
    )
    visual_parameter = model.visual.patch_embed.proj.weight
    projector_parameter = model.visual.merger.mlp[0].weight
    frozen_hashes = {
        "visual.patch_embed.proj.weight": _tensor_sha256(visual_parameter),
        "visual.merger.mlp.0.weight": _tensor_sha256(projector_parameter),
    }
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(TARGET_MODULES),
    )
    model = get_peft_model(model, peft_config)
    names = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    audit = audit_trainable_names(names)
    audit.update(
        {
            "status": "PASS" if audit["passed"] else "FAIL",
            "model_dir": str(model_dir),
            "target_modules": list(TARGET_MODULES),
            "trainable_names": names,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "all_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "frozen_reference_tensor_sha256": frozen_hashes,
        }
    )
    if not audit["passed"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_model(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "trainable_names"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
