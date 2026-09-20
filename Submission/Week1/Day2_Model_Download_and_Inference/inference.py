from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompt_items(path: Path) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("prompts must be a non-empty list")
    required = {"id", "system", "user", "max_new_tokens"}
    for item in items:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"prompt fields must include {sorted(required)}")
        if not all(
            isinstance(item[key], str) and item[key].strip()
            for key in ("id", "system", "user")
        ):
            raise ValueError("prompt text fields must be non-empty strings")
        if not isinstance(item["max_new_tokens"], int) or item["max_new_tokens"] <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt ids must be unique")
    return items


def slice_new_tokens(generated: torch.Tensor, prompt_length: int) -> torch.Tensor:
    return generated[:, prompt_length:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readable-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.readable_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompt_items(args.prompts)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    input_device = next(model.parameters()).device

    with args.output.open("w", encoding="utf-8") as jsonl:
        for item in prompts:
            messages = [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["user"]},
            ]
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                [rendered],
                return_tensors="pt",
                add_special_tokens=False,
            ).to(input_device)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=item["max_new_tokens"],
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            new_ids = slice_new_tokens(generated, inputs.input_ids.shape[1])
            response = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "id": item["id"],
                "model": str(args.model.resolve()),
                "device": str(input_device),
                "dtype": str(dtype),
                "seed": 42,
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": item["max_new_tokens"],
                },
                "messages": messages,
                "chat_template_text": rendered,
                "input_tokens": int(inputs.input_ids.shape[1]),
                "output_tokens": int(new_ids.shape[1]),
                "elapsed_seconds": round(elapsed, 3),
                "peak_gpu_memory_gib": round(
                    torch.cuda.max_memory_allocated() / 1024**3, 3
                ),
                "response": response,
            }
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            readable = args.readable_dir / f"day2_{item['id']}.txt"
            readable.write_text(
                "SYSTEM\n"
                + item["system"]
                + "\n\nUSER\n"
                + item["user"]
                + "\n\nGENERATION\n"
                + json.dumps(record["generation"], ensure_ascii=False)
                + "\n\nCHAT TEMPLATE\n"
                + rendered
                + "\n\nRESPONSE\n"
                + response
                + "\n",
                encoding="utf-8",
            )
            print(
                f"[{item['id']}] tokens={record['output_tokens']} "
                f"seconds={elapsed:.3f}"
            )
            print(response)


if __name__ == "__main__":
    main()
