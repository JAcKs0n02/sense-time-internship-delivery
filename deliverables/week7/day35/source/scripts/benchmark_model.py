#!/usr/bin/env python3
"""Run the frozen offline benchmark for one BF16/AWQ/GPTQ model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

from week7_deployment.benchmark import summarize_timings
from week7_deployment.hashing import portable_repo_path, sha256_file


MIB = 1024 * 1024


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _validate_inputs(config: dict[str, object], repo_root: Path) -> None:
    for section_name in ("generation_eval", "perplexity_eval"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} config missing")
        path = repo_root / str(section.get("path", ""))
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != section.get("sha256"):
            raise ValueError(f"{section_name} sha256 mismatch")


def _gpu_used_mib() -> float:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    process_id = str(os.getpid())
    total = 0.0
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == process_id:
            total += float(fields[1])
    return total


def _weight_bytes(model_dir: Path) -> int:
    return sum(
        path.stat().st_size
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix in {".safetensors", ".bin"}
    )


def _flatten_messages(row: dict[str, object]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("perplexity row has no messages")
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
    )


def _perplexity(model, tokenizer, rows, window_tokens: int) -> tuple[float, float]:
    import torch

    total_nll = 0.0
    total_tokens = 0
    for row in rows:
        encoded = tokenizer(
            _flatten_messages(row),
            return_tensors="pt",
            truncation=True,
            max_length=window_tokens,
        ).to(model.device)
        input_ids = encoded["input_ids"]
        if input_ids.shape[-1] < 2:
            continue
        with torch.inference_mode():
            output = model(**encoded, labels=input_ids)
        predicted_tokens = int(input_ids.shape[-1] - 1)
        total_nll += float(output.loss.detach().float().cpu()) * predicted_tokens
        total_tokens += predicted_tokens
    if total_tokens == 0:
        raise ValueError("perplexity corpus produced zero scored tokens")
    nll = total_nll / total_tokens
    return nll, math.exp(nll)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-label", choices=("bf16", "awq", "gptq"), required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load_object(args.config)
    _validate_inputs(config, args.repo_root)
    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("models config missing")
    model_dir = args.model_dir or Path(str(models[args.model_label]))
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if torch.cuda.device_count() != 1:
        raise RuntimeError("formal protocol requires exactly one visible GPU")
    set_seed(int(config["seed"]))
    torch.cuda.empty_cache()
    before_smi = _gpu_used_mib()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True
    )
    dtype = torch.bfloat16 if args.model_label == "bf16" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()
    static_allocated = torch.cuda.memory_allocated() / MIB
    static_smi_delta = max(0.0, _gpu_used_mib() - before_smi)
    torch.cuda.reset_peak_memory_stats()

    generation_section = dict(config["generation_eval"])
    generation_payload = _load_object(
        args.repo_root / str(generation_section["path"])
    )
    prompts = generation_payload.get("prompts")
    if not isinstance(prompts, list) or len(prompts) < 20:
        raise ValueError("generation benchmark requires twenty prompts")

    def generate(prompt: str, max_new_tokens: int) -> tuple[str, int, float]:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        output_ids = generated[0][encoded["input_ids"].shape[-1] :]
        return tokenizer.decode(output_ids, skip_special_tokens=True), len(output_ids), elapsed

    warmup_runs = int(generation_section["warmup_runs"])
    for index in range(warmup_runs):
        generate(str(prompts[index]["prompt"]), 32)

    output_tokens: list[int] = []
    durations: list[float] = []
    ttfts: list[float] = []
    responses: list[dict[str, object]] = []
    for index in range(int(generation_section["measured_runs"])):
        prompt_row = prompts[index]
        prompt = str(prompt_row["prompt"])
        _, _, ttft = generate(prompt, 1)
        text, tokens, duration = generate(
            prompt, int(generation_section["max_new_tokens"])
        )
        output_tokens.append(tokens)
        durations.append(duration)
        ttfts.append(ttft)
        responses.append(
            {
                "id": prompt_row["id"],
                "prompt": prompt,
                "response": text,
                "output_tokens": tokens,
                "duration_seconds": duration,
                "offline_first_token_seconds": ttft,
            }
        )

    perplexity_section = dict(config["perplexity_eval"])
    ppl_rows = [
        json.loads(line)
        for line in (
            args.repo_root / str(perplexity_section["path"])
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    nll, perplexity = _perplexity(
        model, tokenizer, ppl_rows, int(perplexity_section["window_tokens"])
    )
    timing = summarize_timings(output_tokens, durations, ttfts)
    receipt = {
        "schema_version": "1.0",
        "model": args.model_label,
        "model_dir": model_dir.as_posix(),
        "protocol_sha256": sha256_file(args.config),
        "config": portable_repo_path(args.config, args.repo_root),
        "weight_bytes": _weight_bytes(model_dir),
        "static_load_mib": max(static_allocated, static_smi_delta),
        "torch_static_allocated_mib": static_allocated,
        "nvidia_smi_static_delta_mib": static_smi_delta,
        "peak_mib": max(torch.cuda.max_memory_allocated() / MIB, _gpu_used_mib() - before_smi),
        **timing.as_dict(),
        "nll": nll,
        "perplexity": perplexity,
        "load_smoke_passed": True,
        "generation_smoke_passed": bool(responses and responses[0]["response"]),
        "responses": responses,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "autoawq": _package_version("autoawq"),
            "gptqmodel": _package_version("gptqmodel"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({"status": "PASS", "output": args.output.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
