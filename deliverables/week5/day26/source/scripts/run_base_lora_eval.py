#!/usr/bin/env python3
"""Run the frozen Day26 Base-vs-LoRA VLM comparison and make a blind A/B packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATEGORIES = ("natural_scene", "ocr", "chart_table", "ui", "formula")
EXPECTED_MODES = {"direct", "correction"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_adapter_identity(adapter_dir: Path, expected_sha256: str) -> str:
    weight_path = adapter_dir / "adapter_model.safetensors"
    if not weight_path.is_file():
        raise ValueError(f"adapter weight is missing: {weight_path}")
    actual = sha256(weight_path)
    if actual != expected_sha256:
        raise ValueError(
            f"adapter weight SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def is_ignorable_huggingface_cache_file(model_dir: Path, path: Path) -> bool:
    if path.is_symlink():
        return False
    relative = path.relative_to(model_dir).as_posix()
    if relative == ".cache/huggingface/.gitignore":
        return True
    return (
        path.suffix == ".metadata"
        and relative.startswith(".cache/huggingface/download/")
    )


def verify_base_model_identity(model_dir: Path, manifest_path: Path) -> dict[str, str]:
    if not model_dir.is_dir():
        raise ValueError(f"base model directory is missing: {model_dir}")
    if not manifest_path.is_file():
        raise ValueError(f"base model manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    repository = manifest.get("repository")
    revision = manifest.get("revision")
    files = manifest.get("files")
    if manifest.get("status") != "downloaded_verified":
        raise ValueError("base model manifest is not downloaded_verified")
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("base model manifest repository is missing")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision.lower())
    ):
        raise ValueError("base model manifest revision must be a 40-character commit SHA")
    if not isinstance(files, list) or not files:
        raise ValueError("base model manifest files must be a non-empty list")

    expected_paths: list[str] = []
    rebuilt_files: list[dict[str, Any]] = []
    for row in files:
        relative = row.get("relative_path") if isinstance(row, dict) else None
        if not isinstance(relative, str):
            raise ValueError("base model manifest contains an invalid relative path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe base model manifest path: {relative}")
        path = model_dir / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"base model file integrity failed: {relative}")
        rebuilt = {
            "relative_path": relative_path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if rebuilt["bytes"] != row.get("bytes") or rebuilt["sha256"] != row.get("sha256"):
            raise ValueError(f"base model file integrity failed: {relative}")
        expected_paths.append(relative_path.as_posix())
        rebuilt_files.append(rebuilt)

    actual_paths = sorted(
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file() and not is_ignorable_huggingface_cache_file(model_dir, path)
    )
    if sorted(expected_paths) != actual_paths or len(set(expected_paths)) != len(expected_paths):
        raise ValueError("base model directory does not exactly match the manifest file set")
    files_digest = hashlib.sha256(
        json.dumps(rebuilt_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        len(rebuilt_files) != manifest.get("file_count")
        or sum(row["bytes"] for row in rebuilt_files) != manifest.get("total_bytes")
        or files_digest != manifest.get("files_digest_sha256")
    ):
        raise ValueError("base model manifest aggregate integrity failed")
    return {
        "base_model_repository": repository,
        "base_model_revision": revision,
        "base_model_files_digest_sha256": files_digest,
        "base_model_manifest_sha256": sha256(manifest_path),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        rows.append(row)
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_eval_records(records: list[dict[str, Any]], split: str) -> None:
    if split not in {"dev", "final"}:
        raise ValueError("split must be dev or final")
    if len(records) != 20:
        raise ValueError(f"{split} comparison must contain exactly 20 cases")
    if len({row.get("id") for row in records}) != 20:
        raise ValueError("case IDs must be unique")
    category_counts = Counter(row.get("category") for row in records)
    if category_counts != Counter({category: 4 for category in CATEGORIES}):
        raise ValueError(f"category balance changed: {dict(category_counts)}")
    by_image: dict[str, set[str]] = defaultdict(set)
    for row in records:
        required = (
            "id",
            "split",
            "category",
            "mode",
            "source_image_id",
            "image_relative_path",
            "image_sha256",
            "user_instruction",
            "reference_answer",
        )
        missing = [field for field in required if not row.get(field)]
        if missing:
            raise ValueError(f"{row.get('id')} missing fields: {missing}")
        if row["split"] != split:
            raise ValueError(f"record {row['id']} does not belong to split {split}")
        by_image[row["source_image_id"]].add(row["mode"])
    if len(by_image) != 10 or any(modes != EXPECTED_MODES for modes in by_image.values()):
        raise ValueError("each of ten held-out images must have exactly two modes")


def reserve_final_run(
    path: Path, run_id: str, *, run_spec_sha256: str, resume: bool = False
) -> dict[str, Any]:
    if path.exists():
        current = read_json(path)
        if resume and current.get("run_id") == run_id:
            if current.get("run_spec_sha256") != run_spec_sha256:
                raise ValueError("final run spec does not match the reserved run")
            return current
        raise ValueError(f"final test is already reserved by run {current.get('run_id')}")
    payload = {
        "run_id": run_id,
        "run_spec_sha256": run_spec_sha256,
        "reserved_at": utc_now(),
        "max_runs": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload


def validate_resume_rows(
    rows: list[dict[str, Any]], run_id: str, run_spec_sha256: str
) -> None:
    mismatched = [
        row
        for row in rows
        if row.get("run_id") != run_id
        or row.get("run_spec_sha256") != run_spec_sha256
    ]
    if mismatched:
        raise ValueError("raw resume rows do not match the immutable run spec")


def build_run_spec_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_blind_packet(
    rows: list[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if len(rows) != 20:
        raise ValueError("blind packet requires exactly 20 paired cases")
    rng = random.Random(seed)
    base_first = set(rng.sample(range(len(rows)), k=len(rows) // 2))
    packet = []
    key = []
    for index, row in enumerate(rows):
        if index in base_first:
            a_model, b_model = "base", "lora"
        else:
            a_model, b_model = "lora", "base"
        packet.append(
            {
                "case_id": row["case_id"],
                "image_relative_path": row["image_relative_path"],
                "user_instruction": row["user_instruction"],
                "reference_answer": row["reference_answer"],
                "response_a": row[f"{a_model}_response"],
                "response_b": row[f"{b_model}_response"],
            }
        )
        key.append({"case_id": row["case_id"], "a_model": a_model, "b_model": b_model})
    return packet, key


def build_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def apply_lora_scale(model: Any, scale: float) -> int:
    if not 0 < scale <= 2:
        raise ValueError("LoRA scale must satisfy 0 < scale <= 2")
    changed = 0
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict):
            continue
        for adapter_name in list(scaling):
            scaling[adapter_name] *= scale
            changed += 1
    if changed == 0:
        raise RuntimeError("no LoRA scaling entries were found")
    return changed


def generate_one(
    *, model: Any, processor: Any, image_path: Path, prompt: str, generation: dict[str, Any]
) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    messages = build_messages(image_path, prompt)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    generation_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=generation["max_new_tokens"],
            do_sample=False,
            repetition_penalty=generation["repetition_penalty"],
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    if not response.strip():
        raise RuntimeError("model returned an empty response")
    return {
        "response": response,
        "input_tokens": int(inputs.input_ids.shape[-1]),
        "output_tokens": int(trimmed[0].shape[-1]),
        "generation_seconds": round(generation_seconds, 3),
        "latency_seconds": round(time.perf_counter() - started, 3),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    records = read_json(args.cases)
    if not isinstance(records, list):
        raise ValueError("cases file must contain a JSON list")
    validate_eval_records(records, args.split)
    base_identity = verify_base_model_identity(args.model_dir, args.base_model_manifest)
    adapter_model_sha256 = verify_adapter_identity(
        args.adapter_dir, args.expected_adapter_sha256
    )
    adapter_config_path = args.adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise ValueError(f"adapter config is missing: {adapter_config_path}")
    adapter_config_sha256 = sha256(adapter_config_path)
    config_sha256 = sha256(args.config)
    scoring_config_path = getattr(args, "scoring_config", None) or args.config
    scoring_config_sha256 = sha256(scoring_config_path)
    cases_sha256 = sha256(args.cases)
    run_spec_sha256 = build_run_spec_sha256(
        {
            "run_id": args.run_id,
            "split": args.split,
            "config_sha256": config_sha256,
            "scoring_config_sha256": scoring_config_sha256,
            "cases_sha256": cases_sha256,
            **base_identity,
            "adapter_config_sha256": adapter_config_sha256,
            "adapter_model_sha256": adapter_model_sha256,
            "lora_scale": args.lora_scale,
        }
    )
    if args.split == "final":
        if args.run_ledger is None:
            raise ValueError("final evaluation requires --run-ledger")
        reserve_final_run(
            args.run_ledger,
            args.run_id,
            run_spec_sha256=run_spec_sha256,
            resume=args.resume,
        )

    for row in records:
        image_path = args.data_root / row["image_relative_path"]
        if not image_path.is_file() or sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"image integrity failed: {row['id']}")

    existing = read_jsonl(args.raw_output)
    if existing and not args.resume:
        raise ValueError("raw output already exists; use --resume for the same run")
    if existing:
        validate_resume_rows(existing, args.run_id, run_spec_sha256)
    done = {(row.get("case_id"), row.get("model")) for row in existing}

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with bfloat16 support is required")
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generation = config["generation"]
    size = {
        "shortest_edge": int(generation["min_pixels"]),
        "longest_edge": int(generation["max_pixels"]),
    }
    processor = AutoProcessor.from_pretrained(
        args.model_dir, local_files_only=True, size=size
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()

    for model_label in ("base", "lora"):
        if model_label == "lora":
            model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
            scaled_module_count = apply_lora_scale(model, args.lora_scale)
            model.eval()
        for execution_index, item in enumerate(records, 1):
            if (item["id"], model_label) in done:
                continue
            metrics = generate_one(
                model=model,
                processor=processor,
                image_path=args.data_root / item["image_relative_path"],
                prompt=item["user_instruction"],
                generation=generation,
            )
            payload = {
                "schema_version": "1.0",
                "run_id": args.run_id,
                "run_spec_sha256": run_spec_sha256,
                "split": args.split,
                "execution_index": execution_index,
                "case_id": item["id"],
                "model": model_label,
                "lora_scale": args.lora_scale if model_label == "lora" else 0.0,
                "completed_at": utc_now(),
                **metrics,
            }
            append_jsonl(args.raw_output, payload)
            print(
                f"[{model_label} {execution_index:02d}/20] {item['id']} "
                f"tokens={payload['output_tokens']} latency={payload['latency_seconds']}s",
                flush=True,
            )

    all_raw = read_jsonl(args.raw_output)
    lookup = {(row["case_id"], row["model"]): row for row in all_raw}
    paired = []
    for item in records:
        base = lookup[(item["id"], "base")]
        lora = lookup[(item["id"], "lora")]
        paired.append(
            {
                "case_id": item["id"],
                "category": item["category"],
                "mode": item["mode"],
                "image_relative_path": item["image_relative_path"],
                "image_sha256": item["image_sha256"],
                "source": item.get("source", ""),
                "license": item.get("license", ""),
                "user_instruction": item["user_instruction"],
                "reference_answer": item["reference_answer"],
                "base_response": base["response"],
                "lora_response": lora["response"],
                "base_metrics": {key: value for key, value in base.items() if key.endswith("seconds") or key.endswith("tokens") or key.endswith("mib")},
                "lora_metrics": {key: value for key, value in lora.items() if key.endswith("seconds") or key.endswith("tokens") or key.endswith("mib")},
            }
        )
    write_json(args.paired_output, paired)
    packet, key = build_blind_packet(paired, seed=int(config["blind_seed"]))
    write_json(args.blind_packet, packet)
    write_json(args.blind_key, key)
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "run_id": args.run_id,
        "split": args.split,
        "completed_at": utc_now(),
        "case_count": len(paired),
        "generation_config_sha256": config_sha256,
        "scoring_config_sha256": scoring_config_sha256,
        "cases_sha256": cases_sha256,
        "run_spec_sha256": run_spec_sha256,
        **base_identity,
        "base_identity_check": "PRE_RUN_MANIFEST_SHA256_MATCH",
        "adapter_config_sha256": adapter_config_sha256,
        "adapter_model_sha256": adapter_model_sha256,
        "adapter_directory": str(args.adapter_dir),
        "adapter_identity_check": "PRE_RUN_SHA256_MATCH",
        "lora_scale": args.lora_scale,
        "scaled_module_count": scaled_module_count,
        "raw_record_count": len(all_raw),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    write_json(args.summary, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-model-manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scoring-config", type=Path)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--blind-packet", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--run-ledger", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("dev", "final"), required=True)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
