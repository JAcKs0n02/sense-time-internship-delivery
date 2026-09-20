#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


EXPECTED_SOURCE_SHA256 = "841aeb528f3613c789b14f13b1e550e17f5acdd6fb62f71ab30fd7576be11ca8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_prompt_list(payload: dict[str, Any], kind: str, expected: int) -> list[dict[str, str]]:
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != expected:
        raise ValueError(f"{kind} prompts must contain exactly {expected} records")
    ids: list[str] = []
    for position, item in enumerate(prompts, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{kind} prompt {position} must be an object")
        prompt_id = item.get("id")
        text = item.get("text")
        if item.get("prompt_type") != kind:
            raise ValueError(f"{prompt_id or position} has wrong prompt_type")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"{kind} prompt {position} has invalid id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{prompt_id} has empty text")
        ids.append(prompt_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{kind} prompt ids must be unique")
    return prompts


def validate_generation(config: dict[str, Any]) -> None:
    generation = config.get("generation_config")
    if not isinstance(generation, dict):
        raise ValueError("generation_config must be an object")
    expected = {
        "do_sample": False,
        "max_new_tokens": 512,
        "repetition_penalty": 1.0,
        "seed": 42,
        "dtype": "bfloat16",
        "device": "cuda:0",
    }
    for key, value in expected.items():
        if generation.get(key) != value:
            raise ValueError(f"generation_config.{key} must equal {value!r}")
    if config.get("system_message") is not None:
        raise ValueError("system_message must be null for the frozen comparison")


def validate_inputs(
    source_path: Path,
    config_path: Path,
    safety_path: Path,
    business_path: Path,
) -> dict[str, Any]:
    actual_hash = sha256(source_path)
    if actual_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"source SHA-256 mismatch: {actual_hash} != {EXPECTED_SOURCE_SHA256}"
        )
    source = load_object(source_path)
    config = load_object(config_path)
    safety_payload = load_object(safety_path)
    business_payload = load_object(business_path)
    validate_generation(config)
    if config.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("evaluation config source SHA-256 does not match frozen source")
    for name, payload in (("safety", safety_payload), ("business", business_payload)):
        if payload.get("source_sha256") != EXPECTED_SOURCE_SHA256:
            raise ValueError(f"{name} subset source SHA-256 does not match frozen source")
        if payload.get("training_allowed") is not False:
            raise ValueError(f"{name} subset must set training_allowed=false")
    safety = validate_prompt_list(safety_payload, "safety", 10)
    business = validate_prompt_list(business_payload, "business", 5)
    all_prompts = safety + business
    all_ids = [item["id"] for item in all_prompts]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("all evaluation prompt ids must be unique")
    if source.get("prompts") != all_prompts:
        raise ValueError("Day 20 prompt subsets do not exactly reproduce the frozen source")
    counts = config.get("expected_counts")
    if counts != {"safety": 10, "business": 5, "per_model": 15, "total_responses": 30}:
        raise ValueError("expected_counts must freeze the 10/5/15/30 contract")
    return {
        "valid": True,
        "source_sha256": actual_hash,
        "counts": {"safety": 10, "business": 5, "total": 15},
        "prompt_ids": all_ids,
        "generation_config": config["generation_config"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen Week 4 Day 20 inputs.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--safety", type=Path, required=True)
    parser.add_argument("--business", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_inputs(args.source, args.config, args.safety, args.business)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
