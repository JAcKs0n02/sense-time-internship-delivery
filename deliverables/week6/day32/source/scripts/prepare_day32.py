#!/usr/bin/env python3
"""Build the local Day32 gate from exact Day31 evaluation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "deliverables/week6/source"))

from week6_agent.day32_agent import load_prompt_manifest  # noqa: E402


EXPECTED_ADAPTER_SHA256 = (
    "7d7672cbbe7679d5badb20b040870b616e17c7093813baeb6605ee498925623c"
)
EXPECTED_GENERATION = {
    "do_sample": False,
    "max_new_tokens": 192,
    "repetition_penalty": 1,
    "seed": 42,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--day31-frozen", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prompt-v1", type=Path, required=True)
    parser.add_argument("--prompt-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        frozen_sha = _sha256(args.frozen)
        if args.frozen.read_bytes() != args.day31_frozen.read_bytes():
            raise ValueError("Day32 frozen evaluation differs from Day31")
        frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        prompt_manifest = load_prompt_manifest(args.prompt_v1, args.prompt_v2)
        if not isinstance(frozen, list) or len(frozen) != 30:
            raise ValueError("frozen evaluation must contain exactly 30 cases")
        if baseline.get("status") != "completed" or baseline.get("case_count") != 30:
            raise ValueError("Day31 SFT baseline is not a completed 30-case report")
        if baseline.get("generation_config") != EXPECTED_GENERATION:
            raise ValueError("Day31 generation configuration changed")
        if baseline.get("protocol", {}).get("frozen_eval_file_sha256") != frozen_sha:
            raise ValueError("Day31 baseline is not bound to the frozen evaluation")
        adapter_sha = baseline.get("model_identity", {}).get("adapter_model_sha256")
        if adapter_sha != EXPECTED_ADAPTER_SHA256:
            raise ValueError("Day31 adapter identity changed")
        report = {
            "schema_version": "1.0",
            "status": "ready",
            "case_count": len(frozen),
            "frozen_eval_sha256": frozen_sha,
            "day31_frozen_sha256": _sha256(args.day31_frozen),
            "baseline_sha256": _sha256(args.baseline),
            "adapter_sha256": adapter_sha,
            "generation_config": baseline["generation_config"],
            "protocol": baseline["protocol"],
            "prompt_manifest": prompt_manifest,
        }
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
