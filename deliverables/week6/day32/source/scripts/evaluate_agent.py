#!/usr/bin/env python3
"""Run the prompt-v2 Day32 evaluation against the unchanged Day31 SFT model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DAY32_SOURCE = Path(__file__).resolve().parents[1]
REPO_ROOT = DAY32_SOURCE.parents[3]
DAY31_SCRIPTS = DAY32_SOURCE.parents[1] / "day31/source/scripts"
sys.path.insert(0, str(REPO_ROOT / "deliverables/week6/source"))
sys.path.insert(0, str(DAY31_SCRIPTS))

from day31_harness import (  # noqa: E402
    WEEK4_MERGED_MANIFEST_SHA256,
    evaluate_frozen_cases,
    resolve_scheme_a_lineage,
    sha256_file,
)
from week6_agent.day30_agent import build_day30_react_agent  # noqa: E402
from week6_agent.day32_agent import (  # noqa: E402
    build_day32_react_agent,
    load_prompt_manifest,
    validate_prompt_selection_split,
)


def verify_day32_inference_lineage(
    preflight_path: Path,
    model_path: Path,
    day30_config_path: Path,
    *,
    resolver=resolve_scheme_a_lineage,
) -> dict[str, object]:
    """Revalidate model bytes while reusing Day31's already signed parser receipt.

    The LLaMA-Factory parser checkout is a training-data concern. Day32 changes only
    the inference prompt, so a later removal of that checkout's `.git` directory must
    not invalidate otherwise identical model bytes and evaluation inputs.
    """
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    lineage = preflight.get("lineage")
    receipt = preflight.get("official_parser_smoke")
    if preflight.get("status") != "ready" or not isinstance(lineage, dict):
        raise ValueError("Day31 preflight is not ready")
    if not isinstance(receipt, dict) or receipt.get("exit_code") != 0:
        raise ValueError("Day31 parser receipt is not signed as successful")
    manifest_path = lineage.get("merged_manifest_path")
    if not isinstance(manifest_path, str):
        raise ValueError("Day31 merged manifest path is missing")
    current = resolver(
        merged_manifest_path=Path(manifest_path),
        merged_model_dir=model_path,
        base_manifest_path=None,
        base_model_dir=None,
        dpo_adapter_dir=None,
        day30_config_path=day30_config_path,
    )
    if (
        current.get("merged_manifest_sha256") != WEEK4_MERGED_MANIFEST_SHA256
        or lineage.get("merged_manifest_sha256") != WEEK4_MERGED_MANIFEST_SHA256
        or Path(str(lineage.get("model_dir", ""))).resolve() != model_path.resolve()
    ):
        raise ValueError("Day32 model lineage differs from Day31")
    return preflight


def _serialise_result(result: dict[str, object]) -> dict[str, object]:
    messages = result.pop("messages", [])
    result["raw_messages"] = [
        {
            "type": getattr(message, "type", None),
            "content": str(getattr(message, "content", "")),
            "tool_calls": getattr(message, "tool_calls", []),
            "tool_call_id": getattr(message, "tool_call_id", None),
            "response_metadata": getattr(message, "response_metadata", {}),
        }
        for message in messages
    ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DAY32_SOURCE.parents[1] / "day31/source/data/tool_sft_dev_v2.json",
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=DAY32_SOURCE.parents[1] / "day31/source/data/knowledge_base_v2.json",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DAY32_SOURCE.parents[1] / "day31/source/data/dataset_manifest_v2.json",
    )
    parser.add_argument(
        "--prompt-main",
        type=Path,
        default=DAY32_SOURCE / "configs/system_prompt_main.txt",
    )
    parser.add_argument(
        "--prompt-ablation",
        type=Path,
        default=DAY32_SOURCE / "configs/system_prompt_ablation_input_fidelity.txt",
    )
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--prompt-variant", choices=("main", "ablation"), required=True)
    parser.add_argument("--policy-mode", choices=("raw", "guarded"), default="raw")
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        supplied_manifest = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
        current_manifest = load_prompt_manifest(args.prompt_main, args.prompt_ablation)
        if supplied_manifest != current_manifest:
            raise ValueError("prompt manifest identity mismatch")
        selection = validate_prompt_selection_split(
            args.eval_file, args.dataset_manifest
        )
        if not args.model_path.is_dir() or not args.adapter_path.is_dir():
            raise ValueError("model or adapter path missing")
        if not all(
            (args.adapter_path / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_config.json")
        ):
            raise ValueError("adapter files missing")
        preflight = verify_day32_inference_lineage(
            args.preflight,
            args.model_path,
            DAY32_SOURCE.parents[1] / "day30/source/configs/agent_config.json",
        )
        frozen = json.loads(args.eval_file.read_text(encoding="utf-8"))
        generation = {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": 1.0,
            "seed": args.seed,
        }
        protocol = {
            "evaluation_profile": "v2",
            "selection_split": selection,
            "max_steps": args.max_steps,
            "frozen_eval_file_sha256": sha256_file(args.eval_file),
            "knowledge_base_sha256": sha256_file(args.knowledge_base),
            "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
            "preflight_sha256": sha256_file(args.preflight),
            "policy_mode": args.policy_mode,
        }
        baseline = None
        if args.prompt_variant == "ablation":
            if args.compare_to is None:
                raise ValueError("ablation comparison requires --compare-to")
            baseline = json.loads(args.compare_to.read_text(encoding="utf-8"))
            if baseline.get("status") != "completed" or baseline.get("case_count") != len(frozen):
                raise ValueError("comparison baseline is not completed")
            if baseline.get("generation_config") != generation:
                raise ValueError("generation configuration differs from baseline")
            if baseline.get("protocol") != protocol:
                raise ValueError("development evaluation protocol differs from baseline")
            if baseline.get("prompt_variant") != "main":
                raise ValueError("comparison baseline must use the main Prompt")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2

    from week6_agent.model_adapter import HuggingFaceLocalChatModel

    model_identity = {
        "mode": "post",
        "model_path": str(args.model_path.resolve()),
        "model_path_sha256": sha256_file(args.model_path / "config.json"),
        "adapter_path": str(args.adapter_path.resolve()),
        "adapter_model_sha256": sha256_file(args.adapter_path / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(args.adapter_path / "adapter_config.json"),
    }
    if baseline is not None and model_identity != baseline.get("model_identity"):
        print("model identity differs from Day31 baseline", file=sys.stderr)
        return 2
    model = HuggingFaceLocalChatModel.from_local_model(
        str(args.model_path), adapter_path=str(args.adapter_path), **generation
    )
    factory = (
        build_day30_react_agent
        if args.prompt_variant == "main"
        else build_day32_react_agent
    )
    agent = factory(
        model,
        args.knowledge_base,
        max_steps=args.max_steps,
        runtime_limit_seconds=60,
        policy_mode=args.policy_mode,
    )
    report = evaluate_frozen_cases(
        frozen,
        lambda row: _serialise_result(agent.invoke(row["conversations"][0]["value"])),
        generation,
        model_identity,
        protocol,
        knowledge_base_path=args.knowledge_base,
    )
    report["prompt_manifest"] = current_manifest
    report["prompt_variant"] = args.prompt_variant
    report["policy_mode"] = args.policy_mode
    report["lineage"] = preflight["lineage"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
