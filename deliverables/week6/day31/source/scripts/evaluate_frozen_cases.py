#!/usr/bin/env python3
"""Run one deterministic frozen-eval pass against a local base or PEFT adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from day31_harness import evaluate_frozen_cases, sha256_file, verify_ready_merged_preflight


DAY31 = Path(__file__).resolve().parents[1]
REPO_ROOT = DAY31.parents[3]
sys.path.insert(0, str(REPO_ROOT / "deliverables/week6/source"))


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
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--mode", choices=("baseline", "post"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-to", type=Path, help="Baseline/post result whose generation config must match this run.")
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--evaluation-profile", choices=("legacy", "v2"), default="legacy")
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--knowledge-base", type=Path)
    parser.add_argument("--policy-mode", choices=("raw", "guarded"), default="raw")
    parser.add_argument(
        "--split-role",
        choices=("legacy_frozen", "development", "final_test"),
        default="legacy_frozen",
    )
    args = parser.parse_args()
    if args.mode == "baseline" and args.adapter_path is not None:
        print("baseline must not receive a Day31 adapter", file=sys.stderr)
        return 2
    if args.mode == "post" and args.adapter_path is None:
        print("post evaluation requires a Day31 adapter", file=sys.stderr)
        return 2
    if args.evaluation_profile == "v2" and args.split_role == "legacy_frozen":
        print("v2 evaluation requires an explicit development or final_test split role", file=sys.stderr)
        return 2
    if args.evaluation_profile == "legacy" and args.split_role != "legacy_frozen":
        print("legacy evaluation must use the legacy_frozen split role", file=sys.stderr)
        return 2
    if not args.model_path.is_dir() or (args.adapter_path and not args.adapter_path.is_dir()):
        print("model or adapter path missing", file=sys.stderr)
        return 2
    if args.adapter_path and not all((args.adapter_path / name).is_file() for name in ("adapter_model.safetensors", "adapter_config.json")):
        print("post adapter files missing", file=sys.stderr)
        return 2
    if not (args.model_path / "config.json").is_file():
        print("model config.json missing", file=sys.stderr)
        return 2
    try:
        preflight = verify_ready_merged_preflight(args.preflight, args.model_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"preflight mismatch; materialize verified merged model: {error}", file=sys.stderr)
        return 2
    lineage = preflight["lineage"]
    from week6_agent.day30_agent import build_day30_react_agent
    from week6_agent.model_adapter import HuggingFaceLocalChatModel

    config = {"do_sample": False, "max_new_tokens": args.max_new_tokens, "repetition_penalty": 1.0, "seed": args.seed}
    if args.evaluation_profile == "v2":
        eval_path = args.eval_file
        knowledge_base = args.knowledge_base
        manifest_path = DAY31 / "data/dataset_manifest_v2.json"
        if eval_path is None or knowledge_base is None:
            print("v2 evaluation requires --eval-file and --knowledge-base", file=sys.stderr)
            return 2
    else:
        eval_path = args.eval_file or DAY31 / "data/tool_sft_eval_frozen.json"
        knowledge_base = args.knowledge_base or DAY31.parents[1] / "day28/source/data/knowledge_base.json"
        manifest_path = DAY31 / "data/dataset_manifest.json"
    if not eval_path.is_file() or not knowledge_base.is_file() or not manifest_path.is_file():
        print("evaluation data, knowledge base, or manifest missing", file=sys.stderr)
        return 2
    rows = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        print("evaluation file must contain a non-empty JSON list", file=sys.stderr)
        return 2
    protocol = {
        "evaluation_profile": args.evaluation_profile,
        "split_role": args.split_role,
        "policy_mode": args.policy_mode,
        "max_steps": args.max_steps,
        "frozen_eval_file_sha256": sha256_file(eval_path),
        "knowledge_base_sha256": sha256_file(knowledge_base),
        "tool_sources_sha256": sha256_file(manifest_path),
        "day30_config_sha256": sha256_file(DAY31.parents[1] / "day30/source/configs/agent_config.json"),
        "base_model_config_sha256": sha256_file(args.model_path / "config.json"),
        "preflight_sha256": sha256_file(args.preflight),
        "lineage": lineage,
    }
    if args.compare_to:
        previous = json.loads(args.compare_to.read_text(encoding="utf-8"))
        if previous.get("status") != "completed" or previous.get("model_identity", {}).get("mode") != "baseline" or previous.get("case_count") != len(rows) or previous.get("generation_config") != config or previous.get("protocol") != protocol:
            print("comparison run is not a compatible completed baseline", file=sys.stderr)
            return 2
    model = HuggingFaceLocalChatModel.from_local_model(str(args.model_path), adapter_path=str(args.adapter_path) if args.adapter_path else None, **config)
    agent = build_day30_react_agent(
        model,
        knowledge_base,
        max_steps=args.max_steps,
        policy_mode=args.policy_mode,
    )
    report = evaluate_frozen_cases(
        rows,
        lambda row: _serialise_result(agent.invoke(row["conversations"][0]["value"])),
        config,
        {"mode": args.mode, "model_path": str(args.model_path.resolve()), "model_path_sha256": sha256_file(args.model_path / "config.json"), "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None, "adapter_model_sha256": sha256_file(args.adapter_path / "adapter_model.safetensors") if args.adapter_path and (args.adapter_path / "adapter_model.safetensors").is_file() else None, "adapter_config_sha256": sha256_file(args.adapter_path / "adapter_config.json") if args.adapter_path and (args.adapter_path / "adapter_config.json").is_file() else None}, protocol,
        knowledge_base_path=knowledge_base,
    )
    report["policy_mode"] = args.policy_mode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
