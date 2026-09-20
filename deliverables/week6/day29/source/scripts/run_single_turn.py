#!/usr/bin/env python3
"""Run the one fixed Day29 case with the verified local DPO model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DAY29_SOURCE = Path(__file__).resolve().parents[1]
WEEK6_SOURCE = DAY29_SOURCE.parents[1] / "source"
REPO_ROOT = DAY29_SOURCE.parents[3]
CONFIG_PATH = DAY29_SOURCE / "configs" / "agent_config.json"
KNOWLEDGE_BASE = DAY29_SOURCE.parents[1] / "day28/source/data/knowledge_base.json"
sys.path.insert(0, str(WEEK6_SOURCE))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-model-path", type=Path, required=True)
    parser.add_argument("--remote-adapter-path", type=Path, required=True)
    parser.add_argument("--expected-source-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--run-spec", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from week6_agent.agent_factory import (
            build_formal_evidence_bundle,
            build_react_agent,
            verify_run_spec_document,
        )
        from week6_agent.model_adapter import HuggingFaceLocalChatModel
        from week6_agent.trace_schema import serialize_message

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        current_artifacts, current_payload, _ = build_formal_evidence_bundle(
            config=config,
            config_path=CONFIG_PATH,
            model_path=args.remote_model_path,
            adapter_path=args.remote_adapter_path,
            source_manifest_path=args.expected_source_manifest,
            repo_root=REPO_ROOT,
            knowledge_base_path=KNOWLEDGE_BASE,
        )
        run_spec_document = json.loads(args.run_spec.read_text(encoding="utf-8"))
        run_spec_sha256 = verify_run_spec_document(run_spec_document, current_payload)
        recorded_artifacts = {
            "model_preflight": json.loads(
                (args.output_directory / "model_preflight.json").read_text(encoding="utf-8")
            ),
            "effective_config": json.loads(
                (args.output_directory / "effective_config.json").read_text(encoding="utf-8")
            ),
            "code_tool_prompt_manifest": json.loads(
                (args.output_directory / "code_tool_prompt_manifest.json").read_text(
                    encoding="utf-8"
                )
            ),
            "environment": json.loads(
                (args.output_directory / "environment.json").read_text(encoding="utf-8")
            ),
        }
        if recorded_artifacts != current_artifacts:
            raise ValueError("initialization artifacts mismatch current run_spec inputs")
        model = HuggingFaceLocalChatModel.from_local_model(
            str(args.remote_model_path),
            adapter_path=str(args.remote_adapter_path),
            **config["generation"],
        )
        agent = build_react_agent(
            model,
            KNOWLEDGE_BASE,
            max_steps=config["max_steps"],
            runtime_limit_seconds=config["runtime_limit_seconds"],
        )
        if agent.recursion_limit != current_artifacts["effective_config"]["recursion_limit"]:
            raise ValueError("compiled agent recursion limit mismatch")
        result = agent.invoke(config["formal_prompt"])
        artifact = {
            "schema_version": "1.0",
            "model_id": config["model_id"],
            "model_manifest_sha256": current_artifacts["model_preflight"][
                "manifest_sha256"
            ],
            "run_spec_sha256": run_spec_sha256,
            "prompt": config["formal_prompt"],
            "runner_evidence": current_artifacts,
            "messages": [serialize_message(message) for message in result.pop("messages")],
            **result,
        }
        args.output_directory.mkdir(parents=True, exist_ok=True)
        (args.output_directory / "single_turn_trace.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0 if artifact["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
