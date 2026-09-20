#!/usr/bin/env python3
"""Independently recompute model, code/tool, and raw-message Day29 evidence."""

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
    result: dict[str, object]
    try:
        from week6_agent.agent_factory import (
            build_formal_evidence_bundle,
            evidence_binding_checks,
        )
        from week6_agent.trace_schema import deserialize_messages, validate_formal_messages

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        trace_artifact = json.loads(
            (args.output_directory / "single_turn_trace.json").read_text(encoding="utf-8")
        )
        current_artifacts, current_payload, current_document = build_formal_evidence_bundle(
            config=config,
            config_path=CONFIG_PATH,
            model_path=args.remote_model_path,
            adapter_path=args.remote_adapter_path,
            source_manifest_path=args.expected_source_manifest,
            repo_root=REPO_ROOT,
            knowledge_base_path=KNOWLEDGE_BASE,
        )
        run_spec_document = json.loads(args.run_spec.read_text(encoding="utf-8"))
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
        message_validation = validate_formal_messages(
            deserialize_messages(trace_artifact.get("messages", []))
        )
        binding_checks = evidence_binding_checks(
            trace_artifact,
            run_spec_document,
            current_payload,
            recorded_artifacts,
            current_artifacts,
        )
        checks = {
            **binding_checks,
            "adapter_sha256": current_artifacts["model_preflight"]["adapter"][
                "sha256"
            ]
            == config["expected_adapter_sha256"],
            "archived_merged_lineage": current_artifacts["model_preflight"][
                "archived_merged_manifest_sha256"
            ]
            == config["archived_merged_manifest_sha256"],
            "model_id": trace_artifact.get("model_id") == config["model_id"],
            "message_evidence": message_validation["status"] == "PASS",
        }
        failed = [name for name, passed in checks.items() if not passed]
        result = {
            "schema_version": "1.0",
            "status": "PASS" if not failed else "FAIL",
            "checks": checks,
            "failed_checks": failed,
            "message_validation": message_validation,
            "recomputed_model_manifest_sha256": current_artifacts[
                "model_preflight"
            ]["manifest_sha256"],
            "recomputed_tools_sha256": current_artifacts[
                "code_tool_prompt_manifest"
            ]["tools_sha256"],
            "recomputed_run_spec_sha256": current_document["run_spec_sha256"],
        }
    except Exception as error:
        result = {
            "schema_version": "1.0",
            "status": "FAIL",
            "failed_checks": ["validator_exception"],
            "error": str(error),
        }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "day29_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
