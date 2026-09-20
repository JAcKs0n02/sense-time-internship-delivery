#!/usr/bin/env python3
"""Run every frozen Day30 case with one verified local DPO runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DAY30_SOURCE = Path(__file__).resolve().parents[1]
WEEK6_SOURCE = DAY30_SOURCE.parents[1] / "source"
REPO_ROOT = DAY30_SOURCE.parents[3]
CONFIG_PATH = DAY30_SOURCE / "configs/agent_config.json"
CASES_PATH = DAY30_SOURCE / "data/multistep_cases.json"
KNOWLEDGE_BASE = DAY30_SOURCE.parents[1] / "day28/source/data/knowledge_base.json"
MARKER_PATH = Path("/tmp/week6_day30_must_not_exist")
sys.path.insert(0, str(WEEK6_SOURCE))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-model-path", type=Path, required=True)
    parser.add_argument("--remote-adapter-path", type=Path, required=True)
    parser.add_argument("--expected-source-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--run-spec", type=Path, required=True)
    return parser.parse_args(argv)


def _recorded_artifacts(output_directory: Path) -> dict[str, Any]:
    return {
        name: json.loads((output_directory / f"{name}.json").read_text(encoding="utf-8"))
        for name in (
            "model_preflight",
            "effective_config",
            "code_tool_prompt_manifest",
            "environment",
        )
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from week6_agent.day30_agent import (
            build_day30_evidence_bundle,
            build_day30_react_agent,
            verify_day30_run_spec,
        )
        from week6_agent.day30_validation import validate_case_result
        from week6_agent.model_adapter import HuggingFaceLocalChatModel
        from week6_agent.trace_schema import serialize_message

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        case_document = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        if case_document.get("frozen_before_formal_run") is not True:
            raise ValueError("Day30 cases are not frozen")
        current_artifacts, current_payload, _ = build_day30_evidence_bundle(
            config=config,
            config_path=CONFIG_PATH,
            model_path=args.remote_model_path,
            adapter_path=args.remote_adapter_path,
            source_manifest_path=args.expected_source_manifest,
            repo_root=REPO_ROOT,
            knowledge_base_path=KNOWLEDGE_BASE,
            cases_path=CASES_PATH,
        )
        run_spec_document = json.loads(args.run_spec.read_text(encoding="utf-8"))
        run_spec_sha256 = verify_day30_run_spec(run_spec_document, current_payload)
        if _recorded_artifacts(args.output_directory) != current_artifacts:
            raise ValueError("Day30 initialization artifacts mismatch current inputs")
        if MARKER_PATH.exists():
            raise ValueError(f"side-effect marker already exists: {MARKER_PATH}")

        model = HuggingFaceLocalChatModel.from_local_model(
            str(args.remote_model_path),
            adapter_path=str(args.remote_adapter_path),
            **config["generation"],
        )
        artifacts: list[dict[str, Any]] = []
        for case in case_document["cases"]:
            try:
                agent = build_day30_react_agent(
                    model,
                    KNOWLEDGE_BASE,
                    max_steps=config["max_steps"],
                    runtime_limit_seconds=config["runtime_limit_seconds"],
                )
                result = agent.invoke(case["prompt"])
                messages = result.pop("messages")
                validation = validate_case_result(case, result)
                artifact = {
                    "schema_version": "1.0",
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "prompt": case["prompt"],
                    "model_id": config["model_id"],
                    "model_manifest_sha256": current_artifacts["model_preflight"][
                        "manifest_sha256"
                    ],
                    "run_spec_sha256": run_spec_sha256,
                    "messages": [serialize_message(message) for message in messages],
                    "side_effect_marker_exists_after": MARKER_PATH.exists(),
                    **result,
                    "validation": validation,
                }
            except Exception as error:
                artifact = {
                    "schema_version": "1.0",
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "prompt": case["prompt"],
                    "model_id": config["model_id"],
                    "model_manifest_sha256": current_artifacts["model_preflight"][
                        "manifest_sha256"
                    ],
                    "run_spec_sha256": run_spec_sha256,
                    "status": "failed",
                    "error_code": "case_exception",
                    "error": str(error),
                    "messages": [],
                    "trace": [],
                    "final_answer": "",
                    "tool_names": list(config["tool_names"]),
                    "max_steps": config["max_steps"],
                    "recursion_limit": config["max_steps"] * 2 + 4,
                    "runtime_limit_seconds": config["runtime_limit_seconds"],
                    "side_effect_marker_exists_after": MARKER_PATH.exists(),
                    "validation": {
                        "case_id": case["case_id"],
                        "status": "FAIL",
                        "checks": {"case_execution": False},
                        "failed_checks": ["case_execution"],
                        "auditable_stage_count": 0,
                    },
                }
            artifacts.append(artifact)

        args.output_directory.mkdir(parents=True, exist_ok=True)
        trace_path = args.output_directory / "multistep_traces.jsonl"
        trace_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in artifacts),
            encoding="utf-8",
        )
        passed = sum(row["validation"]["status"] == "PASS" for row in artifacts)
        summary = {
            "schema_version": "1.0",
            "status": "PASS" if passed == len(artifacts) else "FAIL",
            "case_count": len(artifacts),
            "passed": passed,
            "failed": len(artifacts) - passed,
            "pass_rate": passed / len(artifacts) if artifacts else 0,
            "teacher_case_status": next(
                row["validation"]["status"]
                for row in artifacts
                if row["case_id"] == "teacher_nebula_budget"
            ),
            "side_effect_marker_absent": not MARKER_PATH.exists(),
            "cases": [
                {
                    "case_id": row["case_id"],
                    "status": row["validation"]["status"],
                    "failed_checks": row["validation"]["failed_checks"],
                }
                for row in artifacts
            ],
        }
        _write_json(args.output_directory / "multistep_summary.json", summary)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
