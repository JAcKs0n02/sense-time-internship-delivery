#!/usr/bin/env python3
"""Independently recompute Day30 evidence and semantic case results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage


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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _recorded_artifacts(output_directory: Path) -> dict[str, Any]:
    return {
        name: _load_json(output_directory / f"{name}.json")
        for name in (
            "model_preflight",
            "effective_config",
            "code_tool_prompt_manifest",
            "environment",
        )
    }


def _raw_provenance(messages: list[Any]) -> bool:
    from week6_agent.model_adapter import parse_model_output

    saw_final = False
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        raw = message.response_metadata.get("raw_generated_text")
        if not isinstance(raw, str) or not raw.strip():
            return False
        if message.tool_calls:
            reparsed = parse_model_output(raw, call_id_prefix="day30-validator")
            expected = [(call["name"], call["args"]) for call in message.tool_calls]
            actual = [(call["name"], call["args"]) for call in reparsed.tool_calls]
            if actual != expected:
                return False
        else:
            saw_final = True
            if raw != message.content:
                return False
    return saw_final


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result: dict[str, Any]
    try:
        from week6_agent.day30_agent import (
            build_day30_evidence_bundle,
            verify_day30_run_spec,
        )
        from week6_agent.day30_validation import validate_case_result
        from week6_agent.trace_schema import (
            deserialize_messages,
            extract_trace,
            final_answer,
        )

        config = _load_json(CONFIG_PATH)
        case_document = _load_json(CASES_PATH)
        cases = {case["case_id"]: case for case in case_document["cases"]}
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
        run_spec = _load_json(args.run_spec)
        run_spec_sha = verify_day30_run_spec(run_spec, current_payload)
        recorded = _recorded_artifacts(args.output_directory)
        rows = [
            json.loads(line)
            for line in (args.output_directory / "multistep_traces.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        case_results: list[dict[str, Any]] = []
        for row in rows:
            case = cases[row["case_id"]]
            messages = deserialize_messages(row.get("messages", []))
            recomputed_result = {
                "status": row.get("status"),
                "error_code": row.get("error_code"),
                "tool_names": row.get("tool_names"),
                "max_steps": row.get("max_steps"),
                "trace": extract_trace(messages),
                "final_answer": final_answer(messages),
            }
            semantic = validate_case_result(case, recomputed_result)
            binding_checks = {
                "prompt": bool(messages)
                and messages[0].type == "human"
                and messages[0].content == case["prompt"],
                "trace_recomputed": row.get("trace") == recomputed_result["trace"],
                "answer_recomputed": row.get("final_answer")
                == recomputed_result["final_answer"],
                "raw_model_provenance": _raw_provenance(messages),
                "run_spec": row.get("run_spec_sha256") == run_spec_sha,
                "model_identity": row.get("model_manifest_sha256")
                == current_artifacts["model_preflight"]["manifest_sha256"],
                "stored_validation": row.get("validation") == semantic,
                "side_effect_marker_absent": row.get(
                    "side_effect_marker_exists_after"
                )
                is False,
            }
            failed_bindings = [
                name for name, passed in binding_checks.items() if not passed
            ]
            case_results.append(
                {
                    "case_id": row["case_id"],
                    "status": (
                        "PASS"
                        if semantic["status"] == "PASS" and not failed_bindings
                        else "FAIL"
                    ),
                    "semantic_validation": semantic,
                    "binding_checks": binding_checks,
                    "failed_binding_checks": failed_bindings,
                }
            )
        checks = {
            "cases_frozen": case_document.get("frozen_before_formal_run") is True,
            "exact_case_set": {row["case_id"] for row in rows} == set(cases),
            "unique_case_rows": len(rows) == len({row["case_id"] for row in rows}),
            "initialization_artifacts": recorded == current_artifacts,
            "all_cases_pass": bool(case_results)
            and all(row["status"] == "PASS" for row in case_results),
            "teacher_case_pass": any(
                row["case_id"] == "teacher_nebula_budget" and row["status"] == "PASS"
                for row in case_results
            ),
            "side_effect_marker_absent": not MARKER_PATH.exists(),
        }
        failed = [name for name, passed in checks.items() if not passed]
        result = {
            "schema_version": "1.0",
            "status": "PASS" if not failed else "FAIL",
            "checks": checks,
            "failed_checks": failed,
            "case_results": case_results,
            "recomputed_run_spec_sha256": run_spec_sha,
            "recomputed_tools_sha256": current_artifacts[
                "code_tool_prompt_manifest"
            ]["tools_sha256"],
        }
    except Exception as error:
        result = {
            "schema_version": "1.0",
            "status": "FAIL",
            "failed_checks": ["validator_exception"],
            "error": str(error),
        }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "day30_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
