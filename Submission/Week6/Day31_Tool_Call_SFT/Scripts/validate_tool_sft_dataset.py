#!/usr/bin/env python3
"""Validate the Day31 ShareGPT tool-call data without a model or GPU."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DAY31_SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DAY31_SOURCE / "scripts"))
from build_tool_sft_dataset import (  # noqa: E402
    EXPECTED_TRAIN_COUNTS,
    KNOWLEDGE_BASE,
    KNOWLEDGE_BASE_V2,
    build_eval_rows,
    build_dataset_bundle,
    build_manifest,
    build_train_rows,
    canonical_json,
    execute_tool,
    tool_definitions,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _roles_valid(conversations: object) -> bool:
    if not isinstance(conversations, list) or len(conversations) < 4:
        return False
    roles = [message.get("from") for message in conversations if isinstance(message, dict)]
    if len(roles) != len(conversations) or roles[0] != "human" or roles[-1] != "gpt":
        return False
    return roles == ["human", *[role for _ in range((len(roles) - 2) // 2) for role in ("function_call", "observation")], "gpt"]


def _arguments_valid(name: str, arguments: object) -> bool:
    required = {
        "calculator": {"expression": str},
        "knowledge_retrieval": {"query": str},
        "code_executor": {"source": str},
    }
    expected = required.get(name)
    return (
        expected is not None
        and isinstance(arguments, dict)
        and set(arguments) == set(expected)
        and all(isinstance(arguments[key], value_type) for key, value_type in expected.items())
    )


def _tools_schema_valid(value: object) -> bool:
    try:
        return isinstance(value, str) and canonical_json(json.loads(value)) == canonical_json(tool_definitions())
    except (TypeError, json.JSONDecodeError):
        return False


def _decoded_calls(conversations: object) -> list[tuple[str, dict[str, Any], dict[str, Any]]] | None:
    if not _roles_valid(conversations):
        return None
    calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for index in range(1, len(conversations) - 1, 2):
        try:
            call_value = json.loads(conversations[index]["value"])
            observation = json.loads(conversations[index + 1]["value"])
        except (KeyError, TypeError, json.JSONDecodeError):
            return None
        name, arguments = call_value.get("name"), call_value.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict) or not isinstance(observation, dict):
            return None
        calls.append((name, arguments, observation))
    return calls


def _validate_row(
    row: object,
    registered_tools: set[str],
    knowledge_base_path: Path = KNOWLEDGE_BASE,
) -> list[str]:
    if not isinstance(row, dict):
        return ["row_not_object"]
    errors: list[str] = []
    conversations = row.get("conversations")
    if not _roles_valid(conversations):
        return ["role_alternation"]
    if not _tools_schema_valid(row.get("tools")):
        errors.append("tools_schema_mismatch")
    audit_react = row.get("audit", {}).get("react") if isinstance(row.get("audit"), dict) else None
    audit = row.get("audit") if isinstance(row.get("audit"), dict) else {}
    if audit.get("prompt") != conversations[0].get("value"):
        errors.append("audit_prompt_projection")
    if audit.get("final") != conversations[-1].get("value"):
        errors.append("audit_final_projection")
    if not isinstance(audit_react, list) or len(audit_react) != (len(conversations) - 2) // 2:
        errors.append("audit_trace_shape")
        audit_react = []
    for index in range(1, len(conversations) - 1, 2):
        call, observation_message = conversations[index], conversations[index + 1]
        try:
            call_value = json.loads(call["value"])
            observation = json.loads(observation_message["value"])
        except (KeyError, TypeError, json.JSONDecodeError):
            errors.append(f"json_message_{index}")
            continue
        name, arguments = call_value.get("name"), call_value.get("arguments")
        if name not in registered_tools:
            errors.append(f"unregistered_tool_{index}")
            continue
        if not _arguments_valid(name, arguments):
            errors.append(f"invalid_arguments_{index}")
            continue
        if canonical_json(execute_tool(name, arguments, knowledge_base_path)) != canonical_json(observation):
            errors.append(f"nonreproducible_observation_{index}")
        audit_index = (index - 1) // 2
        if audit_index < len(audit_react):
            audit = audit_react[audit_index]
            if not isinstance(audit, dict) or {
                "Thought", "Action", "Action Input", "Observation"
            } - set(audit):
                errors.append(f"audit_fields_{index}")
            elif (
                not isinstance(audit["Thought"], str)
                or not audit["Thought"].strip()
                or len(audit["Thought"].strip()) > 80
            ):
                errors.append(f"audit_thought_{index}")
            elif audit["Action"] != name or audit["Action Input"] != arguments or canonical_json(audit["Observation"]) != canonical_json(observation):
                errors.append(f"audit_projection_{index}")
    return errors


def _prompt(row: dict[str, Any]) -> str:
    return row["conversations"][0]["value"]


def _prompt_contamination(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> tuple[int, int]:
    exact = 0
    near = 0
    for train in train_rows:
        for evaluation in eval_rows:
            train_prompt, eval_prompt = _prompt(train), _prompt(evaluation)
            if train_prompt == eval_prompt:
                exact += 1
            elif difflib.SequenceMatcher(a=train_prompt, b=eval_prompt, autojunk=False).ratio() >= 0.80:
                near += 1
    return exact, near


def _semantic_call(name: str, arguments: dict[str, Any], observation: dict[str, Any]) -> str:
    """Fingerprint tool semantics rather than user wording or JSON formatting."""
    if name == "calculator":
        expression = arguments.get("expression", "")
        try:
            return f"calculator:{ast.dump(ast.parse(expression, mode='eval'), include_attributes=False)}"
        except (SyntaxError, TypeError, ValueError):
            return f"calculator_invalid:{' '.join(str(expression).split())}"
    if name == "knowledge_retrieval":
        data = observation.get("data", {})
        if observation.get("status") == "success":
            return f"knowledge_product:{data.get('product_id')}"
        if observation.get("status") == "ambiguous":
            candidates = sorted(candidate.get("product_id") for candidate in data.get("candidates", []) if isinstance(candidate, dict))
            return f"knowledge_ambiguous:{canonical_json(candidates)}"
        return f"knowledge_{observation.get('status')}:{' '.join(str(arguments.get('query', '')).casefold().split())}"
    source = arguments.get("source", "")
    try:
        return f"code:{ast.dump(ast.parse(source, mode='exec'), include_attributes=False)}"
    except (SyntaxError, TypeError, ValueError):
        return f"code_invalid:{' '.join(str(source).split())}"


def _semantic_case_fingerprints(rows: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    fingerprints: set[tuple[str, ...]] = set()
    for row in rows:
        calls = _decoded_calls(row.get("conversations"))
        if calls is not None:
            fingerprints.add(tuple(_semantic_call(name, arguments, observation) for name, arguments, observation in calls))
    return fingerprints


def _recovery_termination_valid(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("category") != "recovery_or_termination":
            continue
        calls = _decoded_calls(row.get("conversations"))
        if calls is None or len(calls) != 1 or calls[0][2].get("status") == "success":
            return False
    return True


def _special_case_sequences_valid(eval_rows: list[dict[str, Any]]) -> tuple[bool, bool]:
    by_id = {row.get("id"): row for row in eval_rows}
    zero_shipping = by_id.get("eval-aurora-zero-shipping-calculator-required")
    code_only = by_id.get("eval-code-no-unnecessary-knowledge")
    zero_calls = _decoded_calls(zero_shipping.get("conversations")) if isinstance(zero_shipping, dict) else None
    code_calls = _decoded_calls(code_only.get("conversations")) if isinstance(code_only, dict) else None
    zero_valid = bool(
        zero_calls
        and [call[0] for call in zero_calls] == ["knowledge_retrieval", "calculator"]
        and zero_calls[0][2].get("data", {}).get("shipping_cny") == 0
    )
    code_valid = bool(code_calls and [call[0] for call in code_calls] == ["code_executor"])
    return zero_valid, code_valid


def validate_dataset(data_directory: Path) -> dict[str, Any]:
    train_path = data_directory / "tool_sft_train_100.json"
    eval_path = data_directory / "tool_sft_eval_frozen.json"
    manifest_path = data_directory / "dataset_manifest.json"
    train_rows, eval_rows, manifest = _load(train_path), _load(eval_path), _load(manifest_path)
    errors: list[str] = []
    if not isinstance(train_rows, list) or not isinstance(eval_rows, list) or not isinstance(manifest, dict):
        raise ValueError("dataset files must contain train list, eval list, and manifest object")
    train_ids = [row.get("id") for row in train_rows if isinstance(row, dict)]
    eval_ids = [row.get("id") for row in eval_rows if isinstance(row, dict)]
    registered_tools = {definition["name"] for definition in tool_definitions()}
    row_errors = [error for row in [*train_rows, *eval_rows] for error in _validate_row(row, registered_tools)]
    errors.extend(row_errors)
    exact_overlap, near_overlap = _prompt_contamination(train_rows, eval_rows)
    train_counts = Counter(row.get("category") for row in train_rows if isinstance(row, dict))
    rebuild_train, rebuild_eval = build_train_rows(), build_eval_rows()
    expected_manifest = build_manifest(rebuild_train, rebuild_eval, data_directory)
    semantic_overlap = _semantic_case_fingerprints(train_rows).intersection(_semantic_case_fingerprints(eval_rows))
    zero_shipping_valid, code_only_valid = _special_case_sequences_valid(eval_rows)
    manifest_file_sha256s = (
        manifest.get("train", {}).get("file_sha256") == _sha256_file(train_path)
        and manifest.get("eval_frozen", {}).get("file_sha256") == _sha256_file(eval_path)
    )
    checks = {
        "train_exactly_100": len(train_rows) == 100,
        "eval_exactly_30": len(eval_rows) == 30,
        "unique_train_ids": len(train_ids) == 100 and len(set(train_ids)) == 100,
        "unique_eval_ids": len(eval_ids) == 30 and len(set(eval_ids)) == 30,
        "split_ids_disjoint": not set(train_ids).intersection(eval_ids),
        "split_membership": all(row.get("split") == "train" for row in train_rows if isinstance(row, dict)) and all(row.get("split") == "eval_frozen" for row in eval_rows if isinstance(row, dict)),
        "category_counts": dict(sorted(train_counts.items())) == EXPECTED_TRAIN_COUNTS,
        "role_alternation": not any(error == "role_alternation" for error in row_errors),
        "valid_json_arguments": not any(error.startswith("json_message") or error.startswith("invalid_arguments") for error in row_errors),
        "registered_tool_names": not any(error.startswith("unregistered_tool") for error in row_errors),
        "tools_schema_valid": not any(error == "tools_schema_mismatch" for error in row_errors),
        "reproducible_observations": not any(error.startswith("nonreproducible_observation") for error in row_errors),
        "audit_one_to_one": not any(error.startswith("audit_") for error in row_errors),
        "audit_content_projection": not any(error.startswith("audit_prompt_") or error.startswith("audit_final_") or error.startswith("audit_thought_") for error in row_errors),
        "exact_prompt_overlap_zero": exact_overlap == 0,
        "near_prompt_overlap_zero": near_overlap == 0,
        "semantic_case_overlap_zero": not semantic_overlap,
        "finite_token_proxy_lengths": all(math.isfinite(float(len(canonical_json(row)))) and len(canonical_json(row)) > 0 for row in [*train_rows, *eval_rows]),
        "deterministic_rebuild": canonical_json(train_rows) == canonical_json(rebuild_train) and canonical_json(eval_rows) == canonical_json(rebuild_eval),
        "manifest_file_sha256s": manifest_file_sha256s,
        "manifest_metadata_frozen": canonical_json(manifest) == canonical_json(expected_manifest),
        "recovery_termination_semantics": _recovery_termination_valid([*train_rows, *eval_rows]),
        "zero_shipping_calculator_required": zero_shipping_valid,
        "code_only_no_unnecessary_retrieval": code_only_valid,
        "formal_llamafactory_runtime": "pending_autodl_preflight",
    }
    for name, passed in checks.items():
        if passed is False:
            errors.append(name)
    return {
        "schema_version": "1.0",
        "valid": not errors,
        "errors": sorted(set(errors)),
        "checks": checks,
        "counts": {"train": len(train_rows), "eval_frozen": len(eval_rows), "exact_prompt_overlap": exact_overlap, "near_prompt_overlap": near_overlap, "semantic_case_overlap": len(semantic_overlap)},
        "note": "Local ShareGPT schema smoke passed structurally; formal LLaMA-Factory version/revision and parser execution remain an AutoDL preflight fact.",
    }


def _v2_self_contained(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        prompt = _prompt(row)
        calls = _decoded_calls(row.get("conversations"))
        if not calls:
            return False
        first_value = next(iter(calls[0][1].values()), None)
        if first_value is None or str(first_value) not in prompt:
            return False
        if len(calls) == 2:
            retrieved = calls[0][2].get("data", {})
            expression = f"{retrieved.get('price_cny')} + {retrieved.get('shipping_cny')}"
            if calls[1][0] != "calculator" or calls[1][1] != {"expression": expression}:
                return False
    return True


def _v2_product_ids(rows: list[dict[str, Any]]) -> set[str]:
    product_ids: set[str] = set()
    for row in rows:
        if row.get("category") != "knowledge_plus_calculator":
            continue
        calls = _decoded_calls(row.get("conversations"))
        if calls:
            product_id = calls[0][2].get("data", {}).get("product_id")
            if isinstance(product_id, str):
                product_ids.add(product_id)
    return product_ids


def validate_dataset_v2(data_directory: Path) -> dict[str, Any]:
    file_names = {
        "train_core_100": "tool_sft_train_core_100.json",
        "train_expanded": "tool_sft_train_expanded.json",
        "train_v2": "tool_sft_train_v2.json",
        "dev_v2": "tool_sft_dev_v2.json",
        "test_v2": "tool_sft_test_v2.json",
    }
    rows = {
        name: _load(data_directory / file_name)
        for name, file_name in file_names.items()
    }
    manifest = _load(data_directory / "dataset_manifest_v2.json")
    registered_tools = {definition["name"] for definition in tool_definitions()}
    primary_rows = [
        *rows["train_core_100"],
        *rows["train_expanded"],
        *rows["dev_v2"],
        *rows["test_v2"],
    ]
    row_errors = [
        error
        for row in primary_rows
        for error in _validate_row(row, registered_tools, KNOWLEDGE_BASE_V2)
    ]
    expected = build_dataset_bundle()
    expected_train = [*expected["train_core_100"], *expected["train_expanded"]]
    deterministic = (
        canonical_json(rows["train_core_100"]) == canonical_json(expected["train_core_100"])
        and canonical_json(rows["train_expanded"]) == canonical_json(expected["train_expanded"])
        and canonical_json(rows["train_v2"]) == canonical_json(expected_train)
        and canonical_json(rows["dev_v2"]) == canonical_json(expected["dev_v2"])
        and canonical_json(rows["test_v2"]) == canonical_json(expected["test_v2"])
    )
    manifest_hashes = all(
        manifest.get("splits", {}).get(name, {}).get("file_sha256")
        == _sha256_file(data_directory / file_name)
        for name, file_name in file_names.items()
    )
    train_ids = _v2_product_ids(rows["train_v2"])
    dev_ids = _v2_product_ids(rows["dev_v2"])
    test_ids = _v2_product_ids(rows["test_v2"])
    all_ids = [row.get("id") for row in primary_rows]
    checks: dict[str, bool | str] = {
        "core_exactly_100": len(rows["train_core_100"]) == 100,
        "expanded_minimum_200": len(rows["train_expanded"]) >= 200,
        "combined_train_projection": rows["train_v2"] == [
            *rows["train_core_100"], *rows["train_expanded"]
        ],
        "dev_minimum_40": len(rows["dev_v2"]) >= 40,
        "final_test_minimum_100": len(rows["test_v2"]) >= 100,
        "unique_ids": len(all_ids) == len(set(all_ids)),
        "self_contained_inputs": _v2_self_contained(primary_rows),
        "split_product_ids_disjoint": bool(train_ids and dev_ids and test_ids)
        and train_ids.isdisjoint(dev_ids)
        and train_ids.isdisjoint(test_ids)
        and dev_ids.isdisjoint(test_ids),
        "natural_prompt_families": all(
            not _prompt(row).startswith(("训练", "冻结")) for row in primary_rows
        ),
        "valid_rows": not row_errors,
        "deterministic_rebuild": deterministic,
        "manifest_schema": manifest.get("schema_version") == "2.0",
        "manifest_file_sha256s": manifest_hashes,
        "knowledge_base_binding": manifest.get("knowledge_base", {}).get("sha256")
        == _sha256_file(KNOWLEDGE_BASE_V2),
        "formal_llamafactory_runtime": "pending_autodl_preflight",
    }
    errors = sorted(set(row_errors))
    errors.extend(name for name, passed in checks.items() if passed is False)
    return {
        "schema_version": "2.0",
        "profile": "week6_agent_formal_v2",
        "valid": not errors,
        "errors": errors,
        "checks": checks,
        "counts": {name: len(value) for name, value in rows.items()},
        "product_partitions": {
            "train": sorted(train_ids),
            "dev": sorted(dev_ids),
            "test": sorted(test_ids),
        },
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=DAY31_SOURCE / "data")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile", choices=("legacy", "v2"), default="legacy")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = (
            validate_dataset_v2(args.data_directory)
            if args.profile == "v2"
            else validate_dataset(args.data_directory)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        report = {
            "schema_version": "1.0",
            "valid": False,
            "errors": [f"dataset_load_error:{error.__class__.__name__}"],
            "checks": {"dataset_loadable": False},
            "counts": {},
            "note": "Validation could not load the dataset; no training claim is valid.",
        }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
