"""Deterministic post-trace policy checks for the formal Week6 Agent runtime."""

from __future__ import annotations

import ast
import json
from typing import Any, Iterable


POLICY_MODES = ("raw", "guarded")


def _successful_observation(event: dict[str, Any]) -> bool:
    observation = event.get("observation")
    return (
        isinstance(observation, dict)
        and observation.get("status") == "success"
        and observation.get("error_code") is None
    )


def _argument_schema_valid(event: dict[str, Any]) -> bool:
    action = event.get("action")
    value = event.get("action_input")
    expected = {
        "calculator": "expression",
        "knowledge_retrieval": "query",
        "code_executor": "source",
    }.get(action)
    return (
        expected is not None
        and isinstance(value, dict)
        and set(value) == {expected}
        and isinstance(value[expected], str)
        and bool(value[expected])
    )


def _literal_number(node: ast.AST) -> int | float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
    return None


def _is_grounded_sum(expression: object, price: object, shipping: object) -> bool:
    if not isinstance(expression, str) or not isinstance(price, (int, float)) or not isinstance(
        shipping, (int, float)
    ):
        return False
    try:
        parsed = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError):
        return False
    if not isinstance(parsed, ast.BinOp) or not isinstance(parsed.op, ast.Add):
        return False
    return _literal_number(parsed.left) == price and _literal_number(parsed.right) == shipping


def _looks_like_malformed_tool_output(final_text: str) -> bool:
    stripped = final_text.strip()
    if "<tool_call>" in stripped or "</tool_call>" in stripped:
        return True
    if not stripped.startswith("{"):
        return False
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return '"name"' in stripped or '"arguments"' in stripped
    return isinstance(value, dict) and ("name" in value or "arguments" in value)


def validate_trace_policy(
    prompt: str,
    trace: Iterable[dict[str, Any]],
    final_text: str,
) -> dict[str, Any]:
    """Validate observable tool use without changing the model-produced trace."""
    events = list(trace)
    violations: list[str] = []

    if any(not _argument_schema_valid(event) for event in events):
        violations.append("argument_schema_invalid")

    retrieval_indexes = [
        index for index, event in enumerate(events) if event.get("action") == "knowledge_retrieval"
    ]
    if retrieval_indexes:
        first = retrieval_indexes[0]
        if first != 0:
            violations.append("retrieval_not_first")
        calculator_index = next(
            (
                index
                for index in range(first + 1, len(events))
                if events[index].get("action") == "calculator"
            ),
            None,
        )
        if calculator_index is None:
            violations.append("calculator_required_after_retrieval")
        elif _successful_observation(events[first]):
            data = events[first]["observation"].get("data")
            args = events[calculator_index].get("action_input")
            grounded = (
                isinstance(data, dict)
                and isinstance(args, dict)
                and _is_grounded_sum(
                    args.get("expression"), data.get("price_cny"), data.get("shipping_cny")
                )
            )
            if not grounded:
                violations.append("calculator_arguments_not_grounded")

    for event in events:
        if event.get("action") != "code_executor":
            continue
        action_input = event.get("action_input")
        source = action_input.get("source") if isinstance(action_input, dict) else None
        if not isinstance(source, str) or source not in prompt:
            violations.append("code_source_not_preserved")
        break

    failed_indexes = [
        index for index, event in enumerate(events) if not _successful_observation(event)
    ]
    if failed_indexes and failed_indexes[0] < len(events) - 1:
        violations.append("continued_after_tool_error")

    if _looks_like_malformed_tool_output(final_text):
        violations.append("malformed_tool_output")

    unique_violations = list(dict.fromkeys(violations))
    return {
        "status": "PASS" if not unique_violations else "FAIL",
        "violations": unique_violations,
        "checks": {
            "argument_schema_valid": "argument_schema_invalid" not in unique_violations,
            "observation_grounded_arguments": "calculator_arguments_not_grounded"
            not in unique_violations,
            "required_multistep_sequence": not any(
                item in unique_violations
                for item in ("retrieval_not_first", "calculator_required_after_retrieval")
            ),
            "code_source_preserved": "code_source_not_preserved" not in unique_violations,
            "safe_termination": "continued_after_tool_error" not in unique_violations,
            "format_valid": "malformed_tool_output" not in unique_violations,
        },
    }
