"""Semantic validation for frozen Day30 multi-step and AST-inspection cases."""

from __future__ import annotations

from typing import Any

from week6_agent.tools.calculator import CalculatorTool
from week6_agent.trace_schema import repeated_action_input


THREE_TOOL_NAMES = ["calculator", "knowledge_retrieval", "code_executor"]


def _contains_number(text: str, value: int | float) -> bool:
    return str(value) in text


def _product_checks(case: dict[str, Any], result: dict[str, Any]) -> dict[str, bool]:
    trace = result.get("trace", [])
    expected = case["expected"]
    retrieval = trace[0].get("observation") if len(trace) > 0 else None
    calculation = trace[1].get("observation") if len(trace) > 1 else None
    retrieval_data = retrieval.get("data", {}) if isinstance(retrieval, dict) else {}
    calculator_data = calculation.get("data", {}) if isinstance(calculation, dict) else {}
    expression = (
        trace[1].get("action_input", {}).get("expression") if len(trace) > 1 else None
    )
    recomputed = CalculatorTool().run(expression) if isinstance(expression, str) else None
    answer = str(result.get("final_answer", ""))
    if expected["over_budget"]:
        conclusion = "超过" in answer and "没有超过" not in answer and "未超过" not in answer
    else:
        conclusion = any(token in answer for token in ("没有超过", "未超过", "不超过"))
    return {
        "retrieval_success": isinstance(retrieval, dict)
        and retrieval.get("status") == "success",
        "retrieved_product": retrieval_data.get("product_id") == expected["product_id"],
        "retrieved_price": retrieval_data.get("price_cny") == expected["price_cny"],
        "retrieved_shipping": retrieval_data.get("shipping_cny")
        == expected["shipping_cny"],
        "calculator_success": isinstance(calculation, dict)
        and calculation.get("status") == "success",
        "calculator_expression": recomputed is not None
        and recomputed.status == "success"
        and recomputed.data.get("value") == expected["total_cny"],
        "calculator_value": calculator_data.get("value") == expected["total_cny"],
        "final_total": _contains_number(answer, expected["total_cny"]),
        "final_budget": _contains_number(answer, expected["budget_cny"]),
        "final_conclusion": conclusion,
    }


def _code_checks(case: dict[str, Any], result: dict[str, Any]) -> dict[str, bool]:
    trace = result.get("trace", [])
    expected = case["expected"]
    observation = trace[0].get("observation") if trace else None
    data = observation.get("data", {}) if isinstance(observation, dict) else {}
    expected_risks = sorted(expected["risk_nodes"])
    checks = {
        "code_observation_present": isinstance(observation, dict),
        "code_status": isinstance(observation, dict)
        and observation.get("status") == expected["tool_status"],
        "syntax_valid": data.get("syntax_valid") is expected["syntax_valid"],
        "risk_nodes": sorted(data.get("risk_nodes", [])) == expected_risks,
        "final_answer_present": bool(str(result.get("final_answer", "")).strip()),
    }
    if "error_code" in expected:
        checks["error_code"] = (
            isinstance(observation, dict)
            and observation.get("error_code") == expected["error_code"]
        )
    return checks


def validate_case_result(
    case: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Recompute success from tool observations and the frozen case contract."""
    trace = result.get("trace", [])
    sequence = [row.get("action") for row in trace]
    answer = str(result.get("final_answer", ""))
    checks = {
        "agent_completed": result.get("status") == "success",
        "three_tools_bound": result.get("tool_names") == THREE_TOOL_NAMES,
        "expected_tool_sequence": sequence == case["expected_tool_sequence"],
        "no_repeated_action_input": not repeated_action_input(trace),
        "bounded_step_count": len(trace) <= int(result.get("max_steps", 6)),
        "final_answer_present": bool(answer.strip()),
    }
    if case["category"] == "product_budget":
        checks.update(_product_checks(case, result))
    elif case["category"] == "code_inspection":
        checks.update(_code_checks(case, result))
    else:
        checks["known_category"] = False
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "case_id": case["case_id"],
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "auditable_stage_count": len(trace) + (1 if answer.strip() else 0),
    }
