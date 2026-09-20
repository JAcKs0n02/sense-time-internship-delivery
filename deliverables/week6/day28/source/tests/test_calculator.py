from __future__ import annotations

import sys
from pathlib import Path


WEEK6_SOURCE = Path(__file__).resolve().parents[3] / "source"
sys.path.insert(0, str(WEEK6_SOURCE))

from week6_agent.limits import (  # noqa: E402
    MAX_EXPRESSION_LENGTH,
    MAX_EXPONENT_ABS,
    MAX_INTEGER_DIGITS,
    MAX_RESULT_ABS,
)
from week6_agent.tools.calculator import CalculatorTool  # noqa: E402


def test_calculator_returns_hand_checked_arithmetic_result():
    """Catches an incorrect operator implementation or precedence handling."""
    result = CalculatorTool().run("123 * 456")

    assert result.status == "success"
    assert result.error_code is None
    assert result.data == {"value": 56088}


def test_calculator_supports_parentheses_and_unary_signs():
    """Catches a whitelist interpreter that omits valid expression nodes."""
    result = CalculatorTool().run("-(2 + 3) * +4")

    assert result.status == "success"
    assert result.data == {"value": -20}


def test_calculator_exposes_a_langchain_structured_tool():
    """Catches Day 29 receiving a wrapper with the wrong schema or result shape."""
    result = CalculatorTool().as_langchain_tool().invoke({"expression": "12 * 3"})

    assert result == {"status": "success", "data": {"value": 36}, "error_code": None}


def test_calculator_rejects_function_call_without_executing_it():
    """Catches accidental evaluation of arbitrary Python expressions."""
    result = CalculatorTool().run("__import__('os').system('id')")

    assert result.status == "rejected"
    assert result.error_code == "forbidden_node"
    assert result.data == {}


def test_calculator_reports_syntax_and_division_errors_structurally():
    """Catches raw parser and arithmetic exceptions leaking to callers."""
    syntax = CalculatorTool().run("2 +")
    divide_by_zero = CalculatorTool().run("1 / 0")

    assert (syntax.status, syntax.error_code) == ("rejected", "syntax_error")
    assert (divide_by_zero.status, divide_by_zero.error_code) == (
        "rejected",
        "division_by_zero",
    )


def test_calculator_rejects_non_finite_constants_and_results():
    """Catches acceptance of values that cannot be safely serialized."""
    result = CalculatorTool().run("1e309")

    assert (result.status, result.error_code) == ("rejected", "non_finite_value")


def test_calculator_enforces_input_and_numeric_resource_limits():
    """Catches missing guards before large inputs consume unbounded resources."""
    calculator = CalculatorTool()
    cases = [
        "1" * (MAX_EXPRESSION_LENGTH + 1),
        "1" * (MAX_INTEGER_DIGITS + 1),
        f"2 ** {MAX_EXPONENT_ABS + 1}",
        f"{MAX_RESULT_ABS} * 2",
    ]

    for expression in cases:
        result = calculator.run(expression)
        assert (result.status, result.error_code) == ("rejected", "limit_exceeded")


def test_calculator_rejects_deep_expression_trees():
    """Catches recursive interpretation without an AST depth guard."""
    result = CalculatorTool().run("-" * 40 + "1")

    assert (result.status, result.error_code) == ("rejected", "limit_exceeded")
