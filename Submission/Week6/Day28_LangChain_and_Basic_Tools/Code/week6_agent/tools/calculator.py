"""A bounded arithmetic-expression calculator built on an AST whitelist."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from week6_agent.limits import (
    MAX_AST_DEPTH,
    MAX_AST_NODES,
    MAX_EXPONENT_ABS,
    MAX_EXPRESSION_LENGTH,
    MAX_INTEGER_DIGITS,
    MAX_RESULT_ABS,
)
from week6_agent.tool_result import ToolResult


class CalculatorInput(BaseModel):
    """Input accepted by the LangChain-compatible calculator wrapper."""

    expression: str = Field(description="A single arithmetic expression.")


class _ToolFailure(Exception):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code


class CalculatorTool:
    """Interpret only a finite, explicit subset of arithmetic AST nodes."""

    _BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def run(self, expression: str) -> ToolResult:
        """Return a structured result without evaluating Python source code."""
        if not isinstance(expression, str):
            return ToolResult("rejected", error_code="invalid_expression")
        if len(expression) > MAX_EXPRESSION_LENGTH:
            return ToolResult("rejected", error_code="limit_exceeded")

        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError, MemoryError):
            return ToolResult("rejected", error_code="syntax_error")

        if self._node_count(tree) > MAX_AST_NODES or self._tree_depth(tree) > MAX_AST_DEPTH:
            return ToolResult("rejected", error_code="limit_exceeded")

        try:
            value = self._interpret(tree.body)
        except _ToolFailure as failure:
            return ToolResult("rejected", error_code=failure.error_code)
        except ZeroDivisionError:
            return ToolResult("rejected", error_code="division_by_zero")
        except (ArithmeticError, OverflowError, ValueError):
            return ToolResult("rejected", error_code="non_finite_value")

        try:
            self._ensure_finite_and_bounded(value)
        except _ToolFailure as failure:
            return ToolResult("rejected", error_code=failure.error_code)
        return ToolResult("success", data={"value": value})

    def as_langchain_tool(self) -> StructuredTool:
        """Return the stable wrapper Day 29 binds to its ReAct agent."""
        return StructuredTool.from_function(
            func=self._invoke_for_langchain,
            name="calculator",
            description="Safely calculates one bounded arithmetic expression.",
            args_schema=CalculatorInput,
        )

    def _invoke_for_langchain(self, expression: str) -> dict[str, Any]:
        return self.run(expression).to_dict()

    @staticmethod
    def _node_count(tree: ast.AST) -> int:
        return sum(1 for _ in ast.walk(tree))

    @staticmethod
    def _tree_depth(node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(CalculatorTool._tree_depth(child) for child in children)

    def _interpret(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            return self._constant_value(node.value)
        if isinstance(node, ast.UnaryOp):
            operation = self._UNARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise _ToolFailure("forbidden_node")
            value = operation(self._interpret(node.operand))
            self._ensure_finite_and_bounded(value)
            return value
        if isinstance(node, ast.BinOp):
            operation = self._BINARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise _ToolFailure("forbidden_node")
            left = self._interpret(node.left)
            right = self._interpret(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT_ABS:
                raise _ToolFailure("limit_exceeded")
            value = operation(left, right)
            self._ensure_finite_and_bounded(value)
            return value
        raise _ToolFailure("forbidden_node")

    @staticmethod
    def _constant_value(value: object) -> int | float:
        if type(value) is int:
            if len(str(abs(value))) > MAX_INTEGER_DIGITS:
                raise _ToolFailure("limit_exceeded")
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise _ToolFailure("non_finite_value")
            return value
        raise _ToolFailure("forbidden_node")

    @staticmethod
    def _ensure_finite_and_bounded(value: object) -> None:
        if type(value) not in (int, float):
            raise _ToolFailure("non_finite_value")
        if isinstance(value, float) and not math.isfinite(value):
            raise _ToolFailure("non_finite_value")
        if abs(value) > MAX_RESULT_ABS:
            raise _ToolFailure("limit_exceeded")
