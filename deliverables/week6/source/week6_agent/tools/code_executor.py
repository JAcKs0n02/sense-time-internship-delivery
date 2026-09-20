"""AST-only Python source inspection; this module never executes inspected code."""

from __future__ import annotations

import ast
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from week6_agent.limits import MAX_AST_DEPTH, MAX_AST_NODES
from week6_agent.tool_result import ToolResult


MAX_CODE_LENGTH = 4096


class CodeExecutorInput(BaseModel):
    """Input accepted by the LangChain-compatible AST inspection wrapper."""

    source: str = Field(
        description=(
            "Python source to parse and inspect only. The source is never executed."
        )
    )


class CodeExecutor:
    """Parse Python source and report syntax/structural risk without execution."""

    _RISK_NODE_TYPES: tuple[type[ast.AST], ...] = (
        ast.AsyncWith,
        ast.Attribute,
        ast.Call,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Nonlocal,
        ast.With,
    )

    def inspect(self, source: str) -> ToolResult:
        """Inspect an AST without compiling or evaluating the supplied source."""
        if not isinstance(source, str):
            return ToolResult(
                "rejected",
                data=self._base_data(syntax_valid=False),
                error_code="invalid_source",
            )
        if len(source) > MAX_CODE_LENGTH:
            return ToolResult(
                "rejected",
                data={
                    **self._base_data(syntax_valid=False),
                    "limit": "source_length",
                    "actual": len(source),
                    "maximum": MAX_CODE_LENGTH,
                },
                error_code="limit_exceeded",
            )

        try:
            tree = ast.parse(source, mode="exec")
        except (SyntaxError, ValueError, MemoryError) as error:
            if isinstance(error, SyntaxError):
                error_line = error.lineno
                error_column = error.offset
                error_message = error.msg
            else:
                error_line = None
                error_column = None
                error_message = error.__class__.__name__
            return ToolResult(
                "rejected",
                data={
                    **self._base_data(syntax_valid=False),
                    "error_line": error_line,
                    "error_column": error_column,
                    "error_message": error_message,
                },
                error_code="syntax_error",
            )

        nodes = list(ast.walk(tree))
        node_count = len(nodes)
        max_depth = self._tree_depth(tree)
        metrics = self._base_data(
            syntax_valid=True, node_count=node_count, max_depth=max_depth
        )
        if node_count > MAX_AST_NODES:
            return ToolResult(
                "rejected",
                data={
                    **metrics,
                    "limit": "ast_node_count",
                    "actual": node_count,
                    "maximum": MAX_AST_NODES,
                },
                error_code="limit_exceeded",
            )
        if max_depth > MAX_AST_DEPTH:
            return ToolResult(
                "rejected",
                data={
                    **metrics,
                    "limit": "ast_depth",
                    "actual": max_depth,
                    "maximum": MAX_AST_DEPTH,
                },
                error_code="limit_exceeded",
            )

        risk_nodes = sorted(
            {type(node).__name__ for node in nodes if isinstance(node, self._RISK_NODE_TYPES)}
        )
        data = {**metrics, "risk_nodes": risk_nodes}
        if risk_nodes:
            return ToolResult(
                "rejected", data=data, error_code="risk_nodes_detected"
            )
        return ToolResult("success", data=data)

    def as_langchain_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self._invoke_for_langchain,
            name="code_executor",
            description=(
                "Parses Python source with ast and reports syntax and risky node types. "
                "It never executes the source."
            ),
            args_schema=CodeExecutorInput,
        )

    def _invoke_for_langchain(self, source: str) -> dict[str, Any]:
        return self.inspect(source).to_dict()

    @staticmethod
    def _base_data(
        *, syntax_valid: bool, node_count: int = 0, max_depth: int = 0
    ) -> dict[str, Any]:
        return {
            "syntax_valid": syntax_valid,
            "error_line": None,
            "error_column": None,
            "error_message": None,
            "node_count": node_count,
            "max_depth": max_depth,
            "risk_nodes": [],
        }

    @staticmethod
    def _tree_depth(node: ast.AST) -> int:
        maximum = 0
        stack = [(node, 1)]
        while stack:
            current, depth = stack.pop()
            maximum = max(maximum, depth)
            stack.extend((child, depth + 1) for child in ast.iter_child_nodes(current))
        return maximum
