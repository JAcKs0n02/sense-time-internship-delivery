"""A small result type with both attribute and JSON-object interfaces."""

from __future__ import annotations

from typing import Any


class ToolResult(dict[str, Any]):
    """JSON-serializable result returned by every constrained tool."""

    def __init__(
        self,
        status: str,
        data: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(status=status, data=dict(data or {}), error_code=error_code)

    @property
    def status(self) -> str:
        return self["status"]

    @property
    def data(self) -> dict[str, Any]:
        return self["data"]

    @property
    def error_code(self) -> str | None:
        return self["error_code"]

    def to_dict(self) -> dict[str, Any]:
        return dict(self)
