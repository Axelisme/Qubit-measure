"""Fixed operation tools over the GUI-owned operation and session state."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def status(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read current GUI orientation and every live GUI-owned operation."""
    del ctx, arguments
    raise NotImplementedError("status projection is not implemented")


def build_operation_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        "status": {
            "handler": partial(status, ctx),
            "description": "Index the live GUI session and all in-flight operations.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    }
