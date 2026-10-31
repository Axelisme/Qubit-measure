"""Standalone writeback tool; answering a recipe never happens here."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.recipe_contracts import WritebackReceipt
from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.writeback import write_current_draft


def apply_writeback(
    ctx: MeasureToolContext, arguments: Mapping[str, object]
) -> WritebackReceipt:
    """Write current drafts for arguments['tab'], optionally restricted by items.

    tab must be a non-empty string; items must be omitted, null or a JSON array of
    non-empty stable target_name strings. Reject other keys/invalid values before
    GUI access. Return the confirmed progress receipt, including a failed stage's
    uncertainty, without retry or rollback. This never answers a waiting recipe.
    """
    tab = arguments.get("tab")
    if not isinstance(tab, str) or not tab:
        raise ValueError("tab must be a non-empty string")
    if arguments.keys() - {"tab", "items"}:
        raise ValueError("apply_writeback only takes tab and items")
    raw_items = arguments.get("items")
    items: list[str] | None = None
    if raw_items is not None:
        if not isinstance(raw_items, list):
            raise ValueError("items must be an array of non-empty stable names")
        items = []
        for name in raw_items:
            if not isinstance(name, str) or not name:
                raise ValueError("items must be an array of non-empty stable names")
            items.append(name)
    return write_current_draft(ctx, tab, items)


def build_writeback_tools(ctx: MeasureToolContext) -> ToolTable:
    """Bind standalone apply_writeback to ctx without accessing the GUI."""
    return {
        "apply_writeback": {
            "handler": partial(apply_writeback, ctx),
            "description": (
                "Apply current Primary/Post writeback drafts by stable target_name. "
                "Omit items for all; [] writes none. Ignores checkboxes; rejects unknown, "
                "duplicate or ambiguous names before writing. Primary then Post, first "
                "error stops. Returns confirmed progress, submissions and receipt recovery. "
                "Timeout is unknown: query recovery before resubmitting. No retry, rollback "
                "or recipe answer."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tab": {"type": "string", "minLength": 1},
                    "items": {
                        "type": ["array", "null"],
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": ["tab"],
                "additionalProperties": False,
            },
        }
    }
