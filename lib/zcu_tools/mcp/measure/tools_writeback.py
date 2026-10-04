"""Standalone writeback tool; answering a recipe never happens here."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.recipe import WritebackReceipt
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
                "Write current Primary and Post draft candidates by stable target_name. "
                "Omit items for all; an empty array writes none. Ignores checkboxes and "
                "rejects unknown or ambiguous names before any write. Resolves current "
                "IDs, not earlier proposal IDs. Writes Primary then Post and stops on "
                "the first error without retry or rollback. Returns tab, status "
                "(finished|failed), completed ([{stage,written}]), skipped and "
                "not_started. Failed replies also include failed_stage, error "
                "({code,reason,message}) and failed_stage_may_have_partial_writes. "
                "Only GUI-confirmed writes count as completed. The failed stage is "
                "separate from not_started. Reads do not refresh guards: explicitly "
                "observe tab/context before writing. Does not answer recipe questions."
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
