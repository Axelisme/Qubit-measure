"""Standalone writeback tool; answering a recipe never happens here."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.measure.recipe_contracts import AnalysisStage, WritebackReceipt
from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.writeback import write_current_draft


def apply_writeback(
    ctx: MeasureToolContext, arguments: Mapping[str, object]
) -> WritebackReceipt:
    """Write current drafts for arguments['tab'], restricted by stage and items.

    tab must be a non-empty string. stage is primary/post; omitted or null selects
    both, rejecting ambiguous names with the available stages. items is omitted or
    null for all targets in scope, or a JSON array of non-empty stable target_name
    strings; [] writes none. Reject other keys/invalid values before GUI access.
    Return confirmed progress and submission recovery, including a failed stage's
    uncertainty, without retry or rollback. This never answers a waiting recipe.
    """
    tab = arguments.get("tab")
    if not isinstance(tab, str) or not tab:
        raise ValueError("tab must be a non-empty string")
    if arguments.keys() - {"tab", "stage", "items"}:
        raise ValueError("apply_writeback only takes tab, stage and items")
    raw_stage = arguments.get("stage")
    stage: AnalysisStage | None = None
    if raw_stage is not None:
        if not isinstance(raw_stage, str) or raw_stage not in ("primary", "post"):
            raise ValueError("stage must be primary, post or null")
        stage = "primary" if raw_stage == "primary" else "post"
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
    return write_current_draft(ctx, tab, items, stage=stage)


def build_writeback_tools(ctx: MeasureToolContext) -> ToolTable:
    """Bind standalone apply_writeback to ctx without accessing the GUI."""
    return {
        "apply_writeback": {
            "handler": partial(apply_writeback, ctx),
            "description": (
                "Apply current writeback drafts by stage and stable target_name. "
                "Omit stage for Primary then Post; omit items for all; [] writes none. "
                "Ignores checkboxes; rejects unknown, duplicate or ambiguous names before "
                "writing. First error stops. Returns confirmed progress and submission "
                "recovery. Timeout is unknown: query recovery before resubmitting. "
                "No retry, rollback or recipe answer."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tab": {"type": "string", "minLength": 1},
                    "stage": {
                        "type": ["string", "null"],
                        "enum": ["primary", "post", None],
                        "description": "Restrict targets to this stage. Omitted/null selects both; shared names are ambiguous.",
                    },
                    "items": {
                        "type": ["array", "null"],
                        "items": {"type": "string", "minLength": 1},
                        "description": "Stable target_name values within stage scope. Omitted/null selects all; [] writes none.",
                    },
                },
                "required": ["tab"],
                "additionalProperties": False,
            },
        }
    }
