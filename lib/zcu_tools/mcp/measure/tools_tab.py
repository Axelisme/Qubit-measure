"""Fixed public declarations for the 05 read/open tab tools."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def experiments(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Index live GUI adapters; derive each summary from its guide behavior."""
    raise NotImplementedError("05 experiments implementation pending")


def guide(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the named live GUI adapter's guide."""
    raise NotImplementedError("05 guide implementation pending")


def tab_open(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Create/activate a tab; a failed from_file load must close that new tab."""
    raise NotImplementedError("05 tab_open implementation pending")


def tab_get(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Project requested sections of one explicit tab without changing focus."""
    raise NotImplementedError("05 tab_get implementation pending")


def tab_live(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read live run progress/figure without changing the focused tab."""
    raise NotImplementedError("05 tab_live implementation pending")


_TAB = {"type": "string", "minLength": 1, "description": "Explicit GUI tab id"}
_EXPERIMENT = {"type": "string", "minLength": 1, "description": "Live experiment name"}


TAB_READ_TOOLS: dict[str, dict[str, Any]] = {
    "experiments": {
        "handler": experiments,
        "description": "List live experiments as {experiments: [{name, summary}]}; summary is the first sentence of the current guide behavior. Optional prefix filters by name.",
        "inputSchema": {
            "type": "object",
            "properties": {"prefix": {"type": "string"}},
        },
    },
    "guide": {
        "handler": guide,
        "description": "Read the current GUI experiment guide {behavior, expects_md, expects_ml, typical_writeback, recommended}; it is orientation, not a cfg contract.",
        "inputSchema": {
            "type": "object",
            "properties": {"experiment": _EXPERIMENT},
            "required": ["experiment"],
        },
    },
    "tab_open": {
        "handler": tab_open,
        "description": "Open and focus a new tab; optionally load an existing compatible data file without a SoC. On load failure close the new tab or report cleanup failure; return {tab, experiment} only after success.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "experiment": _EXPERIMENT,
                "from_file": {"type": "string", "minLength": 1},
            },
            "required": ["experiment"],
        },
    },
    "tab_get": {
        "handler": tab_get,
        "description": "Read explicit tab sections without changing GUI focus. 05 provides the base summary/cfg/analyze_params/analysis/post/artifacts projection; 06 adds cfg type/choice/lock detail, 09 adds artifact status/last saved paths. Missing later fields must be marked partial, not fabricated.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab": _TAB,
                "include": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "summary",
                            "cfg",
                            "analyze_params",
                            "analysis",
                            "post",
                            "artifacts",
                        ],
                    },
                    "default": ["summary"],
                },
            },
            "required": ["tab"],
        },
    },
    "tab_live": {
        "handler": tab_live,
        "description": "Read one tab's running/progress/elapsed_s/eta_s and run figure PNG path without changing focus. If there has been no run/result, report reason=no_run.",
        "inputSchema": {
            "type": "object",
            "properties": {"tab": _TAB},
            "required": ["tab"],
        },
    },
}


def build_tab_read_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        name: {**entry, "handler": partial(entry["handler"], ctx)}
        for name, entry in TAB_READ_TOOLS.items()
    }
