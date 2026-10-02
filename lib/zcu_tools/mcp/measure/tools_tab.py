"""Fixed public declarations for the 05 read/open tab tools."""

from __future__ import annotations

import re
from functools import partial
from typing import Any

from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def _tab_snapshot(ctx: MeasureToolContext, tab: str) -> dict[str, Any]:
    tabs = ctx.gui.read_internal("tab.snapshot", {"tab_id": tab})["tabs"]
    if len(tabs) != 1 or tabs[0].get("tab_id") != tab:
        raise GuiRpcError(f"unknown tab {tab!r}", reason="unknown_tab")
    return tabs[0]


def _figure(ctx: MeasureToolContext, tab: str, pane: str) -> str | None:
    path = ctx.session.new_png_path()
    try:
        reply = ctx.send_gui_rpc(
            "tab.get_figure",
            {"tab_id": tab, "subtab_id": pane, "out_path": str(path)},
        )
    except GuiRpcError as exc:
        # The pane can exist without a figure, especially before the first
        # progress update. Other errors (including transport errors) must surface.
        if exc.code == "precondition_failed":
            return None
        raise
    if reply.get("saved_to") != str(path) or not path.is_file():
        raise GuiRpcError("GUI did not write the requested figure", reason="missing")
    return str(path)


def experiments(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Index live GUI adapters; derive each summary from its guide behavior."""
    prefix = arguments.get("prefix", "")
    if not isinstance(prefix, str):
        raise ValueError("prefix must be a string")
    names = ctx.gui.read_internal("adapter.list", {})["adapters"]
    result = []
    for name in names:
        if not name.startswith(prefix):
            continue
        current = ctx.gui.read_internal("adapter.guide", {"adapter_name": name})[
            "guide"
        ]
        behavior = current["behavior"].strip()
        first = re.search(r"[^.!?。！？]*[.!?。！？]", behavior)
        result.append(
            {"name": name, "summary": first.group().strip() if first else behavior}
        )
    return {"experiments": result}


def guide(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the named live GUI adapter's guide."""
    return ctx.gui.read_internal(
        "adapter.guide", {"adapter_name": arguments["experiment"]}
    )["guide"]


def tab_open(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Create/activate a tab; a failed from_file load must close that new tab."""
    experiment = arguments["experiment"]
    if "from_file" in arguments:
        loaded = ctx.send_gui_rpc(
            "tab.open_file",
            {"adapter_name": experiment, "data_path": arguments["from_file"]},
        )
        return {
            "tab": loaded["tab_id"],
            "experiment": experiment,
            "cfg_backfill": loaded["cfg_backfill"],
        }
    previous_focus = ctx.gui.read_internal("tab.list_all", {})["active_tab_id"]
    created = ctx.send_gui_rpc("tab.new", {"adapter_name": experiment})
    tab = created["tab_id"]
    try:
        ctx.send_gui_rpc("tab.set_active", {"tab_id": tab})
    except Exception as open_error:
        try:
            ctx.send_gui_rpc("tab.close", {"tab_id": tab})
        except Exception as cleanup_error:
            raise GuiRpcError(
                f"opening tab {tab!r} failed: {open_error}; "
                f"cleanup also failed: {cleanup_error}; tab may remain open",
                reason="cleanup_failed",
            ) from cleanup_error
        if previous_focus is not None:
            try:
                ctx.send_gui_rpc("tab.set_active", {"tab_id": previous_focus})
            except Exception as restore_error:
                raise GuiRpcError(
                    f"opening tab {tab!r} failed: {open_error}; "
                    f"restoring focus to {previous_focus!r} also failed: "
                    f"{restore_error}; previous focus may not be restored",
                    reason="cleanup_failed",
                ) from restore_error
        raise
    return {"tab": tab, "experiment": experiment}


def tab_close(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    tab = arguments["tab"]
    ctx.send_gui_rpc(
        "tab.close",
        {
            "tab_id": tab,
            "discard_unsaved": arguments.get("discard_unsaved", False),
        },
    )
    return {"closed": tab}


def tab_get(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Project requested sections of one explicit tab without changing focus."""
    tab = arguments["tab"]
    include = arguments.get("include", ["summary"])
    valid = {"summary", "cfg", "analyze_params", "analysis", "post", "artifacts"}
    if not isinstance(include, list) or any(
        not isinstance(item, str) or item not in valid for item in include
    ):
        raise ValueError(f"include must be a list of {sorted(valid)}")
    result: dict[str, Any] = {}
    snap = None
    if "summary" in include or "artifacts" in include:
        snap = _tab_snapshot(ctx, tab)
        result["operation_state"] = snap
    if "summary" in include:
        assert snap is not None
        interaction = snap["interaction"]
        state = {
            "running": bool(interaction["is_running"]),
            "analyzing": bool(interaction["is_analyzing"]),
            "has_result": bool(interaction["has_run_result"]),
            "has_analysis": bool(interaction["has_analyze_result"]),
            "has_post": bool(interaction["has_post_analyze_result"]),
        }
        result["summary"] = {
            "experiment": snap["adapter_name"],
            "state": state,
            "source_file": snap.get("result_source_path"),
        }
    if "cfg" in include:
        result["cfg"] = ctx.gui.read_internal("tab.get_cfg", {"tab_id": tab})
    if "analyze_params" in include:
        primary = ctx.gui.read_internal("tab.get_analyze_params", {"tab_id": tab})
        post = ctx.gui.read_internal("tab.get_post_analyze_params", {"tab_id": tab})
        result["analyze_params"] = {
            "primary": {
                "definitions": primary["definitions"],
                "values": primary["analyze_params"],
            },
            "post": {
                "definitions": post["definitions"],
                "values": post["post_analyze_params"],
            },
        }
    for key, method, pane in (
        ("analysis", "tab.get_analyze_result", "analysis"),
        ("post", "tab.get_post_analyze_result", "post_analysis"),
    ):
        if key in include:
            summary = ctx.gui.read_internal(method, {"tab_id": tab})["summary"]
            result[key] = (
                {"reason": "no_result"}
                if summary is None
                else {"summary": summary, "figure": _figure(ctx, tab, pane)}
            )
    if "artifacts" in include:
        assert snap is not None
        keys = {"data": "data", "analysis": "analysis", "post_analysis": "post"}
        result["artifacts"] = [
            {
                "key": keys[artifact["kind"]],
                "kind": "data" if artifact["kind"] == "data" else "image",
                "status": artifact["status"],
                "default_path": artifact["default_path"],
                "last_saved_path": artifact["last_saved_path"],
                "is_saveable": artifact["is_saveable"],
            }
            for artifact in snap["artifacts"]
        ]
    return result


def tab_live(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read live run progress/figure without changing the focused tab."""
    ctx = ctx.bound()
    tab = arguments["tab"]
    snap = _tab_snapshot(ctx, tab)
    has_result = bool(snap["interaction"]["has_run_result"])
    running = bool(snap["interaction"]["is_running"])
    if not running and not has_result:
        return {"running": False, "reason": "no_run", "operation_state": snap}
    progress: list[dict[str, Any]] = []
    eta: float | None = None
    elapsed: float | None = None
    if running:
        operations = ctx.gui.read_internal("operation.active", {})["operations"]
        for operation in operations:
            if operation.get("tab") == tab and operation.get("kind") == "run":
                handle = ctx.gui.expose_operation(operation["op"])
                update = ctx.gui.read_internal(
                    "operation.progress", {}, operation_handle=handle
                )
                bars = update["bars"]
                elapsed = update["elapsed_s"]
                progress = [
                    {"label": bar["format"], "percent": bar["percent"]} for bar in bars
                ]
                estimates = [
                    bar["eta_s"] for bar in bars if bar.get("eta_s") is not None
                ]
                if estimates:
                    eta = max(estimates)
                break
    return {
        "operation_state": snap,
        "running": running,
        "progress": progress,
        "elapsed_s": elapsed,
        "eta_s": eta,
        "figure": _figure(ctx, tab, "run"),
    }


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
        "description": "Open and focus a new tab. With from_file, explicitly read context first; GUI creates and loads without a SoC, closes on load failure or reports cleanup_failed. Return {tab, experiment}, plus cfg_backfill=applied|not_applied for from_file. A backfill failure retains the result. Read tab_get before subsequent guarded writes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "experiment": _EXPERIMENT,
                "from_file": {"type": "string", "minLength": 1},
            },
            "required": ["experiment"],
        },
    },
    "tab_close": {
        "handler": tab_close,
        "description": "Close the specified idle tab. All unsaved artifacts require "
        "discard_unsaved=true. Active operations cannot be discarded. No hidden reads or retry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab": _TAB,
                "discard_unsaved": {"type": "boolean", "default": False},
            },
            "required": ["tab"],
        },
    },
    "tab_get": {
        "handler": tab_get,
        "description": "Read explicit tab sections without changing GUI focus. cfg is the complete GUI-owned cached observation (kind, type, current input, choices and locks); agent edits whole sweeps through tab_edit, not the GUI's leaf control paths. Artifacts project the GUI-owned status, default_path, last_saved_path and is_saveable for data, analysis and post images.",
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
