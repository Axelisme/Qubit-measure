"""Measure MCP tools-debug override tools."""

from __future__ import annotations

from typing import Any

from zcu_tools.mcp.measure.tool_context import (
    _SESSION,
    MeasureToolContext,
    bind_context,
    send_gui_rpc,
)


def tool_gui_debug_resource_versions(arguments: dict[str, Any]) -> dict[str, Any]:
    """Dump the full per-resource version table (DEV — debugging stale-guard).

    Reads resources.versions verbatim — the same table _refresh_versions consumes
    for the optimistic-concurrency guard, but here the version numbers are
    returned as-is instead of being kept as mcp<->RPC bookkeeping hidden from the
    operator. The only window into why a guarded op rejected (or did not reject)
    on a stale dependency. Returns the flat table {resource_key: version_int} —
    the GUI's live authoritative versions (a fresh read), NOT the mcp-side
    ``_LAST_SEEN`` cache. Side effect: like any ``send_gui_rpc`` round-trip the
    call resyncs ``_LAST_SEEN`` to the current table, but it never bumps a
    resource version (those move only on a real edit/run/writeback).

    Note: wire/gui/mcp *code* version握手 (WIRE_VERSION/GUI_VERSION/MCP_VERSION)
    lives in gui_launch / gui_bridge_connect's 'note' field — not in this table.
    """
    del arguments
    res = send_gui_rpc("resources.versions", {})
    return res.get("versions", {})


def tool_gui_debug_operations(arguments: dict[str, Any]) -> dict[str, Any]:
    """Dump the mcp-side per-key operation-handle cache (DEV).

    The ONLY source is the session's semantic-key -> latest operation_id projection
    (ADR-0026 §8). It is NO LONGER on the wait/poll path (the agent drives
    gui_op_poll / gui_op_wait with the handle a START reply gave it); surfacing it
    here answers "what handle did the last run/analyze/setup on this tab/device
    get". 'latest wins' and entries are NEVER removed — a stale key for a completed
    op is normal, so the cache cannot itself certify liveness (the optional 'live'
    field is therefore omitted; the cache has no authoritative source for it).

    The live device enumeration is NOT duplicated here — gui_device_list_operations
    is the dedicated wire enumerator for that.

    Returns {handles: {key: {operation_id: int}}}.
    """
    del arguments
    return _SESSION.debug_operations()


NON_GENERATED_METHODS = frozenset(
    {
        "resources.versions",
    }
)


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_debug_resource_versions": {
        "handler": tool_gui_debug_resource_versions,
        "description": (
            "DEV TOOL (debugging the GUI/MCP itself, not the measurement): dump the "
            "per-resource optimistic-concurrency version table as a flat "
            "{resource_key: int} map, for debugging stale-guard rejections. "
            "These version numbers are normally hidden from the operator "
            "(mcp<->RPC bookkeeping); read them only when debugging why a "
            "guarded op rejected (or failed to reject) on a stale dependency. "
            "Returns the GUI's live (authoritative) table — not the mcp-side "
            "cache. SIDE EFFECT: like any RPC round-trip this resyncs the mcp "
            "last-seen baseline to the current table; it does NOT bump any "
            "resource version (the numbers only move on a real edit/run/writeback). "
            "NOTE: wire/gui/mcp *code* version握手 (WIRE_VERSION / GUI_VERSION / "
            "MCP_VERSION) lives in the gui_launch / gui_bridge_connect 'note' field "
            "— it is not in this table."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_debug_operations": {
        "handler": tool_gui_debug_operations,
        "description": (
            "DEV TOOL (debugging the GUI/MCP itself, not the measurement): dump the "
            "mcp-side per-key operation-handle cache, normally hidden from the "
            "operator. Returns {handles: {key: {operation_id}}} keyed by semantic "
            "key (e.g. 'tab:<id>', 'analyze:<id>', 'post_analyze:<id>', "
            "'device:<name>') -> the latest operation_id captured for that resource "
            "— the only view of run/analyze/post_analyze handles. Use when debugging "
            "no_operation / a stuck wait (e.g. which handle did the last run on this "
            "tab get). The live DEVICE enumeration is NOT duplicated here — "
            "gui_device_list_operations is the dedicated wire enumerator. "
            "Lifecycle: a key is written when the matching start RPC fires "
            "(tab.run_start / tab.analyze / tab.post_analyze / "
            "device.connect/disconnect/setup) with 'latest wins', and is NEVER "
            "removed — entries persist for the entire MCP server process lifetime. "
            "A stale key for a completed operation is normal, so the cache cannot "
            "certify liveness (there is no 'live' field)."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
