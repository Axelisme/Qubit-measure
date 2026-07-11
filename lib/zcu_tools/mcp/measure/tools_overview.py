"""Measure MCP tools-overview override tools."""

from __future__ import annotations

from typing import Any

from zcu_tools.mcp.measure.tool_context import (
    MeasureToolContext,
    bind_context,
    send_gui_rpc,
)


def _assemble_overview() -> dict[str, Any]:
    """One-shot situational overview of the live GUI, fanned out over existing
    read RPCs (no new wire method).

    Packs the readiness flags, the project identity, the active context label, the
    SoC summary, the open tabs (each with its adapter + running flag), the running
    tab and the user's currently-focused tab. ``active_tab`` is where the USER's
    eye is (a collaboration cue) — NOT the agent's operation target, which is
    always the explicit tab_id the agent passes.

    This overview is the single orientation SSOT: it folds in the project paths
    (result_dir/database_path) and the readiness flags, so the retired
    gui_state_check / gui_project_info tools have no separate surface.

    ``project`` is read from project.info only while a project is applied
    (project.info fast-fails no_project otherwise); it carries the full wire
    shape {chip_name, qub_name, res_name, result_dir, database_path}; ``null``
    when no project. ``is_mock`` is likewise read from soc.info only
    while connected (soc.info fast-fails without a SoC), so a not-yet-set-up GUI
    still yields a well-formed overview.
    """
    has_proj = send_gui_rpc("state.has_project", {}).get("value", False)
    has_ctx = send_gui_rpc("state.has_context", {}).get("value", False)
    has_act = send_gui_rpc("state.has_active_context", {}).get("value", False)
    has_soc = send_gui_rpc("state.has_soc", {}).get("value", False)

    project: dict[str, Any] | None = None
    if has_proj:
        info = send_gui_rpc("project.info", {})
        # Mirror the full project.info wire shape (long keys also match the other
        # tool-GUIs: fluxdep/dispersive/autofluxdep). Folding result_dir +
        # database_path here makes the overview the single orientation SSOT,
        # superseding the retired gui_project_info tool.
        project = {
            "chip_name": info.get("chip_name"),
            "qub_name": info.get("qub_name"),
            "res_name": info.get("res_name"),
            "result_dir": info.get("result_dir"),
            "database_path": info.get("database_path"),
        }

    soc: dict[str, Any] = {"connected": has_soc, "is_mock": None}
    if has_soc:
        soc["is_mock"] = send_gui_rpc("soc.info", {}).get("is_mock")

    tab_snaps = send_gui_rpc("tab.snapshot", {}).get("tabs", [])
    tabs = [
        {
            "tab_id": snap.get("tab_id"),
            "adapter": snap.get("adapter_name"),
            "is_running": bool(snap.get("interaction", {}).get("is_running", False)),
        }
        for snap in tab_snaps
    ]

    return {
        "state": {
            "has_project": has_proj,
            "has_context": has_ctx,
            "has_active_context": has_act,
            "has_soc": has_soc,
        },
        "project": project,
        "context": send_gui_rpc("context.active", {}).get("label"),
        "soc": soc,
        "hardware_gate": send_gui_rpc("state.hardware_gate", {}),
        "tabs": tabs,
        "running_tab": send_gui_rpc("run.running_tab", {}).get("tab_id"),
        "active_tab": send_gui_rpc("view.snapshot", {}).get("active_tab_id"),
    }


def tool_gui_overview(arguments: dict[str, Any]) -> dict[str, Any]:
    """Situational overview of the live GUI (see _assemble_overview)."""
    del arguments
    return _assemble_overview()


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_overview": {
        "handler": tool_gui_overview,
        "description": (
            "One-shot SITUATIONAL OVERVIEW of the live GUI and the single "
            "orientation SSOT — call any time to re-orient (it folds in the "
            "readiness flags and the project paths, so there is no separate "
            "state-check or project-info tool). Packs (from existing read RPCs): "
            "state (the four readiness flags has_project / has_context / "
            "has_active_context / has_soc), project ({chip_name, qub_name, "
            "res_name, result_dir, database_path} or null when none applied), "
            "context (active context label), soc ({connected, is_mock}), "
            "hardware_gate ({active:[{kind, origin_kind, note, "
            "active_for_seconds}]}), "
            "tabs ([{tab_id, adapter, is_running}]), running_tab, and active_tab. "
            "active_tab is where the USER is currently focused — a collaboration "
            "cue, NOT your operation target (you always act on an explicit tab_id). "
            "The same overview is folded into gui_bridge_connect's reply, "
            "so right after attaching you already have it."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
