"""The connect tool's live GUI orientation read."""

from __future__ import annotations

from typing import Any

from zcu_tools.mcp.measure.tool_context import (
    MeasureToolContext,
)


def assemble_overview(
    ctx: MeasureToolContext,
) -> dict[str, Any]:
    """One-shot situational overview of the live GUI, fanned out over existing
    read RPCs (no new wire method).

    Packs the readiness flags, the project identity, the active context label, the
    SoC summary, the open tabs (each with its adapter + running flag), the running
    tab and the user's currently-focused tab. ``active_tab`` is where the USER's
    eye is (a collaboration cue) — NOT the agent's operation target, which is
    always the explicit tab_id the agent passes.

    Connect folds in project paths and readiness flags. The dedicated status
    tool will own the compact operation index once it is implemented.

    ``project`` is read from project.info only while a project is applied
    (project.info fast-fails no_project otherwise); it carries the full wire
    shape {chip_name, qub_name, res_name, result_dir, database_path}; ``null``
    when no project. ``is_mock`` is likewise read from soc.info only
    while connected (soc.info fast-fails without a SoC), so a not-yet-set-up GUI
    still yields a well-formed overview.
    """
    has_proj = ctx.session.read_internal("state.has_project", {}).get("value", False)
    has_ctx = ctx.session.read_internal("state.has_context", {}).get("value", False)
    has_act = ctx.session.read_internal("state.has_active_context", {}).get(
        "value", False
    )
    has_soc = ctx.session.read_internal("state.has_soc", {}).get("value", False)

    project: dict[str, Any] | None = None
    if has_proj:
        info = ctx.session.read_internal("project.info", {})
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
        soc["is_mock"] = ctx.session.read_internal("soc.info", {}).get("is_mock")

    tab_snaps = ctx.session.read_internal("tab.snapshot", {}).get("tabs", [])
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
        "context": ctx.session.read_internal("context.active", {}).get("label"),
        "soc": soc,
        "hardware_gate": ctx.session.read_internal("state.hardware_gate", {}),
        "tabs": tabs,
        "running_tab": ctx.session.read_internal("run.running_tab", {}).get("tab_id"),
        "active_tab": ctx.session.read_internal("view.snapshot", {}).get(
            "active_tab_id"
        ),
    }
