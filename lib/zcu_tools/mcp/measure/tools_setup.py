"""Fixed project tool declarations for the measure-gui setup flow."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def project(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read or atomically update a project through the GUI owner."""
    if arguments:
        names = {
            "chip": "chip_name",
            "qubit": "qub_name",
            "resonator": "res_name",
            "scope": "scope_id",
        }
        params = {
            wire: arguments[name] for name, wire in names.items() if name in arguments
        }
        result = ctx.send_gui_rpc("startup.apply", params)
    else:
        result = ctx.session.read_internal("project.info", {})
    return {
        "chip": result["chip_name"],
        "qubit": result["qub_name"],
        "resonator": result["res_name"],
        "result_dir": result["result_dir"],
        "database_path": result["database_path"],
    }


_PROJECT_FIELD = {"type": "string", "minLength": 1}

PROJECT_TOOLS: dict[str, dict[str, Any]] = {
    "project": {
        "handler": project,
        "description": (
            "Read the applied project with no arguments, or atomically update only the "
            "provided chip/qubit/resonator/scope in the GUI. Missing identity fields "
            "on a first update fail. Changing chip/qubit without scope selects the "
            "new identity's GUI default scope; an effective change deactivates the "
            "active context. An equal write or failed update does not. Returns "
            "{chip, qubit, resonator, result_dir, database_path}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "chip": _PROJECT_FIELD,
                "qubit": _PROJECT_FIELD,
                "resonator": _PROJECT_FIELD,
                "scope": _PROJECT_FIELD,
            },
        },
    },
}


def build_setup_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        name: {**entry, "handler": partial(entry["handler"], ctx)}
        for name, entry in PROJECT_TOOLS.items()
    }
