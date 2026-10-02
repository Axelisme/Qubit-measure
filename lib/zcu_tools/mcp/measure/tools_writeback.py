"""Agent writeback view over the GUI-owned shared draft."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def writeback(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    stage = arguments.get("stage", "primary")
    if stage not in ("primary", "post"):
        raise ValueError("stage must be primary or post")
    params = {
        "tab_id": arguments["tab"],
        "subtab_id": "analysis" if stage == "primary" else "post_analysis",
    }
    if "write" in arguments:
        return ctx.send_gui_rpc(
            "tab.writeback_write", {**params, "write": arguments["write"]}
        )
    preview = ctx.send_gui_rpc("tab.writeback_preview", params)
    return {
        "destination": preview["destination_context"],
        "items": [
            {
                "id": item["id"],
                "kind": "md" if item["kind"] == "metadict" else item["kind"],
                "target": item["target_name"],
                "description": item["description"],
                "current": item["current"],
                "proposed": item["proposed"],
            }
            for item in preview["items"]
        ],
    }


def accept(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Accept all current pane candidates without changing selection or observations."""
    tab = arguments.get("tab")
    if not isinstance(tab, str) or not tab:
        raise ValueError("tab must be a non-empty string")
    if arguments.keys() - {"tab"}:
        raise ValueError("accept only takes tab")

    completed: list[dict[str, Any]] = []
    skipped: list[str] = []
    not_started = ["primary", "post"]
    reply: dict[str, Any] = {
        "tab": tab,
        "status": "finished",
        "completed": completed,
        "skipped": skipped,
        "not_started": not_started,
    }
    panes: list[tuple[str, str, bool]] = []
    stage = "primary"
    write_attempted = False
    try:
        # These queries and previews do not reveal resources or unlock GUI guards.
        for stage, subtab, method in (
            ("primary", "analysis", "tab.get_analyze_result"),
            ("post", "post_analysis", "tab.get_post_analyze_result"),
        ):
            result = ctx.send_gui_rpc(method, {"tab_id": tab})
            panes.append((stage, subtab, result["summary"] is not None))

        for stage, subtab, has_result in panes:
            write_attempted = False
            if not has_result:
                skipped.append(stage)
                not_started.remove(stage)
                continue
            params = {"tab_id": tab, "subtab_id": subtab}
            preview = ctx.send_gui_rpc("tab.writeback_preview", params)
            if not preview["has_draft"] or not preview["items"]:
                skipped.append(stage)
                not_started.remove(stage)
                continue
            write = [{"id": item["id"]} for item in preview["items"]]
            # An unsuccessful RPC does not establish that the GUI made no changes.
            write_attempted = True
            written = ctx.send_gui_rpc(
                "tab.writeback_write", {**params, "write": write}
            )["written"]
            completed.append({"stage": stage, "written": written})
            not_started.remove(stage)
    except GuiRpcError as exc:
        not_started.remove(stage)
        reply.update(
            status="failed",
            failed_stage=stage,
            error={"code": exc.code, "reason": exc.reason, "message": str(exc)},
            failed_stage_may_have_partial_writes=write_attempted,
        )
    return reply


WRITEBACK_TOOL: dict[str, Any] = {
    "handler": writeback,
    "description": (
        "Preview shared GUI writeback proposals and their active-context destination. "
        "With write=[{id,target?,value?,edits?}], edit the listed draft items in order, "
        "then apply only those IDs once. A draft edit failure preserves the successful "
        "prefix and starts no context write. GUI checkboxes do not change. "
        "Returns {written:[{id,kind,target,before,after}]} with actual complete values; "
        "same target names across kinds remain separate. No cross-file rollback. "
        "Omitted value retains the proposal; explicit null sets an md value to null. "
        "Read the tab and context explicitly before a guarded write."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "tab": {"type": "string", "minLength": 1},
            "stage": {"type": "string", "enum": ["primary", "post"]},
            "write": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "target": {"type": "string", "minLength": 1},
                        "value": {},
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string", "minLength": 1},
                                    "value": {},
                                },
                                "required": ["path", "value"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["tab"],
        "additionalProperties": False,
    },
}


ACCEPT_TOOL: dict[str, Any] = {
    "handler": accept,
    "description": (
        "Accept every current candidate in the tab's Primary and existing Post panes, "
        "including unchecked items. Keeps panes separate and writes Primary then Post. "
        "Result summaries and previews do not refresh observations; GUI guards still "
        "require prior explicit tab/context reads. Stops at the first error without "
        "retry or rollback. Returns tab, status (finished|failed), completed "
        "([{stage,written}]), skipped, and not_started. A failed reply also includes "
        "failed_stage, error ({code,reason,message}), and "
        "failed_stage_may_have_partial_writes. Only GUI-confirmed writes are completed. "
        "The failed stage is separate from not_started; query/preview failures have "
        "no writes in that stage, while a failed write may have partial effects."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {"tab": {"type": "string", "minLength": 1}},
        "required": ["tab"],
        "additionalProperties": False,
    },
}


def build_writeback_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        "writeback": {**WRITEBACK_TOOL, "handler": partial(writeback, ctx)},
        "accept": {**ACCEPT_TOOL, "handler": partial(accept, ctx)},
    }
