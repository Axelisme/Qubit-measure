"""J2 alignment decisions compose native named context commands and evidence."""

from time import sleep
from typing import NotRequired, TypedDict

from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui
from zcu_tools.mcp.core.stdio_server import ToolTable

from .projection import with_png


class AlignArguments(TypedDict):
    """Alignment intent; GUI owns validation, input and published provenance."""

    name: str
    command: str
    params: NotRequired[dict[str, object]]
    image: NotRequired[bool]


def build_align_tools(bridge: McpBridge) -> ToolTable:
    """Build the stateless J2 tool: open, command, wait, summarize, render.

    All numerical and lifecycle facts come from GUI captures. Native invocation
    errors propagate; accepted commands are never retried or rolled back.
    Polling holds only this call's envelope, not a business state registry.
    """

    def align(arguments: AlignArguments) -> ToolReply:
        name, command = arguments["name"], arguments["command"]
        target: dict[str, object] = {"name": name}
        opened = call_gui(
            bridge,
            GuiRpcCall("spectrum.interactive.open", {**target, "kind": "line"}, 31.0),
        ).data
        before = opened["context"]["state"]
        envelope: dict[str, object] = {
            **target,
            "context_id": opened["context"]["context_id"],
        }
        params = dict(arguments.get("params", {}))
        if command == "move_line" and "conjugate" in params:
            call_gui(
                bridge,
                GuiRpcCall(
                    "spectrum.interactive.command",
                    {
                        **envelope,
                        "command": "set_conjugate",
                        "params": {"enabled": params.pop("conjugate")},
                    },
                    31.0,
                ),
            )
        native = call_gui(
            bridge,
            GuiRpcCall(
                "spectrum.interactive.command",
                {**envelope, "command": command, "params": params},
                31.0,
            ),
        ).data
        if command == "auto_align":
            while native["context"]["info"]["alignment_busy"]:
                sleep(1.0)
                native = call_gui(
                    bridge, GuiRpcCall("interactive.read", target, 31.0)
                ).data
        context = native["context"]
        after = context["state"]
        summary: dict[str, object] = {
            "name": name,
            "stage": "aligned" if command == "finish" else "align",
            "pending": not context["closed"],
            "lines": {
                "half": round(after["flux_half"], 3),
                "integer": round(after["flux_int"], 3),
                "period": round(2.0 * abs(after["flux_int"] - after["flux_half"]), 3),
            },
        }
        changes = {
            label: [round(before[key], 3), round(after[key], 3)]
            for label, key in (("half", "flux_half"), ("integer", "flux_int"))
            if before[key] != after[key]
        }
        if changes:
            summary["changes"] = changes
        if "discarded" in opened:
            summary["discarded"] = opened["discarded"]
        if command == "auto_align":
            error = context["info"]["alignment_error"]
            summary["auto_align"] = {
                "status": "failed" if error is not None else "finished",
                "error": error,
            }
        reply = ToolReply(summary)
        if command == "finish" or arguments.get("image", False):
            method = "spectrum.render" if command == "finish" else "interactive.render"
            return with_png(reply, call_gui(bridge, GuiRpcCall(method, target, 31.0)))
        return reply

    return {
        bridge.config.tool_prefix + "align": {
            "handler": align,
            "description": "Align a named spectrum (J2). move_line: role=half|integer, position=native device value, conjugate=false. auto_align waits until terminal and reports before/after lines. finish accepts manual or inherited lines without requiring auto_align and returns alignment evidence. image=true requests draft evidence for any command. GUI owns validation and draft lifetime; invocation failures do not undo accepted edits.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "command": {
                        "type": "string",
                        "enum": ["move_line", "auto_align", "finish"],
                    },
                    "params": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["half", "integer"]},
                            "position": {"type": "number"},
                            "conjugate": {"type": "boolean", "default": False},
                        },
                    },
                    "image": {"type": "boolean", "default": False},
                },
                "required": ["name", "command"],
            },
        }
    }
