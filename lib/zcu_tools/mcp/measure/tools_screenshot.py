"""Fixed public declaration for the 05 screenshot tool."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def screenshot(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Capture an existing window/dialog into a session-owned PNG path."""
    target = arguments.get("target", "window")
    if not isinstance(target, str) or target not in {
        "window",
        "setup",
        "device",
        "predictor",
        "inspect",
        "arb_waveform",
    }:
        raise ValueError(f"unknown screenshot target: {target!r}")
    path = ctx.session.new_png_path()
    method = "view.screenshot" if target == "window" else "dialog.screenshot"
    params = {"out_path": str(path)}
    if target != "window":
        params["name"] = target
    reply = ctx.send_gui_rpc(method, params)
    if reply.get("saved_to") != str(path) or not path.is_file():
        raise GuiRpcError(
            "GUI did not write the requested screenshot", reason="missing"
        )
    return {"path": str(path)}


SCREENSHOT_TOOL: dict[str, Any] = {
    "handler": screenshot,
    "description": "Return {path} to a session-owned PNG of the requested current GUI window/dialog without changing focus. Path remains readable for the MCP session's lifetime. Closed targets: window, setup, device, predictor, inspect, arb_waveform.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "default": "window",
                "enum": [
                    "window",
                    "setup",
                    "device",
                    "predictor",
                    "inspect",
                    "arb_waveform",
                ],
            }
        },
    },
}


def build_screenshot_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {"screenshot": {**SCREENSHOT_TOOL, "handler": partial(screenshot, ctx)}}
