"""Measure MCP tools-screenshot override tools."""

from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir
from typing import Any

from zcu_tools.mcp.measure.tool_context import (
    MeasureToolContext,
    bind_context,
    send_gui_rpc,
)

_SCREENSHOT_DIALOGS = ("setup", "device", "predictor", "inspect", "startup")


_SCREENSHOT_TARGETS = frozenset({"window", *_SCREENSHOT_DIALOGS})


def tool_gui_screenshot(arguments: dict[str, Any]) -> dict[str, Any]:
    """Capture the main window OR a named dialog as a PNG FILE; return its path.

    ``target`` switches what is grabbed:
      - target='window' → the WHOLE main window (client area + non-dialog floating
        widgets) via the view.screenshot wire method.
      - target=<dialog name> → that named dialog via dialog.screenshot.

    Mirrors gui_tab_get_current_figure / the old dialog grab: the convenience
    layer never returns inline base64 (a full-window grab would blow the
    tool-output token budget — the footgun this override removes). Both wire
    methods return base64 for raw consumers; we decode + write here. When out_path
    is omitted we synthesise a per-target temp path under gettempdir() (a single
    measure_window.png for the window — there is only one — and a per-dialog
    measure_dialog_<name>.png for a dialog), overwriting the previous grab.
    """
    import base64

    target = str(arguments["target"])
    # Client-side validation: reject an unknown target fast with the allowed set
    # rather than letting an invalid dialog name reach (and fail at) the wire.
    if target not in _SCREENSHOT_TARGETS:
        raise ValueError(
            f"target must be one of {sorted(_SCREENSHOT_TARGETS)}, got {target!r}"
        )
    out_path_arg = arguments.get("out_path")
    if target == "window":
        method, params = "view.screenshot", {}
        default_name = "measure_window.png"
    else:
        method, params = "dialog.screenshot", {"name": target}
        default_name = f"measure_dialog_{target}.png"
    out_path = (
        str(out_path_arg)
        if out_path_arg is not None
        else str(Path(gettempdir()) / default_name)
    )
    res = send_gui_rpc(method, params)
    png = base64.b64decode(res["png_b64"])
    Path(out_path).write_bytes(png)
    return {"bytes": res.get("bytes", len(png)), "saved_to": out_path}


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_screenshot": {
        "handler": tool_gui_screenshot,
        "description": (
            "Capture the GUI to a PNG FILE and return its path. 'target' selects "
            "what to grab:\n"
            "  - target='window': the WHOLE main window — its client area AND the "
            "non-dialog floating widgets (the feedback widget, the left-edge "
            "handle) that a per-dialog grab cannot see.\n"
            "  - target=<dialog name> (one of: setup, device, predictor, inspect, "
            "arb_waveform, startup): that dialog; fails PRECONDITION_FAILED if it is not "
            "currently open.\n"
            "The PNG is ALWAYS written to disk and the reply is {saved_to, bytes} — "
            "Read the saved_to path to view it (never inline base64, so it cannot "
            "blow the token budget). Omit out_path to write a per-target file under "
            "the temp dir (overwritten each call); pass out_path (absolute) to "
            "choose the location.\n"
            "Timing note: a floating widget repositions via QTimer.singleShot, so a "
            "screenshot taken in the same turn as a UI change may catch a "
            "pre-reposition frame — do a wire round-trip (any read tool) first, or "
            "re-grab, if a widget looks mislaid."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": [
                        "window",
                        "setup",
                        "device",
                        "predictor",
                        "inspect",
                        "arb_waveform",
                        "startup",
                    ],
                    "description": (
                        "'window' for the whole main window, or a dialog name "
                        "(setup, device, predictor, inspect, arb_waveform, startup)"
                    ),
                },
                "out_path": {
                    "type": "string",
                    "description": (
                        "Optional absolute path to write the PNG; omit to use a "
                        "per-target file under the temp dir"
                    ),
                },
            },
            "required": ["target"],
        },
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
