"""View remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _str,
    _str_opt,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "adapter.list",
        "view:_h_adapter_list",
        MethodSpec(5.0, "List available adapters. Returns {adapters: [name]}."),
    ),
    method_entry(
        "adapter.guide",
        "view:_h_adapter_guide",
        MethodSpec(
            5.0,
            "Read an adapter's human-facing orientation guide BEFORE running it: "
            "prose (not a contract) on {behavior, expects_md, expects_ml, "
            "typical_writeback, recommended} — what the experiment measures, what it "
            "assumes is already in the MetaDict/ModuleLibrary, what a run tends to "
            "write back, and recommended analysis settings. How you actually use it "
            "is your call. Empty fields mean the adapter has no guide written yet.",
            (_str("adapter_name", "Adapter to introspect"),),
        ),
    ),
    method_entry(
        "app.shutdown",
        "view:_h_app_shutdown",
        MethodSpec(
            5.0,
            "Gracefully close the GUI: runs the normal window-close path (persist "
            "session, disconnect devices, cleanup) — the same as a user closing the "
            "window. Returns immediately; the close happens just after. No OS kill. "
            "Prefer this over gui_stop's force path to stop a GUI cleanly.",
        ),
    ),
    method_entry(
        "dialog.screenshot",
        "view:_h_dialog_screenshot",
        MethodSpec(
            10.0,
            "Capture a named dialog as base64 PNG",
            (_str("name", "Dialog name"),),
        ),
    ),
    method_entry(
        "view.snapshot",
        "view:_h_view_snapshot",
        MethodSpec(5.0, "Capture view state summary"),
    ),
    method_entry(
        "view.screenshot",
        "view:_h_view_screenshot",
        MethodSpec(
            10.0,
            "Capture the WHOLE main window (client area + floating widgets) as base64 "
            "PNG. Runs MainWindow.grab() on the main thread (auto-marshalled, like "
            "dialog.screenshot).",
        ),
    ),
    method_entry(
        "tab.get_current_figure",
        "view:_h_tab_get_current_figure",
        MethodSpec(
            10.0,
            "Get the tab's current figure (run 2D map, or analysis fit) as PNG. The "
            "PNG is rendered at a fixed small geometry (token-light), independent of "
            "the GUI window size; the live figure is never permanently resized.",
            (
                _str("tab_id"),
                _str_opt("out_path", "Write PNG here instead of returning base64"),
            ),
        ),
    ),
)
