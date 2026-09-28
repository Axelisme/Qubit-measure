"""View remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _str,
    _str_opt,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "adapter.list",
        "view:h_adapter_list",
        MethodSpec(5.0, "List available adapters. Returns {adapters: [name]}."),
    ),
    method_entry(
        "adapter.guide",
        "view:h_adapter_guide",
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
        "view:h_app_shutdown",
        MethodSpec(
            5.0,
            "Gracefully close the GUI: runs the normal window-close path (persist "
            "session, disconnect devices, cleanup) — the same as a user closing the "
            "window. Returns immediately; the close happens just after. No OS kill.",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "dialog.screenshot",
        "view:h_dialog_screenshot",
        MethodSpec(
            10.0,
            "Capture a named dialog as base64 PNG, or write PNG to out_path and return its path/byte count.",
            (
                _str("name", "Dialog name"),
                _str_opt("out_path", "Write PNG here instead of returning base64"),
            ),
        ),
    ),
    method_entry(
        "view.snapshot",
        "view:h_view_snapshot",
        MethodSpec(
            5.0,
            "Capture view state summary",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "view.screenshot",
        "view:h_view_screenshot",
        MethodSpec(
            10.0,
            "Capture the WHOLE main window (client area + floating widgets) as base64 "
            "PNG. Runs MainWindow.grab() on the main thread (auto-marshalled, like "
            "dialog.screenshot). Optional out_path writes a PNG file instead of "
            "returning base64 bytes.",
            (_str_opt("out_path", "Write PNG here instead of returning base64"),),
        ),
    ),
    method_entry(
        "tab.get_figure",
        "view:h_tab_get_figure",
        MethodSpec(
            10.0,
            "Get a tab pane's figure as PNG (subtab-qualified). Run reads the live "
            "FigureContainer (view-only, not canonical); analysis/post read their "
            "canonical figures from State. Requires (tab_id, subtab_id) with closed "
            "values run|analysis|post_analysis. The PNG is rendered at a fixed "
            "small geometry (token-light), independent of the GUI window size.",
            (
                _str("tab_id"),
                _str("subtab_id", "Pane: run|analysis|post_analysis"),
                _str_opt("out_path", "Write PNG here instead of returning base64"),
            ),
        ),
    ),
)
