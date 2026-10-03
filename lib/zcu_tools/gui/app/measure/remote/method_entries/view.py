"""View remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import (
    optional_integer,
    optional_string,
    required_string,
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
            (required_string("adapter_name", "Adapter to introspect"),),
        ),
    ),
    method_entry(
        "app.shutdown",
        "view:h_app_shutdown",
        MethodSpec(
            5.0,
            "Gracefully close an idle GUI. Any active operation returns busy. "
            "All unsaved artifacts require discard_unsaved=true. Persist session "
            "and clean up through the normal shutdown path after this reply. No OS kill.",
            (
                ParamSpec(
                    "discard_unsaved", JsonType.BOOLEAN, required=False, default=False
                ),
            ),
        ),
        agent=AgentMethodPolicy(exposure="tool", tool_names=("shutdown",)),
    ),
    method_entry(
        "dialog.screenshot",
        "view:h_dialog_screenshot",
        MethodSpec(
            10.0,
            "Capture a named dialog as base64 PNG, or write PNG to out_path and return its path/byte count.",
            (
                required_string("name", "Dialog name"),
                optional_string(
                    "out_path", "Write PNG here instead of returning base64"
                ),
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
            (
                optional_string(
                    "out_path", "Write PNG here instead of returning base64"
                ),
            ),
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
            "small geometry (token-light), independent of the GUI window size. "
            "Optional operation_id applies only to analysis panes and rejects replaced results. "
            "Optional run_operation_id binds the run pane to its original Run. "
            "The two operation tokens are mutually exclusive.",
            (
                required_string("tab_id"),
                required_string("subtab_id", "Pane: run|analysis|post_analysis"),
                optional_integer(
                    "operation_id", "Require this analysis operation's current result"
                ),
                optional_integer(
                    "run_operation_id", "Require this Run's result for the run pane"
                ),
                optional_string(
                    "out_path", "Write PNG here instead of returning base64"
                ),
            ),
        ),
    ),
)
