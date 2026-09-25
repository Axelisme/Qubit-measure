"""Analysis remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import McpMethodPolicy, MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import (
    _expected_versions,  # pyright: ignore[reportPrivateUsage] - package-local ParamSpec helper
    _obj_default,
    _str,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "analyze.cancel",
        "analysis:_h_analyze_cancel",
        MethodSpec(
            5.0,
            "Cancel the tab's in-flight (interactive) analyze: settle its handle as "
            "cancelled and clear is_analyzing so the tab can then be closed. This is "
            "the agent-side counterpart of the GUI 'Done' button for an interactive "
            "picker — interactive analyze is a separate operation from run, so "
            "gui_tab_run_cancel does NOT settle it. This cancel is op-specific (an "
            "interactive analyze needs View teardown that no generic handle cancel can "
            "do — ADR-0026 §8). Returns {ok, cancelled}: ok is always true (the call "
            "succeeded); cancelled is true when an interactive analyze was settled, or "
            "false (a graceful no-op) when none was in flight.",
            (_str("tab_id"),),
            tool_name="gui_tab_analyze_cancel",
        ),
    ),
    method_entry(
        "tab.get_analyze_result",
        "analysis:_h_tab_get_analyze_result",
        MethodSpec(5.0, "Read tab analyze result scalar summary", (_str("tab_id"),)),
    ),
    method_entry(
        "tab.get_analyze_params",
        "analysis:_h_tab_get_analyze_params",
        MethodSpec(5.0, "Read current analyze params", (_str("tab_id"),)),
    ),
    method_entry(
        "tab.analyze",
        "analysis:_h_tab_analyze",
        MethodSpec(
            30.0,
            "Start analyzing the tab's run result. Analyze runs on a worker thread "
            "and returns an operation_id (like run/connect/device); the mcp "
            "gui_tab_analyze_start tool awaits it so the agent sees one synchronous "
            "call. 'updates' optionally overrides analyze params (read the current "
            "ones with gui_tab_get_analyze_params). Makes the tab busy while it runs; "
            "a concurrent save/edit returns precondition_failed until it settles. "
            "Read the fit summary with gui_tab_get_analyze_result.",
            (_str("tab_id"), _obj_default("updates", "Analyze param updates")),
            mcp=McpMethodPolicy.override(
                "gui_tab_analyze_start",
                reason="manual MCP tool adds short-wait handle and fit-result folding",
            ),
        ),
    ),
    method_entry(
        "tab.interact",
        "interactive:_h_tab_interact",
        MethodSpec(
            30.0,
            "Read or execute one command on the tab's active interactive analysis. "
            "Omit payload to discover committed state and commands; use "
            "{command, args} for one validated action, or command=done to settle "
            "the existing analysis operation. The GUI-local preview never becomes "
            "the returned state; figure may show it and preview_active reports it.",
            (
                _str("tab_id"),
                ParamSpec("payload", JsonType.OBJECT, required=False),
                _expected_versions(),
            ),
            mcp=McpMethodPolicy.internal(
                reason="GUI-side contract only; agent-facing tool is outside this task"
            ),
        ),
    ),
    method_entry(
        "tab.get_post_analyze_result",
        "analysis:_h_tab_get_post_analyze_result",
        MethodSpec(
            5.0, "Read tab post-analysis result scalar summary", (_str("tab_id"),)
        ),
    ),
    method_entry(
        "tab.get_post_analyze_params",
        "analysis:_h_tab_get_post_analyze_params",
        MethodSpec(5.0, "Read current post-analysis params", (_str("tab_id"),)),
    ),
    method_entry(
        "tab.post_analyze",
        "analysis:_h_tab_post_analyze",
        MethodSpec(
            30.0,
            "Start the second-layer (post) analysis on the tab's PRIMARY analyze "
            "result. Runs on a worker thread and returns an operation_id (like "
            "tab.analyze); the mcp gui_tab_post_analyze_start tool awaits it so the "
            "agent sees one synchronous call. Fast-fails with precondition_failed when the "
            "tab has no primary analyze result to build on. 'updates' optionally "
            "overrides post params (see gui_tab_get_post_analyze_params). Read the "
            "fit summary with gui_tab_get_post_analyze_result.",
            (_str("tab_id"), _obj_default("updates", "Post-analysis param updates")),
            mcp=McpMethodPolicy.override(
                "gui_tab_post_analyze_start",
                reason="manual MCP tool adds short-wait handle and summary folding",
            ),
        ),
    ),
)
