"""Analysis remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _obj_default,
    _str,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

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
            "run cancellation does NOT settle it. Call this via rpc_call for "
            "interactive View teardown that generic handle cancellation cannot "
            "do (ADR-0026 §8). Returns {ok, cancelled}: ok is always true (the call "
            "succeeded); cancelled is true when an interactive analyze was settled, or "
            "false (a graceful no-op) when none was in flight.",
            (_str("tab_id"),),
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
            "Start analyzing the tab's run result via rpc_call. Runs on a worker "
            "thread; the GUI returns a raw operation_id, which MCP exposes as a "
            "handle, not a completed result. Call wait(op=handle) for the terminal "
            "status, error and Stop feedback. 'updates' optionally overrides "
            "analyze params (read current values with rpc_call on "
            "tab.get_analyze_params). Makes the tab busy while it runs; a "
            "concurrent save/edit returns precondition_failed until it settles. "
            "Read the fit summary with rpc_call on tab.get_analyze_result.",
            (_str("tab_id"), _obj_default("updates", "Analyze param updates")),
        ),
        agent=AgentMethodPolicy(
            operation_key="analyze:{tab_id}", refresh_after_write=True
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
            "Start post analysis on the tab's PRIMARY analyze result via "
            "rpc_call. Runs on a worker thread; MCP maps the GUI operation_id "
            "to a handle. Call wait(op=handle) for the terminal status, error and "
            "Stop feedback; start alone does not mean completion. Fast-fails "
            "with precondition_failed when no primary result exists. 'updates' "
            "overrides post params (read current values with rpc_call on "
            "tab.get_post_analyze_params). Read the fit summary with rpc_call on "
            "tab.get_post_analyze_result.",
            (_str("tab_id"), _obj_default("updates", "Post-analysis param updates")),
        ),
        agent=AgentMethodPolicy(
            operation_key="post_analyze:{tab_id}", refresh_after_write=True
        ),
    ),
)
