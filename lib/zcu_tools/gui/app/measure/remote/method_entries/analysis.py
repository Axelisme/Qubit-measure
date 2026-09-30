"""Analysis remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import (
    default_object,
    required_string,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "analyze.cancel",
        "analysis:h_analyze_cancel",
        MethodSpec(
            5.0,
            "Cancel the tab's in-flight interactive analyze and clear is_analyzing. "
            "The generic operation.cancel handle path uses the same domain "
            "cancellation hook and tears down the interactive View. Run "
            "cancellation does not settle a separate analyze operation. Returns "
            "{ok, cancelled}: cancelled is false when none was in flight; "
            "true requests cancellation, not necessarily worker completion.",
            (required_string("tab_id"),),
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "tab.get_analyze_result",
        "analysis:h_tab_get_analyze_result",
        MethodSpec(
            5.0, "Read tab analyze result scalar summary", (required_string("tab_id"),)
        ),
    ),
    method_entry(
        "tab.get_analyze_params",
        "analysis:h_tab_get_analyze_params",
        MethodSpec(
            5.0,
            "Read primary analyze params as {analyze_params, definitions}. Values "
            "are null before a result exists; definitions come from the live adapter.",
            (required_string("tab_id"),),
        ),
    ),
    method_entry(
        "tab.analyze",
        "analysis:h_tab_analyze",
        MethodSpec(
            30.0,
            "Start analyzing the tab's run result via tab_analyze. Returns "
            "operation_id, interactive, complete params and invalidated_on_success "
            "(analysis.writeback/post.result/post.writeback when previously present). "
            "Invalidation applies only on successful completion. Runs on a worker "
            "thread; the GUI returns a raw operation_id, which MCP exposes as a "
            "handle, not a completed result. Call wait(op=handle) for the terminal "
            "status, error and Stop feedback. 'updates' optionally overrides "
            "analyze params (read current values with rpc_call on "
            "tab.get_analyze_params). Makes the tab busy while it runs; a "
            "concurrent save/edit returns precondition_failed until it settles. "
            "Read the fit summary with rpc_call on tab.get_analyze_result.",
            (
                required_string("tab_id"),
                default_object("updates", "Analyze param updates"),
            ),
        ),
        agent=AgentMethodPolicy(
            exposure="tool",
            tool_names=("tab_analyze",),
            operation_key="analyze:{tab_id}",
            refresh_after_write=True,
        ),
    ),
    method_entry(
        "tab.interact",
        "interactive:h_tab_interact",
        MethodSpec(
            30.0,
            "Read or execute one command on the tab's active interactive analysis. "
            "Omit payload to discover committed state and commands; use "
            "{command, args} for one validated action, or command=done to settle "
            "the existing analysis operation. The GUI-local preview never becomes "
            "the returned state; figure may show it and preview_active reports it.",
            (
                required_string("tab_id"),
                ParamSpec("payload", JsonType.OBJECT, required=False),
            ),
        ),
        agent=AgentMethodPolicy(exposure="tool", tool_names=("tab_interact",)),
    ),
    method_entry(
        "tab.get_post_analyze_result",
        "analysis:h_tab_get_post_analyze_result",
        MethodSpec(
            5.0,
            "Read tab post-analysis result scalar summary",
            (required_string("tab_id"),),
        ),
    ),
    method_entry(
        "tab.get_post_analyze_params",
        "analysis:h_tab_get_post_analyze_params",
        MethodSpec(
            5.0,
            "Read post-analysis params as {post_analyze_params, definitions}. Values "
            "are null before post analysis; definitions come from the live adapter.",
            (required_string("tab_id"),),
        ),
    ),
    method_entry(
        "tab.post_analyze",
        "analysis:h_tab_post_analyze",
        MethodSpec(
            30.0,
            "Start post analysis on the tab's PRIMARY analyze result via tab_analyze. "
            "Returns operation_id, interactive=false, complete params and "
            "invalidated_on_success (post.writeback when previously present). "
            "Invalidation applies only on successful completion. "
            "Runs on a worker thread; MCP maps the GUI operation_id "
            "to a handle. Call wait(op=handle) for the terminal status, error and "
            "Stop feedback; start alone does not mean completion. Fast-fails "
            "with precondition_failed when no primary result exists. 'updates' "
            "overrides post params (read current values with rpc_call on "
            "tab.get_post_analyze_params). Read the fit summary with rpc_call on "
            "tab.get_post_analyze_result.",
            (
                required_string("tab_id"),
                default_object("updates", "Post-analysis param updates"),
            ),
        ),
        agent=AgentMethodPolicy(
            exposure="tool",
            tool_names=("tab_analyze",),
            operation_key="post_analyze:{tab_id}",
            refresh_after_write=True,
        ),
    ),
)
