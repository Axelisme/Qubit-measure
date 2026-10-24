"""Analysis remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import (
    default_object,
    optional_integer,
    required_string,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "tab.get_analyze_result",
        "analysis:h_tab_get_analyze_result",
        MethodSpec(
            5.0,
            "Read the current scalar summary. Optional "
            "operation_id requires that operation's result and also returns complete "
            "operation_state, including raw_data {run_id, saved}. Analysis output "
            "saving does not imply raw data was saved.",
            (required_string("tab_id"), optional_integer("operation_id")),
            readonly=True,
        ),
        agent=AgentMethodPolicy(
            operation_key="analyze:{tab_id}",
        ),
    ),
    method_entry(
        "tab.analyze",
        "analysis:h_tab_analyze",
        MethodSpec(
            30.0,
            "Start analyzing the tab's run result via tab_analyze. Returns "
            "operation_id, interactive, complete params, raw_data {run_id, saved} "
            "and invalidated_on_success "
            "(analysis.writeback/post.result/post.writeback when previously present). "
            "Invalidation applies only on successful completion. Runs on a worker "
            "thread; the GUI returns a raw operation_id, which MCP exposes as a "
            "handle, not a completed result. Call wait(op=handle) for the terminal "
            "status, error and Stop feedback. 'updates' optionally overrides "
            "analyze params. Makes the tab busy while it runs; a "
            "concurrent save/edit returns precondition_failed until it settles. "
            "Run completion does not save raw data; call tab.save_data explicitly. "
            "Analysis does not save raw data either. "
            "Read the fit summary with rpc_call on tab.get_analyze_result. "
            "Optional run_operation_id requires that Run result before following "
            "the pane or starting; a replaced result returns result_superseded.",
            (
                required_string("tab_id"),
                default_object("updates", "Analyze param updates"),
                optional_integer("run_operation_id"),
            ),
        ),
        agent=AgentMethodPolicy(
            exposure="tool",
            tool_names=("tab_analyze",),
            operation_key="analyze:{tab_id}",
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
            "the returned state; figure may show it and preview_active reports it. "
            "Returns the original analysis operation_id, including after done. "
            "Wait on that operation for the actual terminal outcome. "
            "Set include_figure=false for a receipt without PNG rendering.",
            (
                required_string("tab_id"),
                ParamSpec("payload", JsonType.OBJECT, required=False),
                ParamSpec(
                    "include_figure", JsonType.BOOLEAN, required=False, default=True
                ),
            ),
        ),
        agent=AgentMethodPolicy(
            exposure="tool",
            tool_names=("tab_interact",),
            operation_key="analyze:{tab_id}",
        ),
    ),
    method_entry(
        "tab.get_post_analyze_result",
        "analysis:h_tab_get_post_analyze_result",
        MethodSpec(
            60.0,
            "Read the current post-analysis scalar summary. "
            "Optional operation_id requires that operation's result and also returns complete "
            "operation_state. The handler ceiling is 60 seconds; Post start "
            "receipts provide the per-result read budget. After a read timeout, "
            "check tab.snapshot source_operation_id and read with the original "
            "operation_id; a replaced result returns result_superseded.",
            (required_string("tab_id"), optional_integer("operation_id")),
            readonly=True,
        ),
        agent=AgentMethodPolicy(
            operation_key="post_analyze:{tab_id}",
        ),
    ),
    method_entry(
        "tab.post_analyze",
        "analysis:h_tab_post_analyze",
        MethodSpec(
            30.0,
            "Start post analysis on the tab's PRIMARY analyze result via tab_analyze. "
            "Returns operation_id, result_read_timeout_seconds (5 seconds plus "
            "5 per measured tone, capped at 60), source-bound recovery, "
            "interactive=false, complete params and "
            "invalidated_on_success (post.writeback when previously present). "
            "Invalidation applies only on successful completion. "
            "Runs on a worker thread; MCP maps the GUI operation_id "
            "to a handle. Call wait(op=handle) for the terminal status, error and "
            "Stop feedback; start alone does not mean completion. Fast-fails "
            "with precondition_failed when no primary result exists. 'updates' "
            "overrides post params. Read the fit summary with rpc_call on "
            "tab.get_post_analyze_result.",
            (
                required_string("tab_id"),
                default_object("updates", "Post-analysis param updates"),
                optional_integer("operation_id", "Required Primary analysis source"),
                optional_integer("run_operation_id", "Required Run result source"),
            ),
        ),
        agent=AgentMethodPolicy(
            exposure="tool",
            tool_names=("tab_analyze",),
            operation_key="post_analyze:{tab_id}",
        ),
    ),
)
