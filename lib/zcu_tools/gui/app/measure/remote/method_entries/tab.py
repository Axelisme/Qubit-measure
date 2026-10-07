"""Tab remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import (
    optional_string,
    required_json,
    required_object,
    required_string,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "tab.new",
        "tab:h_tab_new",
        MethodSpec(
            10.0,
            "Create a new tab for the named adapter. Returns {tab_id}.",
            (required_string("adapter_name", "Adapter to instantiate"),),
        ),
        agent=AgentMethodPolicy(
            refresh_after_write=True,
            created_resource="tab:{tab_id}",
            created_identity="tab_id",
        ),
    ),
    method_entry(
        "tab.open_file",
        "tab:h_tab_open_file",
        MethodSpec(
            30.0,
            "Create a tab, load a result file without a SoC, and focus it. "
            "Read context.snapshot explicitly first. A load failure closes the new "
            "tab and restores prior focus. Returns the load outcome including "
            "tab_id, cfg_backfill and analysis_error. Failed analysis preparation "
            "retains the loaded result and the new tab. not_applied retains the result. "
            "Read tab.snapshot and tab.get_cfg before subsequent guarded writes.",
            (required_string("adapter_name"), required_string("data_path")),
        ),
        agent=AgentMethodPolicy(
            guard_deps=("context",),
            refresh_after_write=True,
            created_resource="tab:{tab_id}",
            created_identity="tab_id",
        ),
    ),
    method_entry(
        "tab.close",
        "tab:h_tab_close",
        MethodSpec(
            5.0,
            "Close an idle tab. All unsaved artifacts require discard_unsaved=true; "
            "busy operations cannot be discarded. Returns {ok: true}.",
            (
                required_string("tab_id"),
                ParamSpec(
                    "discard_unsaved", JsonType.BOOLEAN, required=False, default=False
                ),
            ),
        ),
        agent=AgentMethodPolicy(
            exposure="tool", tool_names=("tab_close",), refresh_after_write=True
        ),
    ),
    method_entry(
        "tab.set_active",
        "tab:h_tab_set_active",
        MethodSpec(
            5.0,
            "Activate a tab. VIEW-ONLY: this changes which tab the user sees, NOT your "
            "operation target (you always act on an explicit tab_id). Returns {ok: true}.",
            (required_string("tab_id"),),
        ),
    ),
    method_entry(
        "tab.list_all",
        "tab:h_tab_list_all",
        MethodSpec(
            5.0,
            "List all open tabs. Returns {tabs, active_tab_id, running_tab_id}: tabs "
            "is a list of {tab_id, adapter_name, is_running} objects; active_tab_id is "
            "the tab the USER is focused on (a collaboration cue, NOT your operation "
            "target); running_tab_id is the tab currently running (or null when "
            "nothing is running).",
        ),
    ),
    method_entry(
        "tab.snapshot",
        "tab:h_tab_snapshot",
        MethodSpec(
            5.0,
            "Tab operation state. Pass tab_id to inspect existence, result and "
            "analysis revisions/availability, and all effective save paths. "
            "Result arrays are not required for this observation. Read writeback "
            "preview separately for proposal contents. The all-tabs summary is "
            "only an index and does not refresh a per-tab guard baseline.",
            (optional_string("tab_id", "Tab to inspect; omit for all tabs"),),
        ),
        agent=AgentMethodPolicy(
            reveals=(
                "tab:{tab_id}",
                "tab:{tab_id}:result",
                "tab:{tab_id}:analyze",
                "tab:{tab_id}:post_analyze",
                "tab:{tab_id}:path:data",
                "tab:{tab_id}:path:analysis_image",
                "tab:{tab_id}:path:post_analysis_image",
            ),
            reveals_when_nonempty=("tab_id",),
        ),
    ),
    method_entry(
        "tab.get_cfg",
        "tab:h_tab_get_cfg",
        MethodSpec(
            5.0,
            "Return a complete cached resource publication: cfg_ref, status, tree, "
            "source_basis and diagnostics. Reads never refresh or query sources. "
            "Use cfg_ref as expected for tab.edit_cfg or tab.reset_cfg. Paths are arrays of field "
            "segments; writes use the closed __expr/__text/__complex/__ref codec. "
            "Library editor observations use their own independent interface.",
            (required_string("tab_id"),),
        ),
        agent=AgentMethodPolicy(),
    ),
    method_entry(
        "tab.reset_cfg",
        "tab:h_tab_reset_cfg",
        MethodSpec(
            5.0,
            "Reset cfg to current adapter defaults at expected cfg_ref. Returns the "
            "complete new publication, which may be Invalid. Every successful command "
            "adds a revision, even if inputs are unchanged. Busy, stale and notification-"
            "reentrant resets are rejected. Does not clear results or save artifacts.",
            (
                required_string("tab_id"),
                required_object(
                    "expected", "cfg_ref from the complete resource observation"
                ),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "tab.edit_cfg",
        "tab:h_tab_edit_cfg",
        MethodSpec(
            5.0,
            "Atomically apply an ordered edit batch at expected cfg_ref. Paths are "
            "arrays of field-name segments. Scalars use __expr/__text/__complex; "
            "reference inputs use __ref and complete children; range inputs are "
            "complete objects. All edits publish together or none do. Valid "
            "incomplete input may publish Invalid. Every successful command adds "
            "a revision, including equal values. Busy, stale and notification-"
            "reentrant edits are rejected. Returns the complete new observation.",
            (
                required_string("tab_id"),
                required_object(
                    "expected", "cfg_ref from the complete resource observation"
                ),
                required_json(
                    "edits", "Ordered list of {path: [segments], value} inputs"
                ),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
)
