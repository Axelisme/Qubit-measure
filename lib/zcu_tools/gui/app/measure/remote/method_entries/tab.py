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
            "Create a new tab using current global aliases. Unbound aliases allow "
            "creation with an Invalid cfg. The tab retains no source capture; "
            "named alias overrides belong to a single Run. Returns {tab_id}.",
            (required_string("adapter_name", "Adapter to instantiate"),),
        ),
    ),
    method_entry(
        "tab.open_file",
        "tab:h_tab_open_file",
        MethodSpec(
            30.0,
            "Create a tab, load a result file without a SoC, and focus it. "
            "A load failure closes the new "
            "tab and restores prior focus. Returns the load outcome including "
            "tab_id, cfg_backfill and analysis_error. Failed analysis preparation "
            "retains the loaded result and the new tab. not_applied retains the result.",
            (required_string("adapter_name"), required_string("data_path")),
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
            exposure="tool",
            tool_names=("tab_close",),
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
            readonly=True,
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
            "only an index.",
            (optional_string("tab_id", "Tab to inspect; omit for all tabs"),),
            readonly=True,
        ),
    ),
    method_entry(
        "tab.parameter_read",
        "tab:h_tab_parameter_read",
        MethodSpec(
            5.0,
            "Read only requested paths and module_refs from current point/library/aliases "
            "for this tab. Optional aliases maps names to concrete components for this read only. None and {} mean no override. "
            "Returns {values, modules}: absent values are "
            "omitted, null is retained; each module ref maps to an existing "
            "canonical library root or null for missing/custom. Invalid paths, "
            "unknown roles and source failures propagate.",
            (
                required_string("tab_id"),
                ParamSpec("paths", JsonType.ARRAY, required=True),
                ParamSpec("module_refs", JsonType.ARRAY, required=True),
                ParamSpec(
                    "aliases",
                    JsonType.OBJECT,
                    required=False,
                    description="Temporary alias-name to component-name overrides; None and {} mean no override",
                ),
            ),
            readonly=True,
        ),
    ),
    method_entry(
        "tab.get_cfg",
        "tab:h_tab_get_cfg",
        MethodSpec(
            5.0,
            "Return a complete cached resource publication: cfg_ref, status, tree, "
            "source_basis and diagnostics. Ordinary reads do not query sources. "
            "Nonempty aliases requires expected cfg_ref and resolves current inputs "
            "against named overrides without publication. None and {} read cached cfg. Stale "
            "refs reject; the returned ref never refreshes to a newer revision. "
            "Use cfg_ref as expected for tab.edit_cfg or tab.reset_cfg. Paths are arrays of field "
            "segments; writes use the closed __expr/__text/__complex/__ref codec. "
            "Library editor observations use their own independent interface.",
            (
                required_string("tab_id"),
                ParamSpec(
                    "aliases",
                    JsonType.OBJECT,
                    required=False,
                    description="Temporary alias-name to component-name overrides; None and {} mean no override",
                ),
                ParamSpec("expected", JsonType.OBJECT),
            ),
            readonly=True,
        ),
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
    ),
)
