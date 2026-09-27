"""Tab remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ..cfg_observation import CFG_OBSERVATION_DESCRIPTION
from ._params import (
    _json,
    _str,
    _str_opt,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "tab.new",
        "tab:_h_tab_new",
        MethodSpec(
            10.0,
            "Create a new tab for the named adapter. Returns {tab_id}.",
            (_str("adapter_name", "Adapter to instantiate"),),
        ),
        agent=AgentMethodPolicy(
            refresh_after_write=True, created_resource="tab:{tab_id}"
        ),
    ),
    method_entry(
        "tab.close",
        "tab:_h_tab_close",
        MethodSpec(5.0, "Close a tab. Returns {ok: true}.", (_str("tab_id"),)),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "tab.set_active",
        "tab:_h_tab_set_active",
        MethodSpec(
            5.0,
            "Activate a tab. VIEW-ONLY: this changes which tab the user sees, NOT your "
            "operation target (you always act on an explicit tab_id). Returns {ok: true}.",
            (_str("tab_id"),),
        ),
    ),
    method_entry(
        "tab.list_all",
        "tab:_h_tab_list_all",
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
        "tab:_h_tab_snapshot",
        MethodSpec(
            5.0,
            "Tab summary. Pass tab_id to fully inspect this tab's existence; "
            "the all-tabs summary is only an index and does not refresh a "
            "per-tab guard baseline.",
            (_str_opt("tab_id", "Tab to inspect; omit for all tabs"),),
        ),
        agent=AgentMethodPolicy(
            reveals=("tab:{tab_id}",), reveals_when_nonempty=("tab_id",)
        ),
    ),
    method_entry(
        "tab.get_cfg",
        "tab:_h_tab_get_cfg",
        MethodSpec(
            5.0,
            CFG_OBSERVATION_DESCRIPTION,
            (
                _str("tab_id"),
                _str_opt(
                    "prefix",
                    "Return only the sub-tree rooted at this dotted path "
                    "(e.g. 'modules.readout'); omit for the whole cfg. No match → {}",
                ),
            ),
        ),
        agent=AgentMethodPolicy(
            reveals=("tab:{tab_id}:cfg",), reveals_without=("prefix",)
        ),
    ),
    method_entry(
        "tab.set_cfg",
        "tab:_h_tab_set_cfg",
        MethodSpec(
            5.0,
            "Batch-set canonical cfg paths on a tab in order (fail-fast, non-atomic). "
            "Copy paths from tab.get_cfg: agent_edit=true accepts only whole sweep "
            "objects at '<path>' (not edge paths); the default GUI leaf grammar "
            "accepts sweep edges '<path>.<edge>'. Reference keys are '<path>.ref', "
            "and reference children descend directly. Removed "
            "'.sweep.*' / '.value.*' aliases are rejected without mutation and name "
            "their replacement. 'edits' is an ORDERED list of {path, value} objects. "
            "Apply ref-switch edits "
            "before dependent inner-path edits (a ref switch removes child paths). "
            "'value' is a JSON scalar, an md-reference eval tag "
            '{"__kind":"eval","expr":"r_f"}, or a registered value-source tag '
            '{"__kind":"value_ref","key":"device.flux.value","type":"float"}; '
            "value_ref is resolved immediately at set time and stored as a direct "
            "scalar. Discover keys with value.list / value.read. "
            "Returns {valid, removed, added}; removed/added are the final net path-set "
            "difference before versus after the successful whole batch (A→B→A is "
            "empty), not transient churn. A tab that is currently running is rejected "
            "(cancel the run first). Use tab.get_cfg to read the current tree.",
            (
                _str("tab_id"),
                _json("edits", "Ordered list of {path, value} edits"),
                ParamSpec(
                    "agent_edit",
                    JsonType.BOOLEAN,
                    required=False,
                    default=False,
                    description="Agent whole-sweep grammar; GUI leaf edits remain unchanged",
                ),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
)
