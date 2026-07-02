"""Tab remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import McpMethodPolicy, MethodSpec

from ._params import (
    _json,
    _str,
    _str_opt,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "tab.new",
        "tab:_h_tab_new",
        MethodSpec(
            10.0,
            "Create a new tab for the named adapter. Returns {tab_id}.",
            (_str("adapter_name", "Adapter to instantiate"),),
        ),
    ),
    method_entry(
        "tab.close",
        "tab:_h_tab_close",
        MethodSpec(5.0, "Close a tab. Returns {ok: true}.", (_str("tab_id"),)),
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
            tool_name="gui_tab_list",
        ),
    ),
    method_entry(
        "tab.snapshot",
        "tab:_h_tab_snapshot",
        MethodSpec(
            5.0,
            "Tab summary",
            (_str_opt("tab_id", "Tab to inspect; omit for all tabs"),),
        ),
    ),
    method_entry(
        "tab.get_cfg",
        "tab:_h_tab_get_cfg",
        MethodSpec(
            5.0,
            "Read the tab's settable cfg as a NESTED tree of current values (the "
            "read-only view; edit a leaf with tab.set_cfg or editor.set_field on the "
            "tab's editor_id from tab.snapshot, using the leaf's dotted path). Node "
            "shape, distinguished by '$'-prefixed reserved keys: a SCALAR leaf is "
            "its bare current value (null = unset); an ENUM scalar leaf is "
            "{'$value': current, '$choices': [...]}; a SWEEP is a sub-tree of bare "
            "edges {start, stop, expts, step} (each edge accepts ONLY a number/int "
            "via tab.set_cfg — NOT an eval/ref); a REF node "
            "(module/waveform/device selector) is {'$ref': {'current': <chosen>, "
            "'options': [<names>]}, <chosen variant's settable sub-tree>} — only the "
            "CURRENTLY-CHOSEN variant is expanded; 'options' lists bare names while "
            "'current' may be a tagged internal key — switch by passing a bare "
            "'options' name to tab.set_cfg on the ref's dotted path. "
            "Any other dict is a plain section sub-tree (its keys are child fields). "
            "'prefix' (optional, dotted) returns just the sub-tree rooted at that "
            "node (a prefix at a sweep edge returns the whole sweep node); a prefix "
            "matching nothing returns {}.",
            (
                _str("tab_id"),
                _str_opt(
                    "prefix",
                    "Return only the sub-tree rooted at this dotted path "
                    "(e.g. 'modules.readout'); omit for the whole cfg. No match → {}",
                ),
            ),
        ),
    ),
    method_entry(
        "tab.set_cfg",
        "tab:_h_tab_set_cfg",
        MethodSpec(
            5.0,
            "Batch-set cfg fields on a tab in order (fail-fast, non-atomic). 'edits' "
            "is an ORDERED list of {path, value} objects. Apply ref-switch edits "
            "before dependent inner-path edits (a ref switch removes child paths). "
            "'value' is a JSON scalar, an md-reference eval tag "
            '{"__kind":"eval","expr":"r_f"}, or a registered value-source tag '
            '{"__kind":"value_ref","key":"device.flux.value","type":"float"}; '
            "value_ref is resolved immediately at set time and stored as a direct "
            "scalar. Discover keys with value.list / value.read. "
            "Returns {valid, removed, added} aggregated across the batch — the same "
            "shape as editor.set_field. A tab that is currently running is rejected "
            "(cancel the run first). Use tab.get_cfg to read the current tree.",
            (
                _str("tab_id"),
                _json("edits", "Ordered list of {path, value} edits"),
            ),
            mcp=McpMethodPolicy.override(
                "gui_tab_set_cfg",
                reason="batch MCP tool preserves ordered edits and untyped JSON value schema",
            ),
        ),
    ),
)
