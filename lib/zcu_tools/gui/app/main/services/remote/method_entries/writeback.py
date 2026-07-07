"""Writeback remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import (
    _expected_versions,
    _str,
    _str_opt,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "tab.writeback_preview",
        "writeback:_h_tab_writeback_preview",
        MethodSpec(
            5.0,
            "List the tab's persistent writeback draft (pure read — not a dry-run; the "
            "draft was computed once at analyze time). Returns {has_draft, items}; "
            "has_draft is false before any analyze produced a draft. Each item: id "
            "(<kind>-<n>, kind∈md|ml|wf), target_name (apply destination, editable), "
            "kind (metadict|module|waveform), description, selected; metadict adds "
            "proposed_value; module/waveform add editor_id + has_edit_schema, and "
            "may include role_id when the proposal corresponds to a ModuleLibrary "
            "role. A complex metadict proposed_value is carried as "
            '{"__complex__": [re, im]} (JSON has no complex). Edit an item via '
            "gui_tab_writeback_set_item; the user's Edit dialog renders the same "
            "model (WYSIWYG).",
            (_str("tab_id"),),
            tool_name="gui_tab_writeback_list",
        ),
    ),
    method_entry(
        "tab.writeback_set",
        "writeback:_h_tab_writeback_set",
        MethodSpec(
            5.0,
            "Edit a persistent writeback item by id — the single writeback editing "
            "surface. selected? / target_name? apply to any item. proposed_value? is "
            "the METADICT-only facet (a complex value is passed as "
            '{"__complex__": [re, im]}, the same shape the list emits; it applies as '
            "a Python complex). edits? is the MODULE/WAVEFORM-only facet: an ORDERED "
            "list of {path, value} cfg edits applied to the item's draft (no editor_id "
            "needed — the surface resolves it internally). Apply ref-switch edits "
            "before dependent inner-path edits (a ref switch removes child paths); "
            "fail-fast and non-atomic. proposed_value and edits are mutually exclusive "
            "(different item kinds). Echoes the edited {item}; an edits batch also "
            "returns {valid, removed, added} aggregated across the batch (same shape "
            "as tab.set_cfg). Read the item's current paths via tab.writeback_preview.",
            (
                _str("tab_id"),
                _str("id", "writeback item session id (<kind>-<n>)"),
                # Boolean (not JSON): a JSON schema of {type: boolean} makes the
                # client send a real boolean. Declared as JSON, the client may send
                # the string "false", which ``bool("false")`` wrongly reads as True.
                ParamSpec("selected", JsonType.BOOLEAN, required=False),
                _str_opt("target_name", "new apply destination name"),
                ParamSpec(
                    "proposed_value",
                    JsonType.JSON,
                    required=False,
                    description="Proposed metadict scalar (metadict items only)",
                ),
                ParamSpec(
                    "edits",
                    JsonType.JSON,
                    required=False,
                    description="Ordered list of {path, value} cfg edits "
                    "(module/waveform items only)",
                ),
                _expected_versions(),
            ),
            tool_name="gui_tab_writeback_set_item",
        ),
    ),
    method_entry(
        "tab.writeback_apply",
        "writeback:_h_tab_writeback_apply",
        MethodSpec(
            10.0,
            "Apply the tab's persistent writeback draft as-is (edit it first via "
            "gui_tab_writeback_set_item). Applies items currently selected. Returns "
            "{applied_ids, written, context_version}: written lists the destination "
            "names actually pushed, split by kind ({md, ml_modules, ml_waveforms}); "
            "context_version is the bumped 'context' resource version after apply (use "
            "it as an expected_versions guard on a follow-up write).",
            (_str("tab_id"), _expected_versions()),
        ),
    ),
)
