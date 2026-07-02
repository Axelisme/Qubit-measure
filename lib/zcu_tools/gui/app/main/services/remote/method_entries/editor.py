"""Editor remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _expected_versions,
    _json,
    _str,
    _str_opt,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "editor.new",
        "editor:_h_editor_new",
        MethodSpec(
            5.0,
            "Open a stateful editing session over an EXISTING ModuleLibrary "
            "module/waveform (by 'from_name'). To create a new blank/shaped entry, "
            "use context.ml_create_from_role (e.g. role_id='pulse:blank' or a named role) "
            "then editor.new(from_name=name) to edit it. item_kind is 'module' or "
            "'waveform'. Returns {editor_id, tree} (tree = the nested current-value "
            "view, same shape as editor.get / tab.get_cfg).",
            (
                _str("item_kind", "'module' or 'waveform'"),
                _str("from_name", "Existing ml entry name to load for editing"),
            ),
        ),
    ),
    method_entry(
        "editor.set_field",
        "editor:_h_editor_set_field",
        MethodSpec(
            5.0,
            "Set one field in an editing session. 'path' is a dotted path from "
            "editor.new/get (ModuleRef sub-fields descend directly, no 'value' "
            "segment); 'value' is a JSON scalar, or an md-reference expression as "
            '{"__kind":"eval","expr":"r_f - 0.1"} (resolved against MetaDict at '
            "commit), or a registered value source as "
            '{"__kind":"value_ref","key":"device.flux.value","type":"float"} '
            "(resolved immediately at set time and stored as a direct scalar; discover "
            "keys with value.list / value.read). NOTE: eval/value_ref forms are "
            "accepted ONLY on a scalar leaf — a "
            "sweep_edge (a sweep's start/stop/expts/step) accepts ONLY a number/int, "
            "never an eval/value_ref; an adapter's default eval edge cannot be "
            "overwritten this way, pass a numeric value instead. "
            "Returns {valid, removed, added} — does NOT echo cfg content "
            "(that would force a lowering pass that eagerly evaluates EvalValue). "
            "'valid' is whether the whole draft is currently valid; 'removed'/'added' "
            "list settable paths a ModuleRef key switch ('<path>.ref') dropped/"
            "created so you need not re-list after a variant switch. To read cfg use "
            "tab.get_cfg / editor.get (the nested current-value tree).",
            (
                _str("editor_id"),
                _str("path", "Dotted field path"),
                _json(
                    "value",
                    "JSON scalar, {__kind:eval, expr}, or {__kind:value_ref, key, type?}",
                ),
            ),
        ),
    ),
    method_entry(
        "editor.get",
        "editor:_h_editor_get",
        MethodSpec(
            5.0,
            "Read an editing session's settable cfg as a NESTED tree of current "
            "values (the read-only view; edit a leaf with editor.set_field using the "
            "leaf's dotted path). Node shape, distinguished by '$'-prefixed reserved "
            "keys: a SCALAR leaf is its bare current value (null = unset); an ENUM "
            "scalar leaf is {'$value': current, '$choices': [...]}; a SWEEP is a "
            "sub-tree of bare edges {start, stop, expts, step} (each edge accepts "
            "ONLY a number/int via editor.set_field — NOT an eval/ref); a REF node "
            "(module/waveform/device selector) is {'$ref': {'current': <chosen>, "
            "'options': [<names>]}, <chosen variant's settable sub-tree>} — only the "
            "CURRENTLY-CHOSEN variant is expanded; 'options' lists bare names while "
            "'current' may be a tagged internal key — switch by passing a bare "
            "'options' name to editor.set_field on the ref's dotted path. "
            "Any other dict is a plain section sub-tree (its keys are child fields). "
            "'prefix' (optional, dotted) returns just the sub-tree rooted at that "
            "node (a prefix at a sweep edge returns the whole sweep node); a prefix "
            "matching nothing returns {}.",
            (
                _str("editor_id"),
                _str_opt(
                    "prefix",
                    "Return only the sub-tree rooted at this dotted path "
                    "(e.g. 'modules.readout'); omit for the whole draft. No match → {}",
                ),
            ),
        ),
    ),
    method_entry(
        "editor.commit",
        "editor:_h_editor_commit",
        MethodSpec(
            10.0,
            "Save the editing session (from gui_editor_open) as a ModuleLibrary "
            "module/waveform: lower the session (eval expressions resolved against "
            "MetaDict to concrete numbers) and register it into the ModuleLibrary "
            "under 'name'. This is NOT 'apply a tab cfg edit' — tab cfg edits are "
            "already live (WYSIWYG); this persists the draft as a named ml entry. "
            "Returns {}. On success the session is destroyed; on validation failure "
            "it RAISES and the session is kept so you can fix and retry.",
            (
                _str("editor_id"),
                _str("name", "ml entry name to register under"),
                _expected_versions(),
            ),
            tool_name="gui_editor_save",
        ),
    ),
    method_entry(
        "editor.discard",
        "editor:_h_editor_discard",
        MethodSpec(
            5.0,
            "Discard an editing session (from gui_editor_open) without writing to the "
            "ModuleLibrary. Returns {}.",
            (_str("editor_id"),),
        ),
    ),
)
