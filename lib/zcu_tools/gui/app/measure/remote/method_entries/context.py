"""Context remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import (
    optional_string,
    required_json,
    required_string,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "context.aliases_set",
        "context:h_context_aliases_set",
        MethodSpec(
            10.0,
            "Apply the entire ordered alias table through the Inspect Apply owner. "
            "rows is a list of exactly {alias: string, value: string}; [] clears it. "
            "Values are concrete names, empty strings or backward $alias.field links. "
            "Syntax/order errors reject the whole table. An opened entry saves "
            "aliases.yaml atomically before installing/publishing; without an entry "
            "only memory changes. Storage failure preserves the old file/table. "
            "Returns {rows: [{alias, value}]} from the installed authored table, "
            "not proof of resolvability. For resolved rows read "
            "context.snapshot(path='aliases'); that read still requires a point. "
            "Requires holder; allowed during Run, whose fixed capture is unchanged. "
            "No dialog/focus, point, metadata or ledger writes; no hidden read/retry.",
            (required_json("rows", "Complete ordered list of {alias, value} rows"),),
        ),
    ),
    method_entry(
        "context.set_parameter",
        "context:h_context_set_parameter",
        MethodSpec(
            10.0,
            "Set a concrete parameter path at the currently selected point through "
            "the Inspect Set owner. Requires holder and a selected point; EMPTY "
            "fails with no_context before writes. value is a required non-null JSON "
            "YAML value; this method cannot write top-level null. No alias paths, "
            "library edits, component creation or setup writes. Captures the current "
            "target and before once without I/O, then submits once with metadata "
            "preserved. Returns {path, before, value}: before is null for both unset "
            "and explicit null; value is the submitted value, not readback. Read "
            "context.snapshot separately to inspect actual stored values. Owner "
            "schema/conflict/storage failures propagate; no retry or rollback. "
            "Run permits these direct writes; its fixed capture is unchanged. "
            "Later source captures see new values; existing tabs need Reset. "
            "No metadata/ledger, dialog/focus or selection changes.",
            (
                required_string("path", "Concrete point/general parameter path"),
                required_json(
                    "value", "Submitted YAML value; top-level null is not supported"
                ),
            ),
        ),
    ),
    method_entry(
        "context.ml_edit",
        "context:h_context_ml_edit",
        MethodSpec(
            10.0,
            "Edit one ModuleLibrary entry as one atomic Form batch, like tab.edit_cfg. "
            "Each edit has path: [model-field segments] and value: Form input JSON. "
            'Expressions use {"expr": text} and are evaluated once to numbers; '
            "child library keys are plain strings; the root type/style tag is fixed. "
            "Only a valid form is written; otherwise the library is unchanged. "
            "Returns {valid, errors: [{path, message}]}. Unknown paths and retired "
            "__kind value tags are rejected before any write. save_as writes a new "
            "entry, rejects an existing name and leaves 'name' unchanged.",
            (
                ParamSpec(
                    "kind",
                    JsonType.STRING,
                    description="'module' or 'waveform'",
                    enum=("module", "waveform"),
                ),
                required_string("name", "Existing library entry key"),
                required_json(
                    "edits", "Nonempty list of {path: [segments], value} inputs"
                ),
                optional_string("save_as", "New entry key to write instead of name"),
            ),
        ),
    ),
    method_entry(
        "context.labels",
        "context:h_context_labels",
        MethodSpec(
            5.0,
            "List context labels",
            readonly=True,
        ),
    ),
    method_entry(
        "context.active",
        "context:h_context_active",
        MethodSpec(
            5.0,
            "Active context label",
            readonly=True,
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "context.snapshot",
        "context:h_context_snapshot",
        MethodSpec(
            15.0,
            "Read the published parameter context: {target, point, library, aliases}. "
            "aliases contains ordered {alias, value, resolved} rows; failed rows have "
            "resolved=null and error. All come from one fixed capture without kind checks. "
            "target is the existing saved_point locator. "
            "point and library are complete working-unit native documents: point includes "
            "components and metadata; library contains Specs and provenance without asset "
            "materialization. "
            "Complex numbers use {__complex__: [real, imag]}. May return large or sensitive data; "
            "summaries=true preserves dict trees and scalars and describes array/list leaves. "
            "path selects a complete subtree, takes priority over summaries, and returns "
            "{target, aliases, path, value}. Dotted paths start with target/point/library/aliases; "
            "list indices must be canonical nonnegative decimals. Invalid paths name the first bad segment. "
            "Unsupported values fail the entire read in every mode.",
            (
                ParamSpec("summaries", JsonType.BOOLEAN, required=False, default=False),
                optional_string("path"),
            ),
            readonly=True,
        ),
    ),
    method_entry(
        "value.list",
        "context:h_value_list",
        MethodSpec(
            5.0,
            "List registered read-only value sources. Returns "
            "{values: [{key, type, owner, description}]}. These are escape-hatch "
            "resolve-once sources for rare defaults and agent reads; prefer typed "
            "RPCs when a stable API exists.",
            readonly=True,
        ),
    ),
    method_entry(
        "value.read",
        "context:h_value_read",
        MethodSpec(
            5.0,
            "Resolve one registered value source immediately. Returns "
            "{key, type, owner, description, value}. Optional 'type' is one of "
            "int|float|str|bool and must match the registered source type.",
            (
                required_string(
                    "key", "Registered value source key, e.g. device.flux.value"
                ),
                optional_string(
                    "type", "Optional expected type: int, float, str, or bool"
                ),
            ),
            readonly=True,
        ),
    ),
    method_entry(
        "context.ml_del_module",
        "context:h_context_ml_del_module",
        MethodSpec(
            5.0,
            "Delete a point-local module at a concrete component.partition.name key "
            "(partition is pulses/readout/reset); flat names and aliases are not accepted. "
            "Echoes {deleted: name}. LINKED cfg refs "
            "keep the missing key and become invalid until it returns or is edited; "
            "MODIFIED refs retain their inline Custom value.",
            (required_string("name", "Concrete module key, e.g. Q1.pulses.pi"),),
        ),
    ),
    method_entry(
        "context.ml_del_waveform",
        "context:h_context_ml_del_waveform",
        MethodSpec(
            5.0,
            "Delete a point-local waveform at a concrete component.waveforms.name key; "
            "flat names and aliases are not accepted. Echoes {deleted: name}. LINKED cfg refs "
            "keep the missing key and become invalid until it returns or is edited; "
            "MODIFIED refs retain their inline Custom value.",
            (
                required_string(
                    "name", "Concrete waveform key, e.g. Q1.waveforms.drive"
                ),
            ),
        ),
    ),
    method_entry(
        "context.ml_rename_module",
        "context:h_context_ml_rename_module",
        MethodSpec(
            5.0,
            "Rename a point-local module using concrete component.partition.name keys "
            "for old/new (partition is pulses/readout/reset); no flat names or aliases. "
            "Clash fails fast. Echoes "
            "{renamed: new}. LINKED cfg refs keep the missing 'old' key and become "
            "invalid until it returns or is edited; MODIFIED refs retain inline Custom.",
            (
                required_string(
                    "old", "Current concrete module key, e.g. Q1.pulses.pi"
                ),
                required_string(
                    "new", "New concrete module key, e.g. Q1.pulses.pi_half"
                ),
            ),
        ),
    ),
    method_entry(
        "context.ml_rename_waveform",
        "context:h_context_ml_rename_waveform",
        MethodSpec(
            5.0,
            "Rename a point-local waveform using concrete component.waveforms.name keys "
            "for old/new; no flat names or aliases. Clash fails fast. Echoes "
            "{renamed: new}. LINKED cfg refs keep the missing 'old' key and become "
            "invalid until it returns or is edited; MODIFIED refs retain inline Custom.",
            (
                required_string(
                    "old", "Current concrete waveform key, e.g. Q1.waveforms.drive"
                ),
                required_string(
                    "new", "New concrete waveform key, e.g. Q1.waveforms.drive_half"
                ),
            ),
        ),
    ),
    method_entry(
        "context.ml_list_roles",
        "context:h_context_ml_list_roles",
        MethodSpec(
            5.0,
            "List experiment-role templates for context.ml_create_from_role. Returns "
            "{roles: [{role_id, label, item_kind, default_name}]}. Each role seeds a "
            "blank module/waveform with md-linked defaults (e.g. 'res_probe', "
            "'bath_reset'); 'default_name' is the suggested entry name.",
            readonly=True,
        ),
    ),
    method_entry(
        "context.ml_create_from_role",
        "context:h_context_ml_create_from_role",
        MethodSpec(
            10.0,
            "Create a blank ModuleLibrary module/waveform from a named role "
            "(from context.ml_list_roles) and register it under 'name'. The item kind "
            "(module/waveform) is derived from 'role_id'. One-shot: seeds the role's "
            "md-linked defaults (lowered to the md's current values) — it does NOT open "
            "an editing session. Echoes {created: name}. To then change the entry use "
            "context.ml_edit(kind, name, edits).",
            (
                required_string("role_id", "role id from context.ml_list_roles"),
                required_string("name", "new ml entry name"),
            ),
        ),
    ),
)
