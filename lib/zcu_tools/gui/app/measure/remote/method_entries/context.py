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

_EXPECTED_REVISION = ParamSpec(
    "expected_revision",
    JsonType.STRING,
    required=False,
    omit_when_missing=True,
    description=(
        "Installed base SHA-256. Writers may omit to retain unguarded admission; "
        "when present, compare with disk before writing. Null means missing aliases "
        "only. context.reload requires this key and compares the installed base, "
        "not the edited disk hash. Stale revisions return revision_mismatch."
    ),
)

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "context.reload",
        "context_reload:h_context_reload",
        MethodSpec(
            10.0,
            "Reload one edited file belonging to the currently selected point. "
            "path must be the absolute parameters, library or aliases owner path "
            "from entry.info.files. expected_revision must be explicitly supplied "
            "and match the installed base, not the edited disk hash. Null expects "
            "a missing aliases base. Read/validate/install/publish once without "
            "writing or repairing disk. Return {document, path, revision}. "
            "Invalid path/contents return invalid_params with path and cause; "
            "stale base returns precondition_failed/revision_mismatch; non-holder "
            "returns not_holder through dispatch. Failures retain installed state. "
            "No hidden retry, selection change or refresh of fixed tab captures.",
            (
                required_string("path", "Absolute selected-point owner file path"),
                ParamSpec(
                    "expected_revision",
                    JsonType.STRING,
                    required=True,
                    nullable=True,
                    description="Required installed base SHA-256, or null for a missing aliases base",
                ),
            ),
        ),
    ),
    method_entry(
        "context.aliases_set",
        "context:h_context_aliases_set",
        MethodSpec(
            10.0,
            "Apply the entire ordered alias table through the Inspect Apply owner. "
            "rows is a list of exactly {alias: string, value: string | list[string]}; "
            "rows=[] clears the table. Values are concrete names, empty strings, "
            "backward $alias.field links, or ordered lists of nonempty names/links. "
            "An empty value list is unbound. "
            "Syntax/order errors reject the whole table. An opened entry saves "
            "the selected point aliases.yaml atomically before installing/publishing; "
            "without a selected point, fails with no_context. "
            "Storage failure preserves the old file/table. "
            "Returns {rows: [{alias, value}]} from the installed authored table, "
            "not proof of resolvability. For resolved rows read "
            "context.snapshot(path='aliases'); that read still requires a point. "
            "Requires holder; allowed during Run, whose fixed capture is unchanged. "
            "No dialog/focus, point, metadata or ledger writes; no hidden read/retry.",
            (
                required_json("rows", "Complete ordered list of {alias, value} rows"),
                ParamSpec(
                    "expected_revision",
                    JsonType.STRING,
                    required=False,
                    nullable=True,
                    omit_when_missing=True,
                    description=_EXPECTED_REVISION.description,
                ),
            ),
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
            "library edits, component creation or setup writes. Module bindings are "
            "point values: path='MG12.module.readout', value='MG12.readout.mux_ro' "
            "can fill a missing slot; writing the module mapping is also supported. "
            "Captures the current "
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
                _EXPECTED_REVISION,
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
            "Paths use generated Form wrappers, not YAML: pulse duration is "
            "['waveform', 'module', 'length']; waveform selection uses "
            "['waveform'] = {kind: 'inline', module: {style: 'const', length: 5}}. "
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
                _EXPECTED_REVISION,
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
            "int|float|str|bool and must match the registered source type. "
            "An unregistered key returns invalid_params with reason=unregistered_source "
            "and data.path naming the missing source.",
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
        "context.ml_list_roles",
        "context:h_context_ml_list_roles",
        MethodSpec(
            5.0,
            "List experiment-role templates for context.ml_create_from_role. Returns "
            "{roles: [{role_id, label, item_kind, default_name}]}. Each role seeds a "
            "blank module/waveform with md-linked defaults (e.g. 'res_probe', "
            "'bath_reset'); 'default_name' is the suggested entry name. Roles with "
            "required fields, such as readout/mux:blank's tones, need edits in "
            "context.ml_create_from_role before they can be created.",
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
            "(module/waveform) is derived from 'role_id'. Placeholder roles require "
            "resolved same-named context aliases: bind qubit in Aliases before using "
            "qub_probe/pi_pulse; name.component does not select the qubit. Missing "
            "aliases return invalid_params and name the required alias. One-shot: seeds the role's "
            "defaults from one fixed capture, applies optional edits, then validates "
            "and writes the Spec. Edits have the same path/value Form input shape as "
            "context.ml_edit; expressions use {expr: text}. Paths address Form input, "
            "not stored YAML: pulse waveform duration is ['waveform', 'module', 'length'], "
            "pulse wiring is ['ch'], readout wiring is ['gen_ch'] and ['ro_ch'] "
            "(mux ADCs are ['tones', index, 'ro_ch']). Supply required fields "
            "before creation: pulse ch/freq/gain/waveform duration; pulse readout "
            "gen_ch/ro_ch/freq/gain/waveform duration/ro_length; mux readout "
            "gen_ch/mixer_freq/length/ro_length/tones (each needs freq/gain/ro_ch). "
            "nqz accepts 1 or 2. Generator/ADC capability compatibility is checked "
            "at Run admission, not by the library form. Invalid paths, values or "
            "Form results return invalid_params with data.errors: [{path, message}] "
            "and leave the library unchanged. Echoes {created: name}. To change an "
            "existing entry use context.ml_edit(kind, name, edits).",
            (
                required_string("role_id", "role id from context.ml_list_roles"),
                required_string(
                    "name",
                    "New component.partition.name root; module partitions: pulses, "
                    "readout, reset; waveform partition: waveforms. Example: Q1.pulses.probe",
                ),
                ParamSpec(
                    "edits",
                    JsonType.JSON,
                    required=False,
                    description=(
                        "Optional initial Form edits: [{path: [model-field segments], "
                        "value: Form input JSON}]. Expressions use {expr: text}."
                    ),
                ),
                _EXPECTED_REVISION,
            ),
        ),
    ),
)
