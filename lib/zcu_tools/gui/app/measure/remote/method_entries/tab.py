"""Tab remote method entries."""

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
        "tab.run_cfg_command",
        "tab:h_tab_run_cfg_command",
        MethodSpec(
            5.0,
            "Run a named command exposed by the current active Form description. "
            "path is a model-field string array. Same holder, busy and lifetime "
            "guards as edits. Conversion failures leave input unchanged; success "
            "returns input/values/errors. No arbitrary method execution.",
            (
                required_string("tab_id"),
                ParamSpec("path", JsonType.ARRAY),
                required_string("name"),
            ),
        ),
    ),
    method_entry(
        "tab.read_recipe_field",
        "tab:h_tab_read_recipe_field",
        MethodSpec(
            5.0,
            "Read one logical field from current input and fixed captured Specs. "
            "Generated wrappers are hidden; real field names remain. Paths keep str names "
            "and int original list indexes, never element names. path=[] reads "
            "root. Returns metadata, authored input, ordinary value/value_status, "
            "selector/source, readonly and directed logical errors in one turn. "
            "Ref reads do not materialize input, refresh sources or grant writes.",
            (required_string("tab_id"), ParamSpec("path", JsonType.ARRAY)),
            readonly=True,
        ),
    ),
    method_entry(
        "tab.edit_recipe_fields",
        "tab:h_tab_edit_recipe_fields",
        MethodSpec(
            5.0,
            "Atomically edit ordered typed logical path/value leaves (str names/int indexes). Generated selectors "
            "require select_recipe_module; no content write crosses ref ancestors. "
            "Structural/source/guard rejection commits no prefix. Returns changed "
            "and complete logical Form preview errors; invalid ordinary drafts "
            "remain installed. Empty batches still require mutation authority.",
            (required_string("tab_id"), ParamSpec("edits", JsonType.ARRAY)),
        ),
    ),
    method_entry(
        "tab.select_recipe_module",
        "tab:h_tab_select_recipe_module",
        MethodSpec(
            5.0,
            "Select/copy/disable a typed logical generated module selector (str names/int indexes). use_library "
            "requires an actual captured key allowed by the owner. Other actions "
            "must omit key; explicit null is invalid. use_inline explicitly copies "
            "fixed captured Spec content, preserving child keys; inline is a guarded "
            "no-op. Ref ancestors require explicit outer copies first. Returns "
            "changed and a fresh same-turn field observation. No retry or rollback.",
            (
                required_string("tab_id"),
                ParamSpec("path", JsonType.ARRAY),
                ParamSpec(
                    "action",
                    JsonType.STRING,
                    enum=("use_library", "use_inline", "disable"),
                ),
                ParamSpec(
                    "key", JsonType.STRING, required=False, omit_when_missing=True
                ),
            ),
        ),
    ),
    method_entry(
        "tab.new",
        "tab:h_tab_new",
        MethodSpec(
            10.0,
            "Create a new tab and capture current parameter/library/global aliases "
            "once. Unbound aliases may leave Form errors. Reads, edits and Run "
            "reuse this capture; Reset captures new sources. Returns {tab_id}.",
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
            "tab_id, cfg_issue (null or the typed-cfg unavailability reason) "
            "and analysis_error. Form input is unchanged. Failed analysis preparation "
            "retains the loaded result and the new tab.",
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
            "Read requested paths and module_refs from the tab's fixed Open/Reset "
            "parameter capture. Reset with aliases before querying overrides. "
            "Returns {values, modules}: absent values are "
            "omitted, null is retained; each module ref maps to an existing "
            "canonical library root or null for missing slots/roots. Invalid key syntax, "
            "invalid paths, unknown roles and source failures propagate.",
            (
                required_string("tab_id"),
                ParamSpec("paths", JsonType.ARRAY, required=True),
                ParamSpec("module_refs", JsonType.ARRAY, required=True),
            ),
            readonly=True,
        ),
    ),
    method_entry(
        "tab.get_cfg",
        "tab:h_tab_get_cfg",
        MethodSpec(
            5.0,
            "Read Form input, ordinary leaf values and errors from the "
            "tab's fixed source snapshot. Reads do not refresh sources. "
            "Expressions use {expr: text}; library keys are ordinary strings. "
            "Use Reset to capture new sources. Independent library editors retain "
            "their own interface.",
            (required_string("tab_id"),),
            readonly=True,
        ),
    ),
    method_entry(
        "tab.describe_cfg",
        "tab:h_tab_describe_cfg",
        MethodSpec(
            5.0,
            "Read one current single-layer description and local input_present, "
            "input, ordinary value and directed errors in one owner turn. "
            "path is a model-field string array, [] for root. Uses fixed "
            "Open/Reset sources without capture, defaults or materialization. "
            "Unknown/inactive/whole-leaf descendants fail with invalid_params, "
            "reason=form_path and data.form_path. Other author failures retain "
            "the controller_error boundary. Errors are Form preview root/descendant "
            "only; use get_cfg or Run admission for lowered cfg errors. Busy and "
            "observer reads are allowed; no widget is required.",
            (
                required_string("tab_id"),
                required_json("path", "Model-field string array; [] for root"),
            ),
            readonly=True,
        ),
    ),
    method_entry(
        "tab.refresh_sources",
        "tab:h_tab_refresh_sources",
        MethodSpec(
            5.0,
            "Capture current sources once, optionally overriding named aliases "
            "for this Refresh only. Preserve all input, expression drafts and "
            "explicit selections; return revalidated input, values and errors. "
            "Overrides do not change global aliases. Busy, revoked and "
            "notification-reentrant mutations are rejected. Capture/factory "
            "failure preserves the previous form. Does not reset defaults or results.",
            (
                required_string("tab_id"),
                ParamSpec(
                    "aliases",
                    JsonType.OBJECT,
                    required=False,
                    description="Single Refresh alias-name to component-name capture overrides",
                ),
            ),
        ),
    ),
    method_entry(
        "tab.reset_cfg",
        "tab:h_tab_reset_cfg",
        MethodSpec(
            5.0,
            "Capture current sources once, optionally overriding named aliases "
            "for this Reset only, and restore fresh adapter Form defaults. "
            "Returns input, values and errors. Overrides are not persisted "
            "or written to global aliases. Busy and notification-reentrant "
            "resets are rejected. Does not clear results or save artifacts.",
            (
                required_string("tab_id"),
                ParamSpec(
                    "aliases",
                    JsonType.OBJECT,
                    required=False,
                    description="Single Reset alias-name to component-name capture overrides",
                ),
            ),
        ),
    ),
    method_entry(
        "tab.edit_cfg",
        "tab:h_tab_edit_cfg",
        MethodSpec(
            5.0,
            "Atomically write unique model-field leaf paths against fixed sources. "
            "Each edit has path: [segments] and value: Form input JSON. "
            "Expressions use {expr: text}, complex literals use __complex__, "
            "and library keys are plain strings. Edit range leaves separately. "
            "Invalid values stay editable with errors; structural failures preserve "
            "the entire input. Returns input, values and errors.",
            (
                required_string("tab_id"),
                required_json(
                    "edits", "Ordered list of {path: [segments], value} inputs"
                ),
            ),
        ),
    ),
    method_entry(
        "tab.read_elements",
        "tab:h_tab_read_elements",
        MethodSpec(
            5.0,
            "Read every model-list element in one owner turn: original index, name, "
            "authored input, available canonical preview leaves and directed errors. "
            "Invalid elements never hide siblings. Independent of participants/picker; "
            "no converter, source refresh, cfg lowering or UI focus. Readonly and busy-safe.",
            (
                required_string("tab_id"),
                required_json("path", "Nonempty typed model-list path"),
                ParamSpec(
                    "fields",
                    JsonType.JSON,
                    required=False,
                    default=[],
                    description="Optional nonempty relative logical paths; each element returns fields in this order from the same owner turn, including FixedElements identity",
                ),
            ),
            readonly=True,
        ),
    ),
    method_entry(
        "tab.read_module",
        "tab:h_tab_read_module",
        MethodSpec(
            5.0,
            "Read {value, mode, source} for one typed core module path from fixed "
            "Open/Reset sources. Key mode reads the captured library; inline mode "
            "evaluates only this module. Other invalid fields do not block reads. "
            "No native cfg or extras lowering, mutation or source refresh occurs. "
            "This module's invalid values are rejected. Busy reads are allowed.",
            (
                required_string("tab_id"),
                ParamSpec(
                    "field",
                    JsonType.ARRAY,
                    description="Nonempty str field names and int model-list indexes",
                ),
            ),
            readonly=True,
        ),
    ),
    method_entry(
        "tab.use_inline",
        "tab:h_tab_use_inline",
        MethodSpec(
            5.0,
            "Convert one typed core module path to inline input through guarded "
            "option command. Copy a key from the fixed captured library. Preserve "
            "locks. Existing inline is a guarded no-op. Disabled or unresolved "
            "modules fail; select an Inline option to use template/type defaults. "
            "Busy, revoked and reentrant calls are rejected. One changed input "
            "publishes once; failures have no successful prefix. "
            "Returns the complete Form observation.",
            (
                required_string("tab_id"),
                ParamSpec(
                    "field",
                    JsonType.ARRAY,
                    description="Nonempty str field names and int model-list indexes",
                ),
            ),
        ),
    ),
)
