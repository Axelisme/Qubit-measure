"""Writeback remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import (
    optional_integer,
    optional_string,
    required_string,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "tab.writeback_write",
        "writeback:h_tab_writeback_write",
        MethodSpec(
            10.0,
            "Edit write=[{id,target?,value?,edits?}] sequentially in the shared draft. "
            "edits uses the tab.writeback_set module grammar. An edit failure, "
            "including a listed module whose form is invalid, preserves the draft "
            "prefix and starts no context write; an invalid module form fails with "
            "invalid_params, reason form_errors and data.errors [{path, message}]. "
            "After all edits succeed, apply only the listed IDs once without changing "
            "GUI selection. Returns {written:[{id,kind,target,before,after}]} with "
            "complete actual destination values. Kinds are parameter|module|sample; "
            "cross-kind names may coincide. Empty write is a no-op. Explicit null "
            "value sets an md value to null; omission keeps its proposal. "
            "Requires a selected working point; writes retain the captured SavedPoint destination. "
            "Context apply retains existing failure semantics, not cross-file atomicity.",
            (
                required_string("tab_id"),
                required_string("subtab_id", "Pane: analysis|post_analysis"),
                ParamSpec("write", JsonType.JSON, required=True),
            ),
        ),
    ),
    method_entry(
        "tab.writeback_preview",
        "writeback:h_tab_writeback_preview",
        MethodSpec(
            5.0,
            "List a pane's persistent writeback draft (pure read — not a dry-run; the "
            "draft was computed once at analyze time). Requires (tab_id, subtab_id) "
            "with closed values analysis|post_analysis. Returns "
            "{has_draft, items, destination_context}; has_draft is false before any "
            "analyze produced a draft; destination_context is the current active "
            "SessionEnv projection at reply time. Each item: id "
            "(<kind>-<n>, kind∈param|ml|sample), target_name (apply destination, "
            "editable), kind (parameter|module|sample), description, selected; "
            "parameter adds proposed_value; every item includes complete "
            "current/proposed values (parameter values, or module Spec dictionaries "
            "in the same shape for current and proposed). Current is read from "
            "the active context at preview time and is null for a missing target; "
            "proposed module Specs come from the shared GUI form evaluated at the "
            "draft's capture. An invalid module form (invalid_params, reason "
            "form_errors, data.errors) or missing context fails instead of reporting "
            "an empty preview. Module items may include role_id when the proposal "
            "corresponds to a ModuleLibrary role. A complex parameter proposed_value "
            "is carried as "
            '{"__complex__": [re, im]} (JSON has no complex). Edit an item via '
            "rpc_call on tab.writeback_set; the user's Edit dialog renders the same "
            "model (WYSIWYG). Optional operation_id requires that analysis result "
            "before reading the draft; replaced results are rejected.",
            (
                required_string("tab_id"),
                required_string("subtab_id", "Pane: analysis|post_analysis"),
                optional_integer("operation_id"),
            ),
            readonly=True,
        ),
    ),
    method_entry(
        "tab.writeback_set",
        "writeback:h_tab_writeback_set",
        MethodSpec(
            5.0,
            "Edit a pane's persistent writeback item by id — the single writeback "
            "editing surface. Requires (tab_id, subtab_id) with closed values "
            "analysis|post_analysis. selected? / target_name? apply to any item. "
            "proposed_value? is "
            "the METADICT-only facet (a complex value is passed as "
            '{"__complex__": [re, im]}, the same shape the list emits; it applies as '
            "a Python complex). edits? is the MODULE-only facet: one batch "
            "[{path:[field,...], value}] written into the item's form with the "
            "tab.edit_cfg grammar (path segments are model field names; text fields "
            'accept {"expr": text} evaluated at the draft capture). The batch '
            "installs whole: unknown paths fail with invalid_params and change "
            "nothing, while invalid values stay in the form and appear in errors. "
            "proposed_value and edits are mutually exclusive (different item kinds). "
            "Echoes the edited {item}; an edits batch also returns the form "
            "observation {input, values, errors}. Read current/proposed "
            "values via tab.writeback_preview.",
            (
                required_string("tab_id"),
                required_string("subtab_id", "Pane: analysis|post_analysis"),
                required_string("id", "writeback item session id (<kind>-<n>)"),
                # Boolean (not JSON): a JSON schema of {type: boolean} makes the
                # client send a real boolean. Declared as JSON, the client may send
                # the string "false", which ``bool("false")`` wrongly reads as True.
                ParamSpec("selected", JsonType.BOOLEAN, required=False),
                optional_string("target_name", "new apply destination name"),
                ParamSpec(
                    "proposed_value",
                    JsonType.JSON,
                    required=False,
                    description="Proposed parameter scalar (parameter items only)",
                ),
                ParamSpec(
                    "edits",
                    JsonType.JSON,
                    required=False,
                    description="Form edits [{path:[field,...], value}] "
                    "(module items only)",
                ),
            ),
        ),
    ),
    method_entry(
        "tab.writeback_apply",
        "writeback:h_tab_writeback_apply",
        MethodSpec(
            10.0,
            "Apply a pane's persistent writeback draft as-is (edit it first via "
            "rpc_call on tab.writeback_set). Requires (tab_id, subtab_id) with closed "
            "values analysis|post_analysis. Optional ids applies only those items without "
            "changing GUI selection; omitted or null ids uses selected items. Empty ids writes "
            "nothing; unknown or duplicate IDs fail before writing. Returns "
            "{applied_ids, written, context_version, destination_context}: written lists the destination "
            "names actually pushed, split by kind ({parameter, ml_modules, samples}); "
            "an invalid module form writes nothing and fails with invalid_params, "
            "reason form_errors and data.errors; "
            "context_version is the bumped 'context' resource version after apply; "
            "destination_context is the active SessionEnv projection at reply time. "
            "Requires a selected working point but always writes to the draft's captured SavedPoint.",
            (
                required_string("tab_id"),
                required_string("subtab_id", "Pane: analysis|post_analysis"),
                ParamSpec(
                    "ids",
                    JsonType.JSON,
                    required=False,
                    description="Explicit writeback item IDs; omitted uses GUI selection",
                ),
            ),
        ),
    ),
)
