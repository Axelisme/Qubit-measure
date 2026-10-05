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
        "context.ml_edit",
        "context:h_context_ml_edit",
        MethodSpec(
            10.0,
            "Apply nonempty ordered library edits through the shared draft model. "
            "Commit each successful edit, stop at the first failure without rollback. "
            "save_as creates a new entry after the first successful edit; source stays unchanged. "
            "Returns valid/applied/errors. Read context.snapshot explicitly first.",
            (
                required_string("kind"),
                required_string("name"),
                required_json("edits"),
                optional_string("save_as"),
            ),
        ),
        agent=AgentMethodPolicy(
            guard_deps=("context",),
            refresh_after_write=True,
        ),
    ),
    method_entry(
        "context.use",
        "context:h_context_use",
        MethodSpec(
            5.0,
            "Switch the active context to 'label'. Echoes {label, has_active_context}. "
            "An unknown label fails fast (invalid_params) with the available labels; no "
            "applied project fails with precondition_failed.",
            (required_string("label", "Context label to switch to"),),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "context.new",
        "context:h_context_new",
        MethodSpec(
            10.0,
            "Create a new context and make it active. Optional label names it; "
            "otherwise the GUI derives a label from bind_device value/unit. "
            "clone_from='current' copies the active context (or starts empty when "
            "none is active); null starts empty. An unknown clone source fails "
            "without changing active/labels. Echoes {label, has_active_context}.",
            (
                optional_string("label", "Optional explicit context label"),
                optional_string(
                    "bind_device",
                    "Connected flux device to bind: its current value/unit name the "
                    "context (whitelist: FakeDevice->none, YOKOGS200->A). Omit for an "
                    "unbound context (unit=none, no value).",
                ),
                optional_string(
                    "clone_from", "Label of an existing context to clone ml/md from"
                ),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "context.labels",
        "context:h_context_labels",
        MethodSpec(
            5.0,
            "List context labels",
        ),
    ),
    method_entry(
        "context.active",
        "context:h_context_active",
        MethodSpec(
            5.0,
            "Active context label",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "context.snapshot",
        "context:h_context_snapshot",
        MethodSpec(
            15.0,
            "Explicit full read of the active context: {label, md, ml: {modules, "
            "waveforms}}. md contains every value and ml contains every entry's "
            "complete cfg, not just their names. May return large or sensitive data; "
            "use rpc_call only when you need to re-snapshot the whole context "
            "before a guarded mutation. Unsupported values fail the entire read.",
        ),
        agent=AgentMethodPolicy(reveals=("context",)),
    ),
    method_entry(
        "context.md_get",
        "context:h_context_md_get",
        MethodSpec(
            5.0,
            "List MetaDict keys; summaries=true also returns {values} with scalars "
            "and descriptions, never full non-scalar contents.",
            (
                ParamSpec(
                    "summaries",
                    JsonType.BOOLEAN,
                    required=False,
                    default=False,
                    description="Include compact value summaries",
                ),
            ),
        ),
    ),
    method_entry(
        "context.md_get_attr",
        "context:h_context_md_get_attr",
        MethodSpec(
            5.0,
            "Read one MetaDict attribute",
            (required_string("key", "MetaDict key"),),
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
        ),
    ),
    method_entry(
        "context.ml_get",
        "context:h_context_ml_get",
        MethodSpec(
            5.0,
            "List ModuleLibrary modules/waveforms as {modules, waveforms}, each "
            "entry carrying name, discriminator kind/style and description. "
            "With name, return {name, kind: 'module'|'waveform', cfg} for the "
            "named stored cfg without opening an editor. Require kind if the "
            "same name exists in both collections; unknown names fail with "
            "available options.",
            (
                optional_string("name", "Entry to read; omit for the index"),
                optional_string("kind", "module or waveform when names collide"),
            ),
        ),
    ),
    method_entry(
        "context.md_set_attr",
        "context:h_context_md_set_attr",
        MethodSpec(
            5.0,
            "Set one MetaDict attribute; receipt=true returns its actual "
            "{before, after} from the owner turn.",
            (
                required_string("key", "MetaDict key"),
                required_json("value", "JSON-safe value"),
                ParamSpec(
                    "receipt",
                    JsonType.BOOLEAN,
                    required=False,
                    default=False,
                    description="Return actual before/after values",
                ),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "context.md_del_attr",
        "context:h_context_md_del_attr",
        MethodSpec(
            5.0,
            "Delete one MetaDict attribute",
            (required_string("key", "MetaDict key"),),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "context.ml_del_module",
        "context:h_context_ml_del_module",
        MethodSpec(
            5.0,
            "Delete one ModuleLibrary module. Echoes {deleted: name}. LINKED cfg refs "
            "keep the missing key and become invalid until it returns or is edited; "
            "MODIFIED refs retain their inline Custom value.",
            (required_string("name", "Module name"),),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "context.ml_del_waveform",
        "context:h_context_ml_del_waveform",
        MethodSpec(
            5.0,
            "Delete one ModuleLibrary waveform. Echoes {deleted: name}. LINKED cfg refs "
            "keep the missing key and become invalid until it returns or is edited; "
            "MODIFIED refs retain their inline Custom value.",
            (required_string("name", "Waveform name"),),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "context.ml_rename_module",
        "context:h_context_ml_rename_module",
        MethodSpec(
            5.0,
            "Rename a ModuleLibrary module old→new (clash fails fast). Echoes "
            "{renamed: new}. LINKED cfg refs keep the missing 'old' key and become "
            "invalid until it returns or is edited; MODIFIED refs retain inline Custom.",
            (
                required_string("old", "Current module name"),
                required_string("new", "New module name"),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "context.ml_rename_waveform",
        "context:h_context_ml_rename_waveform",
        MethodSpec(
            5.0,
            "Rename a ModuleLibrary waveform old→new (clash fails fast). Echoes "
            "{renamed: new}. LINKED cfg refs keep the missing 'old' key and become "
            "invalid until it returns or is edited; MODIFIED refs retain inline Custom.",
            (
                required_string("old", "Current waveform name"),
                required_string("new", "New waveform name"),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "context.ml_list_roles",
        "context:h_context_ml_list_templates",
        MethodSpec(
            5.0,
            "List experiment-role templates for context.ml_create_from_role. Returns "
            "{roles: [{role_id, label, item_kind, default_name}]}. Each role seeds a "
            "blank module/waveform with md-linked defaults (e.g. 'res_probe', "
            "'bath_reset'); 'default_name' is the suggested entry name.",
        ),
    ),
    method_entry(
        "context.ml_create_from_role",
        "context:h_context_ml_create_from_template",
        MethodSpec(
            10.0,
            "Create a blank ModuleLibrary module/waveform from a named role "
            "(from context.ml_list_roles) and register it under 'name'. The item kind "
            "(module/waveform) is derived from 'role_id'. One-shot: seeds the role's "
            "md-linked defaults (lowered to the md's current values) — it does NOT open "
            "an editing session. Echoes {created: name}. To then change the entry use "
            "rpc_call on editor.new(item_kind, from_name=name).",
            (
                required_string("role_id", "role id from context.ml_list_roles"),
                required_string("name", "new ml entry name"),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
)
