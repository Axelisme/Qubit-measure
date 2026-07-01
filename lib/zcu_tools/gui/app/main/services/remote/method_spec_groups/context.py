"""Context remote method specs."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _json,
    _str,
    _str_opt,
)

SPECS: dict[str, MethodSpec] = {
    "context.use": MethodSpec(
        5.0,
        "Switch the active context to 'label'. Echoes {label, has_active_context}. "
        "An unknown label fails fast (invalid_params) with the available labels; no "
        "applied project fails with precondition_failed.",
        (_str("label", "Context label to switch to"),),
        tool_name="gui_context_switch",
    ),
    "context.new": MethodSpec(
        10.0,
        "Create a new context and make it active. Echoes {label, has_active_context} "
        "— the auto-derived label (the agent cannot name it directly).",
        (
            _str_opt(
                "bind_device",
                "Connected flux device to bind: its current value/unit name the "
                "context (whitelist: FakeDevice->none, YOKOGS200->A). Omit for an "
                "unbound context (unit=none, no value).",
            ),
            _str_opt("clone_from", "Label of an existing context to clone ml/md from"),
        ),
        tool_name="gui_context_create",
    ),
    "context.labels": MethodSpec(5.0, "List context labels"),
    "context.active": MethodSpec(5.0, "Active context label"),
    "context.md_get": MethodSpec(5.0, "List MetaDict keys"),
    "context.md_get_attr": MethodSpec(
        5.0, "Read one MetaDict attribute", (_str("key", "MetaDict key"),)
    ),
    "value.list": MethodSpec(
        5.0,
        "List registered read-only value sources. Returns "
        "{values: [{key, type, owner, description}]}. These are escape-hatch "
        "resolve-once sources for rare defaults and agent reads; prefer typed "
        "RPCs when a stable API exists.",
    ),
    "value.read": MethodSpec(
        5.0,
        "Resolve one registered value source immediately. Returns "
        "{key, type, owner, description, value}. Optional 'type' is one of "
        "int|float|str|bool and must match the registered source type.",
        (
            _str("key", "Registered value source key, e.g. device.flux.value"),
            _str_opt("type", "Optional expected type: int, float, str, or bool"),
        ),
    ),
    "context.ml_get": MethodSpec(
        5.0,
        "List ModuleLibrary entries with their discriminator: returns "
        "{modules: [{name, kind}], waveforms: [{name, style}]}, sorted by name. "
        "'kind' is the module type tag (e.g. 'pulse', 'reset/bath'); 'style' is the "
        "waveform style (e.g. 'gauss', 'const'). Read one entry's full cfg with "
        "gui_context_ml_inspect.",
        tool_name="gui_context_ml_list",
    ),
    "context.md_set_attr": MethodSpec(
        5.0,
        "Set one MetaDict attribute",
        (_str("key", "MetaDict key"), _json("value", "JSON-safe value")),
    ),
    "context.md_del_attr": MethodSpec(
        5.0, "Delete one MetaDict attribute", (_str("key", "MetaDict key"),)
    ),
    "context.ml_del_module": MethodSpec(
        5.0,
        "Delete one ModuleLibrary module. Echoes {deleted: name}. cfg refs pointing "
        "at this entry degrade to inline Custom (the value is kept inline, not lost); "
        "to re-link, edit them.",
        (_str("name", "Module name"),),
        tool_name="gui_context_ml_delete_module",
    ),
    "context.ml_del_waveform": MethodSpec(
        5.0,
        "Delete one ModuleLibrary waveform. Echoes {deleted: name}. cfg refs pointing "
        "at this entry degrade to inline Custom (the value is kept inline, not lost); "
        "to re-link, edit them.",
        (_str("name", "Waveform name"),),
        tool_name="gui_context_ml_delete_waveform",
    ),
    "context.ml_rename_module": MethodSpec(
        5.0,
        "Rename a ModuleLibrary module old→new (clash fails fast). Echoes "
        "{renamed: new}. cfg refs to 'old' degrade to inline Custom (the value is "
        "kept inline, not lost); to re-link, edit them.",
        (_str("old", "Current module name"), _str("new", "New module name")),
    ),
    "context.ml_rename_waveform": MethodSpec(
        5.0,
        "Rename a ModuleLibrary waveform old→new (clash fails fast). Echoes "
        "{renamed: new}. cfg refs to 'old' degrade to inline Custom (the value is "
        "kept inline, not lost); to re-link, edit them.",
        (_str("old", "Current waveform name"), _str("new", "New waveform name")),
    ),
    "context.ml_list_roles": MethodSpec(
        5.0,
        "List experiment-role templates for gui_context_ml_create_from_role. Returns "
        "{roles: [{role_id, label, item_kind, default_name}]}. Each role seeds a "
        "blank module/waveform with md-linked defaults (e.g. 'res_probe', "
        "'bath_reset'); 'default_name' is the suggested entry name.",
    ),
    "context.ml_create_from_role": MethodSpec(
        10.0,
        "Create a blank ModuleLibrary module/waveform from a named role "
        "(gui_context_ml_list_roles) and register it under 'name'. The item kind "
        "(module/waveform) is derived from 'role_id'. One-shot: seeds the role's "
        "md-linked defaults (lowered to the md's current values) — it does NOT open "
        "an editing session. Echoes {created: name}. To then change the entry use "
        "gui_editor_open(from_name=name).",
        (
            _str("role_id", "role id from gui_context_ml_list_roles"),
            _str("name", "new ml entry name"),
        ),
    ),
}
