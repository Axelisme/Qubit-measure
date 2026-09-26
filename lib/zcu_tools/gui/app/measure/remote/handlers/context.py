"""Context remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.value_lookup import ValueInfo

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ._wire_values import _json_safe


def _h_context_use(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    ctx = adapter.context_control
    # A context lives under a project; without one there are no labels to switch
    # to. Map that precondition to agent language rather than leaking a controller
    # error (mirror _h_context_new).
    if not ctx.has_project():
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No project applied yet; apply a project first (gui_project_apply).",
            reason="no_project",
        )
    label = str(params["label"])
    available = list(ctx.get_context_labels())
    if label not in available:
        # Fast-fail an unknown label with the valid choices so the agent can
        # correct without a separate gui_context_list round-trip.
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"unknown context label: {label!r}; available: {available}",
        )
    ctx.use_context(label)
    active = ctx.get_active_context_label()
    return {
        "label": active,
        "has_active_context": active is not None,
    }


def _h_context_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    ctx = adapter.context_control
    # A context lives under a project's experiment dir; without a project the
    # IOManager has no dir to create it in. Translate that precondition into
    # agent language here rather than leaking the internal "IOManager not set
    # up" RuntimeError as a controller_error.
    if not ctx.has_project():
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No project applied yet; apply a project first (gui_project_apply).",
            reason="no_project",
        )
    bind_device = params["bind_device"]
    clone_from = params["clone_from"]
    ctx.new_context(
        bind_device=str(bind_device) if bind_device is not None else None,
        clone_from=str(clone_from) if clone_from is not None else None,
    )
    # new_context makes the new context active — return its label so the agent
    # knows what was created without a follow-up read.
    label = ctx.get_active_context_label()
    return {"label": label, "has_active_context": label is not None}


def _h_context_labels(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"labels": list(adapter.context_control.get_context_labels())}


def _h_context_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"label": adapter.context_control.get_active_context_label()}


def _h_context_md_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    md = adapter.context_control.get_current_md()
    return {"keys": sorted(str(k) for k in md.keys())}


def _h_context_md_get_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    md = adapter.context_control.get_current_md()
    sentinel = object()
    value = md.get(key, sentinel)
    if value is sentinel:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown md key: {key!r}")
    return {"key": key, "value": _json_safe(value)}


def _value_info_to_wire(info: ValueInfo) -> dict[str, object]:
    return {
        "key": info.key,
        "type": info.type_name,
        "owner": info.owner,
        "description": info.description,
    }


def _h_value_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {
        "values": [
            _value_info_to_wire(info)
            for info in adapter.context_control.list_value_sources()
        ]
    }


def _h_value_read(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    raw_type = params.get("type")
    if raw_type is not None and not isinstance(raw_type, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'type' must be a string")
    type_name = cast(str | None, raw_type)
    try:
        info, value = adapter.context_control.read_value_source(key, type_name)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {**_value_info_to_wire(info), "value": value}


def _h_context_ml_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    ml = adapter.context_control.get_current_ml()
    # Each stored cfg is a pydantic discriminated-union value: modules tag on
    # 'type' (e.g. 'pulse', 'reset/bath'), waveforms on 'style' (e.g. 'gauss').
    # Surface the discriminator so the agent can tell entry kinds apart without
    # opening each one (gui_context_ml_inspect).
    return {
        "modules": [
            {"name": name, "kind": getattr(ml.modules[name], "type")}
            for name in sorted(ml.modules.keys())
        ],
        "waveforms": [
            {"name": name, "style": getattr(ml.waveforms[name], "style")}
            for name in sorted(ml.waveforms.keys())
        ],
    }


def _h_context_ml_list_roles(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """List the experiment-role templates available for create_from_role."""
    del params
    catalog = adapter.ctrl.get_role_catalog()
    return {"roles": list(catalog.list_meta())}


def _h_context_ml_create_from_role(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Create a blank ml module/waveform from a named role and register it.

    One-shot: seeds md-linked defaults (lowered against the live md), writes ml.
    Edit afterwards via editor.new(from_name=...).
    """
    role_id = str(params["role_id"])
    name = str(params["name"])
    # The item kind is a property of the role, not an independent agent input —
    # derive it from role_id so the agent cannot pass a mismatching pair. An
    # unknown role_id fails fast as invalid_params; a missing catalog (no project)
    # surfaces as precondition_failed (mirror _h_context_ml_list_roles).
    try:
        item_kind = adapter.ctrl.get_role_catalog().get(role_id).item_kind
    except KeyError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    try:
        adapter.ctrl.create_from_role(item_kind, role_id, name)
    except KeyError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"created": name}


def _h_context_md_set_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    value = params["value"]
    adapter.context_control.set_md_attr(key, value)
    return {}


def _h_context_md_del_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    adapter.context_control.del_md_attr(key)
    return {}


def _h_context_ml_del_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    adapter.context_control.del_ml_module(name)
    return {"deleted": name}


def _h_context_ml_rename_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    old = str(params["old"])
    new = str(params["new"])
    adapter.context_control.rename_ml_module(old, new)
    return {"renamed": new}


def _h_context_ml_rename_waveform(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    old = str(params["old"])
    new = str(params["new"])
    adapter.context_control.rename_ml_waveform(old, new)
    return {"renamed": new}


def _h_context_ml_del_waveform(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    adapter.context_control.del_ml_waveform(name)
    return {"deleted": name}
