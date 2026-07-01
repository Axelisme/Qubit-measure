"""Context remote handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.services.context import MlEntryValidationError
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ProviderError,
    UnavailableValue,
    ValueInfo,
    ValueTypeError,
)

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ._common import Handler
from ._wire_values import _json_safe

logger = logging.getLogger(__name__)


def _h_context_use(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # A context lives under a project; without one there are no labels to switch
    # to. Map that precondition to agent language rather than leaking a controller
    # error (mirror _h_context_new).
    if not adapter.ctrl.has_project():
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No project applied yet; apply a project first (gui_project_apply).",
            reason="no_project",
        )
    label = str(params["label"])
    available = list(adapter.ctrl.get_context_labels())
    if label not in available:
        # Fast-fail an unknown label with the valid choices so the agent can
        # correct without a separate gui_context_list round-trip.
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"unknown context label: {label!r}; available: {available}",
        )
    adapter.ctrl.use_context(label)
    return {
        "label": adapter.ctrl.get_active_context_label(),
        "has_active_context": adapter.ctrl.get_active_context_label() is not None,
    }


def _h_context_new(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # A context lives under a project's experiment dir; without a project the
    # IOManager has no dir to create it in. Translate that precondition into
    # agent language here rather than leaking the internal "IOManager not set
    # up" RuntimeError as a controller_error.
    if not adapter.ctrl.has_project():
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No project applied yet; apply a project first (gui_project_apply).",
            reason="no_project",
        )
    bind_device = params["bind_device"]
    clone_from = params["clone_from"]
    adapter.ctrl.new_context(
        bind_device=str(bind_device) if bind_device is not None else None,
        clone_from=str(clone_from) if clone_from is not None else None,
    )
    # new_context makes the new context active — return its label so the agent
    # knows what was created without a follow-up read.
    label = adapter.ctrl.get_active_context_label()
    return {"label": label, "has_active_context": label is not None}


def _h_context_labels(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"labels": list(adapter.ctrl.get_context_labels())}


def _h_context_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"label": adapter.ctrl.get_active_context_label()}


def _h_context_md_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    md = adapter.ctrl.get_current_md()
    return {"keys": sorted(str(k) for k in md.keys())}


def _h_context_md_get_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    md = adapter.ctrl.get_current_md()
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
            _value_info_to_wire(info) for info in adapter.ctrl.list_value_sources()
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
        info, value = adapter.ctrl.read_value_source(key, type_name)
    except (MissingValue, ValueTypeError, ValueError) as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except UnavailableValue as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    except ProviderError as exc:
        raise RemoteError(ErrorCode.CONTROLLER_ERROR, str(exc)) from exc
    return {**_value_info_to_wire(info), "value": value}


def _h_context_ml_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    ml = adapter.ctrl.get_current_ml()
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
    try:
        catalog = adapter.ctrl.get_role_catalog()
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
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
    except RuntimeError as exc:
        raise RemoteError(ErrorCode.PRECONDITION_FAILED, str(exc)) from exc
    try:
        adapter.ctrl.create_from_role(item_kind, role_id, name)
    except KeyError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except MlEntryValidationError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"created": name}


def _h_context_md_set_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    value = params["value"]
    try:
        adapter.ctrl.set_md_attr(key, value)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_context_md_del_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    try:
        adapter.ctrl.del_md_attr(key)
    except (AttributeError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {}


def _h_context_ml_del_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        adapter.ctrl.del_ml_module(name)
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"deleted": name}


def _h_context_ml_rename_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    old = str(params["old"])
    new = str(params["new"])
    try:
        adapter.ctrl.rename_ml_module(old, new)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"renamed": new}


def _h_context_ml_rename_waveform(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    old = str(params["old"])
    new = str(params["new"])
    try:
        adapter.ctrl.rename_ml_waveform(old, new)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"renamed": new}


def _h_context_ml_del_waveform(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        adapter.ctrl.del_ml_waveform(name)
    except (KeyError, RuntimeError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"deleted": name}


HANDLERS: dict[str, Handler] = {
    "context.use": _h_context_use,
    "context.new": _h_context_new,
    "context.labels": _h_context_labels,
    "context.active": _h_context_active,
    "context.md_get": _h_context_md_get,
    "context.md_get_attr": _h_context_md_get_attr,
    "value.list": _h_value_list,
    "value.read": _h_value_read,
    "context.ml_get": _h_context_ml_get,
    "context.md_set_attr": _h_context_md_set_attr,
    "context.md_del_attr": _h_context_md_del_attr,
    "context.ml_del_module": _h_context_ml_del_module,
    "context.ml_del_waveform": _h_context_ml_del_waveform,
    "context.ml_rename_module": _h_context_ml_rename_module,
    "context.ml_rename_waveform": _h_context_ml_rename_waveform,
    "context.ml_list_roles": _h_context_ml_list_roles,
    "context.ml_create_from_role": _h_context_ml_create_from_role,
}
