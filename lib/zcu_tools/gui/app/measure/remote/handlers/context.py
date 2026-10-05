"""Context remote handlers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES
from zcu_tools.gui.app.measure.services.ports import CfgEdit
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.value_lookup import ValueInfo

from ._wire_values import context_wire_value

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def h_context_ml_edit(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    raw = params["edits"]
    if not isinstance(raw, list) or not raw:
        raise RemoteError(ErrorCode.INVALID_PARAMS, "edits must be a nonempty list")
    edits: list[CfgEdit] = []
    for index, item in enumerate(raw):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or "value" not in item
        ):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"edits[{index}] requires a nonempty string path and value",
            )
        edits.append(CfgEdit(item["path"], item["value"]))
    save_as = params["save_as"]
    return adapter.ctrl.edit_library(
        str(params["kind"]),
        str(params["name"]),
        edits,
        save_as=str(save_as) if save_as is not None else None,
    ).to_wire()


def h_context_use(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    ctx = adapter.context_control
    # A context lives under a project; without one there are no labels to switch
    # to. Map that precondition to agent language rather than leaking a controller
    # error (mirror h_context_new).
    if not ctx.has_project():
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            "No project applied yet; use rpc_call(method='project.apply', params=...) first.",
            reason="no_project",
        )
    label = str(params["label"])
    available = list(ctx.get_context_labels())
    if label not in available:
        # Fast-fail an unknown label with the valid choices so the agent can
        # correct without a separate rpc_call on context.labels.
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"unknown context label: {label!r}; available: {available}",
            reason="unknown_context",
        )
    ctx.use_context(label)
    active = ctx.get_active_context_label()
    return {
        "label": active,
        "has_active_context": active is not None,
    }


def h_context_new(
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
            "No project applied yet; use rpc_call(method='project.apply', params=...) first.",
            reason="no_project",
        )
    label = params.get("label")
    bind_device = params["bind_device"]
    clone_from = params["clone_from"]
    source = ctx.get_active_context_label() if clone_from == "current" else clone_from
    try:
        ctx.new_context(
            label=str(label) if label is not None else None,
            bind_device=str(bind_device) if bind_device is not None else None,
            clone_from=str(source) if source is not None else None,
        )
    except FileExistsError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"context label {label!r} already exists; use context_use or provide a different label",
            reason="context_exists",
        ) from exc
    # new_context makes the new context active — return its label so the agent
    # knows what was created without a follow-up read.
    label = ctx.get_active_context_label()
    return {"label": label, "has_active_context": label is not None}


def h_context_labels(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"labels": list(adapter.context_control.get_context_labels())}


def h_context_active(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"label": adapter.context_control.get_active_context_label()}


def h_context_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    ctx = adapter.context_control
    md = ctx.get_current_md()
    ml = ctx.get_current_ml()
    try:
        snapshot = {
            "label": ctx.get_active_context_label(),
            "md": {key: value for key, value in sorted(md.items())},
            "ml": {
                "modules": {
                    name: cfg.to_dict() for name, cfg in sorted(ml.modules.items())
                },
                "waveforms": {
                    name: cfg.to_dict() for name, cfg in sorted(ml.waveforms.items())
                },
            },
        }
        # Validate every nested value before JSON encoding. Success is a full
        # context observation and advances the MCP guard baseline.
        return json.loads(json.dumps(context_wire_value(snapshot), allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"cannot fully snapshot the active context: {exc}",
            reason="unserializable_context",
        ) from exc


def _md_summary(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return f"{' × '.join(str(size) for size in value.shape)} array"
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(row, (list, tuple)) for row in value):
            widths = {len(row) for row in value}
            if len(widths) == 1:
                return f"{len(value)} × {widths.pop()} matrix"
        return f"{len(value)} items"
    if isinstance(value, dict):
        return f"{len(value)} keys"
    return type(value).__name__


def h_context_md_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    md = adapter.context_control.get_current_md()
    keys = sorted(str(k) for k in md.keys())
    if not params.get("summaries", False):
        return {"keys": keys}
    return {"keys": keys, "values": {key: _md_summary(md.get(key)) for key in keys}}


def h_context_md_get_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    md = adapter.context_control.get_current_md()
    sentinel = object()
    value = md.get(key, sentinel)
    if value is sentinel:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"unknown md key: {key!r}; available: {sorted(map(str, md.keys()))}",
            reason="unknown_md_key",
        )
    try:
        return {"key": key, "value": context_wire_value(value)}
    except (TypeError, ValueError, RecursionError) as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"cannot fully read MetaDict key {key!r}: {exc}",
            reason="unserializable_context",
        ) from exc


def _value_info_to_wire(info: ValueInfo) -> dict[str, object]:
    return {
        "key": info.key,
        "type": info.type_name,
        "owner": info.owner,
        "description": info.description,
    }


def h_value_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {
        "values": [
            _value_info_to_wire(info)
            for info in adapter.context_control.list_value_sources()
        ]
    }


def h_value_read(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    raw_type = params.get("type")
    if raw_type is not None and not isinstance(raw_type, str):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "'type' must be a string")
    type_name = raw_type
    try:
        info, value = adapter.context_control.read_value_source(key, type_name)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {**_value_info_to_wire(info), "value": value}


def h_context_ml_get(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    ml = adapter.context_control.get_current_ml()
    raw_name = params.get("name")
    raw_kind = params.get("kind")
    if raw_kind is not None and raw_kind not in ("module", "waveform"):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "kind must be 'module' or 'waveform'"
        )
    if raw_name is not None:
        if not isinstance(raw_name, str) or not raw_name:
            raise RemoteError(ErrorCode.INVALID_PARAMS, "name must be nonempty")
        matches = [
            kind
            for kind, collection in (("module", ml.modules), ("waveform", ml.waveforms))
            if raw_name in collection
        ]
        if not matches:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"unknown library name {raw_name!r}; available modules: "
                f"{sorted(ml.modules)}, waveforms: {sorted(ml.waveforms)}",
            )
        if raw_kind is None and len(matches) > 1:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"ambiguous library name {raw_name!r}; supply kind='module' or 'waveform'",
            )
        if raw_kind is not None and raw_kind not in matches:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"no {raw_kind} named {raw_name!r}; available kinds: {matches}",
            )
        kind = raw_kind or matches[0]
        cfg = ml.modules[raw_name] if kind == "module" else ml.waveforms[raw_name]
        return {"name": raw_name, "kind": kind, "cfg": cfg.to_dict()}

    # Descriptions belong to the GUI's live shape catalog, not an MCP copy.
    return {
        "modules": [
            {
                "name": name,
                "kind": cfg.type,
                "description": PROGRAM_SHAPES.module(cfg.type).label,
            }
            for name, cfg in sorted(ml.modules.items())
        ],
        "waveforms": [
            {
                "name": name,
                "style": cfg.style,
                "description": PROGRAM_SHAPES.waveform(cfg.style).label,
            }
            for name, cfg in sorted(ml.waveforms.items())
        ],
    }


def h_context_ml_list_templates(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Project template metadata through the unchanged context.ml_list_roles RPC."""
    del params
    catalog = adapter.ctrl.get_template_catalog()
    return {"roles": list(catalog.list_meta())}


def h_context_ml_create_from_template(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Create a ModuleLibrary entry from the template ID in wire role_id.

    One-shot: seeds md-linked defaults (lowered against the live md), writes ml.
    Edit afterwards via editor.new(from_name=...).
    """
    template_id = str(params["role_id"])
    name = str(params["name"])
    # The item kind is a property of the template, not an independent agent input —
    # derive it from role_id so the agent cannot pass a mismatching pair. An
    # unknown role_id fails fast as invalid_params; a missing catalog (no project)
    # surfaces as precondition_failed (mirror h_context_ml_list_templates).
    try:
        item_kind = adapter.ctrl.get_template_catalog().get(template_id).item_kind
    except KeyError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    try:
        adapter.ctrl.create_from_template(item_kind, template_id, name)
    except KeyError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    return {"created": name}


def h_context_md_set_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    value = params["value"]
    ctx = adapter.context_control
    receipt = params.get("receipt", False)
    sentinel = object()
    previous = ctx.get_current_md().get(key, sentinel) if receipt else sentinel
    ctx.set_md_attr(key, value)
    if not receipt:
        return {}
    current = ctx.get_current_md().get(key, sentinel)
    return {
        "before": None if previous is sentinel else context_wire_value(previous),
        "after": context_wire_value(current),
    }


def h_context_md_del_attr(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    key = str(params["key"])
    adapter.context_control.del_md_attr(key)
    return {}


def h_context_ml_del_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    adapter.context_control.del_ml_module(name)
    return {"deleted": name}


def h_context_ml_rename_module(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    old = str(params["old"])
    new = str(params["new"])
    adapter.context_control.rename_ml_module(old, new)
    return {"renamed": new}


def h_context_ml_rename_waveform(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    old = str(params["old"])
    new = str(params["new"])
    adapter.context_control.rename_ml_waveform(old, new)
    return {"renamed": new}


def h_context_ml_del_waveform(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    adapter.context_control.del_ml_waveform(name)
    return {"deleted": name}
