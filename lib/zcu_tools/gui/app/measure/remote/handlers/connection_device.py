"""Connection Device remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.wire import optional_bool, require_int, require_str
from zcu_tools.gui.session.services.connection import (
    ConnectMockRequest,
    ConnectRemoteRequest,
    ConnectRequest,
)
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def coerce_connect_request(params: Mapping[str, object]) -> ConnectRequest:
    """Coerce ``{kind: 'mock'}`` or ``{kind: 'remote', ip, port}``."""
    kind = require_str(params, "kind")
    if kind == "mock":
        return ConnectMockRequest()
    if kind == "remote":
        return ConnectRemoteRequest(
            ip=require_str(params, "ip"),
            port=require_int(params, "port"),
        )
    raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown connect kind: {kind!r}")


def coerce_connect_device_request(
    params: Mapping[str, object],
) -> ConnectDeviceRequest:
    return ConnectDeviceRequest(
        type_name=require_str(params, "type_name"),
        name=require_str(params, "name"),
        address=require_str(params, "address"),
        remember=optional_bool(params, "remember", True),
    )


def coerce_disconnect_device_request(
    params: Mapping[str, object],
) -> DisconnectDeviceRequest:
    return DisconnectDeviceRequest(
        name=require_str(params, "name"),
        remember=optional_bool(params, "remember", True),
    )


def h_soc_connect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Synchronous connect: runs on the Qt main thread (the IO worker blocks on the
    # _dispatch_on_owner marshal), so the connect work + ALL post-connect side
    # effects (State write, soc version bump, SocChangedPayload → FLUX-AWARE-MOCK
    # provisioning) complete before this returns. Connect failures remain
    # controller errors. The worst-case main-thread block is bounded by make_soc_proxy's
    # 1s COMMTIMEOUT for a remote board (mock is instant). Connect is no longer an
    # async operation handle — run / analyze / device keep theirs.
    req = coerce_connect_request(params)
    adapter.ctrl.connect_sync(req)
    # Return the SoC summary directly — the same {description, is_mock} the old
    # finished short-wait reply folded in. The structured cfg is fetched on demand
    # via soc.info (it is ~2 KB and rarely needed at connect time).
    info = adapter.ctrl.get_soc_info()
    return {"soc": {"description": info["description"], "is_mock": info["is_mock"]}}


def h_project_apply(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.session.services.project_settings import ProjectRequest

    # The handler runs on the GUI owner thread: read current identity, resolve
    # omitted fields and apply the result in this single RPC, never from MCP.
    has_project = adapter.ctrl.has_project()
    current = adapter.ctrl.get_session_env() if has_project else None
    chip_raw = params.get("chip_name")
    qub_raw = params.get("qub_name")
    res_raw = params.get("res_name")
    if current is None and (chip_raw is None or qub_raw is None or res_raw is None):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "chip_name, qub_name and res_name are required for a first project",
            reason="missing_project_fields",
        )
    if current is None:
        chip, qub, res = str(chip_raw), str(qub_raw), str(res_raw)
    else:
        chip = str(chip_raw) if chip_raw is not None else current.chip_name
        qub = str(qub_raw) if qub_raw is not None else current.qub_name
        res = str(res_raw) if res_raw is not None else current.res_name

    scope_id_raw = params.get("scope_id")
    if scope_id_raw is not None:
        scope_id = str(scope_id_raw)
    elif current is not None and (chip, qub) == (current.chip_name, current.qub_name):
        scope_id = adapter.ctrl.setup_control.get_setup_preferences().scope_id or None
    else:
        scope_id = None
    req = ProjectRequest(chip_name=chip, qub_name=qub, res_name=res, scope_id=scope_id)
    return adapter.ctrl.apply_project(req)


def h_device_connect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    req = coerce_connect_device_request(params)
    dev = adapter.device_control
    operation_id = dev.start_connect_device(req)
    return {"operation_id": operation_id}


def h_device_disconnect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    req = coerce_disconnect_device_request(params)
    dev = adapter.device_control
    operation_id = dev.start_disconnect_device(req)
    return {"operation_id": operation_id}


def h_device_reconnect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    dev = adapter.device_control
    operation_id = dev.start_reconnect_device(name)
    # Reconnect runs asynchronously like connect/disconnect/setup; expose the
    # operation_id so the MCP short-wait/handle path can track it (FC1).
    return {"operation_id": operation_id}


def h_device_forget(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    dev = adapter.device_control
    dev.forget_device(name)
    # Synchronous sync mutation: echo the forgotten name so the reply is
    # self-verifying (no follow-up read needed to confirm what was dropped).
    return {"forgotten": name}


def h_device_setup(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    updates = cast(dict, params["updates"])  # ParamSpec(_obj)-validated
    dev = adapter.device_control
    info = dev.get_device_info(name)
    if info is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"Device {name!r} has no live info to update",
        )
    try:
        updated = info.with_updates(**dict(updates))
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    operation_id = dev.start_setup_device(SetupDeviceRequest(name=name, info=updated))
    return {"operation_id": operation_id}


_DEVICE_SETUP_PROTECTED = frozenset({"type", "address"})


def _field_type_and_choices(annotation: object) -> tuple[str, list | None]:
    """Wire (type, choices) for a BaseDeviceInfo field annotation.

    Literal[...] → ('enum', [members]). Optional[X] unwraps to X.
    Otherwise map the scalar python type to a JSON type name.

    Pydantic model fields use typing.Literal; get_origin returns the Literal
    sentinel from typing (typing_extensions.Literal is an alias post-3.11).
    """
    import types
    import typing

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    # Literal: get_origin returns typing.Literal for both typing.Literal and
    # typing_extensions.Literal on Python 3.11+ (they are the same object).
    if origin is typing.Literal:
        return "enum", list(args)
    # Optional[X] / Union[X, None] → unwrap to the non-None member.
    # Accept both typing.Union (Optional[T]) and types.UnionType (PEP 604,
    # X | None): get_origin returns types.UnionType for the latter.
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _field_type_and_choices(non_none[0])
    _SCALAR = {float: "float", int: "int", str: "str", bool: "bool"}
    return _SCALAR.get(annotation, "str"), None  # type: ignore[arg-type]


def _device_fields(info: BaseDeviceInfo) -> list[dict[str, object]]:
    """Project either live or State-cached device info with one field grammar."""
    fields: list[dict[str, object]] = []
    for fname, finfo in type(info).model_fields.items():
        ftype, choices = _field_type_and_choices(finfo.annotation)
        entry: dict[str, object] = {
            "name": fname,
            "type": ftype,
            "current": getattr(info, fname, None),
            "settable": fname not in _DEVICE_SETUP_PROTECTED,
        }
        if choices is not None:
            entry["choices"] = choices
        fields.append(entry)
    return fields


def h_device_setup_spec(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    info = adapter.device_control.get_device_info(name)
    if info is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"Device {name!r} has no live info (connect it first)",
        )
    return {"fields": _device_fields(info)}


def h_device_cancel_operation(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    dev = adapter.device_control
    dev.cancel_device_operation(name)
    # Self-verifying echo: cancel succeeded (a non-cancellable / absent op raised
    # above). The terminal outcome is observed via the operation handle.
    return {"ok": True, "cancelled": True}


def h_device_active_operations(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    dev = adapter.device_control
    # Phase C concurrency: enumerate *every* in-flight device operation (sorted
    # by name), each tagged with its kind (connect / disconnect / setup) and its
    # operation 'handle' so the agent can wait on each operation id.
    # device_name is the SSOT key; the duplicate snapshot.name field is dropped.
    return {
        "operations": [
            {
                "handle": op.token,
                "device_name": op.device_name,
                "kind": op.kind.value,
                "type_name": op.snapshot.type_name,
                "address": op.snapshot.address,
                "status": op.snapshot.status.value,
                "error": op.snapshot.error,
            }
            for op in dev.get_active_device_operations()
        ]
    }


def h_device_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    dev = adapter.device_control
    # Project the fine-grained status (DeviceEntry.status), consistent with the
    # snapshot/active_operations projections (single-status SSOT, FC7).
    devices = [
        {
            "name": e.name,
            "type_name": e.type_name,
            "status": e.status,
        }
        for e in dev.list_devices()
    ]
    return {"devices": devices}


def h_device_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    dev = adapter.device_control
    snap = dev.get_device_snapshot(name)
    if snap is None:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown device: {name!r}")
    # ``info`` is a ``BaseDeviceInfo`` (a pydantic ``ConfigBase``); ``to_dict()``
    # yields JSON-safe scalars (address/type/label + driver fields like the
    # source ``value``), so the agent can read the device's live parameters.
    return {
        "snapshot": {
            "name": snap.name,
            "type_name": snap.type_name,
            "address": snap.address,
            "status": snap.status.value,
            "error": snap.error,
            "info": snap.info.to_dict() if snap.info is not None else None,
            "fields": _device_fields(snap.info) if snap.info is not None else [],
        }
    }
