"""Connection Device remote handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

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

from ._common import Handler

logger = logging.getLogger(__name__)


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


def _h_soc_connect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Synchronous connect: runs on the Qt main thread (the IO worker blocks on the
    # _dispatch_on_main marshal), so the connect work + ALL post-connect side
    # effects (State write, soc version bump, SocChangedPayload → FLUX-AWARE-MOCK
    # provisioning) complete before this returns. A connect failure raises a typed
    # RemoteError. The worst-case main-thread block is bounded by make_soc_proxy's
    # 1s COMMTIMEOUT for a remote board (mock is instant). Connect is no longer an
    # async operation handle — run / analyze / device keep theirs.
    req = coerce_connect_request(params)
    try:
        adapter.ctrl.connect_sync(req)
    except Exception as exc:
        raise RemoteError(
            ErrorCode.CONTROLLER_ERROR,
            f"SoC connect failed: {exc}",
        ) from exc
    # Return the SoC summary directly — the same {description, is_mock} the old
    # finished short-wait reply folded in. The structured cfg is fetched on demand
    # via soc.info (it is ~2 KB and rarely needed at connect time).
    info = adapter.ctrl.get_soc_info()
    return {"soc": {"description": info["description"], "is_mock": info["is_mock"]}}


def _h_startup_apply(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.result_scope import ResultScopeError
    from zcu_tools.gui.session.services.startup import StartupProjectRequest

    chip = str(params["chip_name"])
    qub = str(params["qub_name"])
    scope_id_raw = params.get("scope_id")

    req = StartupProjectRequest(
        chip_name=chip,
        qub_name=qub,
        res_name=str(params["res_name"]),
        scope_id=str(scope_id_raw) if scope_id_raw else None,
    )
    # Echo the resolved project (apply always mutates and either succeeds or
    # raises — there is no no-op outcome, so no {applied:false} branch).
    try:
        return adapter.ctrl.apply_startup_project(req)
    except ResultScopeError as exc:
        code = (
            ErrorCode.INVALID_PARAMS
            if exc.reason_code.startswith("scope_")
            else ErrorCode.PRECONDITION_FAILED
        )
        raise RemoteError(code, str(exc), reason=exc.reason_code) from exc


def _h_device_connect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    req = coerce_connect_device_request(params)
    try:
        operation_id = adapter.ctrl.start_connect_device(req)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


def _h_device_disconnect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    req = coerce_disconnect_device_request(params)
    try:
        operation_id = adapter.ctrl.start_disconnect_device(req)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    return {"operation_id": operation_id}


def _h_device_reconnect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        operation_id = adapter.ctrl.start_reconnect_device(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # Reconnect runs asynchronously like connect/disconnect/setup; expose the
    # operation_id so the MCP short-wait/handle path can track it (FC1).
    return {"operation_id": operation_id}


def _h_device_forget(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        adapter.ctrl.forget_device(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # Synchronous sync mutation: echo the forgotten name so the reply is
    # self-verifying (no follow-up read needed to confirm what was dropped).
    return {"forgotten": name}


def _h_device_setup(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    updates = cast(dict, params["updates"])  # ParamSpec(_obj)-validated
    try:
        info = adapter.ctrl.get_device_info(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    if info is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"Device {name!r} has no live info to update",
        )
    try:
        updated = info.with_updates(**dict(updates))
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    try:
        operation_id = adapter.ctrl.start_setup_device(
            SetupDeviceRequest(name=name, info=updated)
        )
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
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


def _h_device_setup_spec(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        info = adapter.ctrl.get_device_info(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    if info is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"Device {name!r} has no live info (connect it first)",
        )
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
    return {"fields": fields}


def _h_device_cancel_operation(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    try:
        adapter.ctrl.cancel_device_operation(name)
    except RuntimeError as exc:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason=getattr(exc, "reason_code", ""),
        ) from exc
    # Self-verifying echo: cancel succeeded (a non-cancellable / absent op raised
    # above). The terminal outcome is observed via the operation handle.
    return {"ok": True, "cancelled": True}


def _h_device_active_operations(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    # Phase C concurrency: enumerate *every* in-flight device operation (sorted
    # by name), each tagged with its kind (connect / disconnect / setup) and its
    # operation 'handle' so the agent can drive gui_op_poll / gui_op_wait per op.
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
            for op in adapter.ctrl.get_active_device_operations()
        ]
    }


def _h_device_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    # Project the fine-grained status (DeviceEntry.status), consistent with the
    # snapshot/active_operations projections (single-status SSOT, FC7).
    devices = [
        {
            "name": e.name,
            "type_name": e.type_name,
            "status": e.status,
        }
        for e in adapter.ctrl.list_devices()
    ]
    return {"devices": devices}


def _h_device_snapshot(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    snap = adapter.ctrl.get_device_snapshot(name)
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
        }
    }


HANDLERS: dict[str, Handler] = {
    "soc.connect": _h_soc_connect,
    "startup.apply": _h_startup_apply,
    "device.connect": _h_device_connect,
    "device.disconnect": _h_device_disconnect,
    "device.reconnect": _h_device_reconnect,
    "device.forget": _h_device_forget,
    "device.setup": _h_device_setup,
    "device.setup_spec": _h_device_setup_spec,
    "device.cancel_operation": _h_device_cancel_operation,
    "device.active_operations": _h_device_active_operations,
    "device.list": _h_device_list,
    "device.snapshot": _h_device_snapshot,
}
