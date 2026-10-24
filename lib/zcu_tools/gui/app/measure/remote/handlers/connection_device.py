"""Connection Device remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.wire import require_int, require_str
from zcu_tools.gui.session.device_models import (
    SetupDeviceRequest,
)
from zcu_tools.gui.session.services.connection import (
    ConnectRemoteRequest,
)

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def coerce_connect_request(params: Mapping[str, object]) -> ConnectRemoteRequest:
    """Coerce a real-board connection request; simulation is selected at startup."""
    kind = require_str(params, "kind")
    if kind == "remote":
        return ConnectRemoteRequest(
            ip=require_str(params, "ip"),
            port=require_int(params, "port"),
        )
    raise RemoteError(
        ErrorCode.INVALID_PARAMS,
        "soc.connect requires kind='remote'; "
        "use a trusted startup script (--script) to install simulation resources",
    )


def h_soc_connect(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # Synchronous real-board connection on the owner thread. make_soc_proxy's
    # COMMTIMEOUT bounds unreachable-board failures; simulation is a separate operation.
    req = coerce_connect_request(params)
    adapter.ctrl.connect_sync(req)
    # Return the SoC summary directly — the same {description, is_mock} the old
    # finished short-wait reply folded in. The structured cfg is fetched on demand
    # via soc.info (it is ~2 KB and rarely needed at connect time).
    info = adapter.ctrl.get_soc_info()
    return {"soc": {"description": info["description"], "is_mock": info["is_mock"]}}


def h_device_setup(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = str(params["name"])
    cfg = params["cfg"]  # ParamSpec validated object; Info owns field validation.
    dev = adapter.device_control
    info = dev.get_device_info(name)
    if info is None:
        raise RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            f"Device {name!r} has no Info model for setup",
        )
    try:
        updated = type(info).model_validate(cfg)
    except ValueError as exc:
        raise RemoteError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
    operation_id = dev.start_setup_device(SetupDeviceRequest(name=name, info=updated))
    return {"operation_id": operation_id}


_DEVICE_SETUP_PROTECTED = frozenset({"type", "address"})


def _field_type_and_choices(annotation: object) -> tuple[str, list[object] | None]:
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
    # tolerated: Device annotations may be arbitrary objects; unknown scalar types keep the string fallback.
    return _SCALAR.get(annotation, "str"), None  # pyright: ignore[reportArgumentType, reportCallIssue]


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
            "unit": dev.get_device_unit(name),
            "info": snap.info.to_dict() if snap.info is not None else None,
            "fields": _device_fields(snap.info) if snap.info is not None else [],
        }
    }
