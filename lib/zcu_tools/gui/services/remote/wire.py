"""Wire types for RemoteControlService.

Frozen dataclasses for request / response envelopes plus strict JSON ↔ dataclass
coercion helpers. All raw-dict validation happens here; no ``Any`` from the wire
ever flows into ``Controller`` or domain services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from zcu_tools.gui.services.connection import (
    ConnectMockRequest,
    ConnectRemoteRequest,
    ConnectRequest,
)
from zcu_tools.gui.services.device import (
    ConnectDeviceRequest,
    DisconnectDeviceRequest,
    SetDeviceValueRequest,
)

from .errors import ErrorCode, ErrorEnvelope, RemoteError

# ---------------------------------------------------------------------------
# Wire envelopes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    id: str
    method: str
    params: Mapping[str, object]


@dataclass(frozen=True)
class Response:
    id: str
    ok: bool
    result: Optional[Mapping[str, object]] = None
    error: Optional[ErrorEnvelope] = None
    # Sidecar GUI-change summary (the change buffer drained at reply time);
    # carried at the envelope level so it never pollutes a tool's result schema
    # and rides on both ok and error replies. Omitted from the wire when empty.
    gui_changes: Optional[list[dict[str, object]]] = None

    def to_wire(self) -> dict[str, object]:
        if self.ok:
            wire: dict[str, object] = {
                "id": self.id,
                "ok": True,
                "result": self.result or {},
            }
        else:
            assert self.error is not None
            wire = {"id": self.id, "ok": False, "error": self.error.to_wire()}
        if self.gui_changes:
            wire["gui_changes"] = self.gui_changes
        return wire


def parse_request(raw: Mapping[str, object]) -> Request:
    rid = _require_str(raw, "id")
    method = _require_str(raw, "method")
    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'params' must be an object, got {type(params).__name__}",
        )
    return Request(id=rid, method=method, params=params)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _require_str(params: Mapping[str, object], key: str) -> str:
    val = params.get(key)
    if val is None:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"missing '{key}'")
    if not isinstance(val, str):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be a string, got {type(val).__name__}",
        )
    if not val:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"'{key}' must be non-empty")
    return val


def _require_int(params: Mapping[str, object], key: str) -> int:
    val = params.get(key)
    if val is None:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"missing '{key}'")
    # bool is a subclass of int in Python; reject it explicitly
    if isinstance(val, bool) or not isinstance(val, int):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be an integer, got {type(val).__name__}",
        )
    return val


def _require_bool(params: Mapping[str, object], key: str) -> bool:
    val = params.get(key)
    if val is None:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"missing '{key}'")
    if not isinstance(val, bool):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be a boolean, got {type(val).__name__}",
        )
    return val


def _optional_str(params: Mapping[str, object], key: str) -> Optional[str]:
    val = params.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be a string if present, got {type(val).__name__}",
        )
    return val


def _optional_bool(params: Mapping[str, object], key: str, default: bool) -> bool:
    val = params.get(key)
    if val is None:
        return default
    if not isinstance(val, bool):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be a boolean if present, got {type(val).__name__}",
        )
    return val


def _optional_float(params: Mapping[str, object], key: str) -> Optional[float]:
    val = params.get(key)
    if val is None:
        return None
    if isinstance(val, bool):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{key}' must be a number if present"
        )
    if not isinstance(val, (int, float)):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be a number if present, got {type(val).__name__}",
        )
    return float(val)


def require_object(params: Mapping[str, object], key: str) -> Mapping[str, object]:
    val = params.get(key)
    if val is None:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"missing '{key}'")
    if not isinstance(val, dict):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be an object, got {type(val).__name__}",
        )
    return val


def require_json_safe(params: Mapping[str, object], key: str) -> object:
    import json

    val = params.get(key)
    if key not in params:
        raise RemoteError(ErrorCode.INVALID_PARAMS, f"missing '{key}'")
    try:
        json.dumps(val)
    except (TypeError, ValueError) as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, f"'{key}' must be JSON-serializable"
        ) from exc
    return val


# ---------------------------------------------------------------------------
# Typed-request coercion
# ---------------------------------------------------------------------------


def coerce_connect_request(params: Mapping[str, object]) -> ConnectRequest:
    """Coerce ``{kind: 'mock'}`` or ``{kind: 'remote', ip, port}``."""
    kind = _require_str(params, "kind")
    if kind == "mock":
        return ConnectMockRequest()
    if kind == "remote":
        return ConnectRemoteRequest(
            ip=_require_str(params, "ip"),
            port=_require_int(params, "port"),
        )
    raise RemoteError(ErrorCode.INVALID_PARAMS, f"unknown connect kind: {kind!r}")


def coerce_connect_device_request(
    params: Mapping[str, object],
) -> ConnectDeviceRequest:
    return ConnectDeviceRequest(
        type_name=_require_str(params, "type_name"),
        name=_require_str(params, "name"),
        address=_require_str(params, "address"),
        remember=_optional_bool(params, "remember", True),
    )


def coerce_disconnect_device_request(
    params: Mapping[str, object],
) -> DisconnectDeviceRequest:
    return DisconnectDeviceRequest(
        name=_require_str(params, "name"),
        remember=_optional_bool(params, "remember", True),
    )


def coerce_set_device_value_request(
    params: Mapping[str, object],
) -> SetDeviceValueRequest:
    return SetDeviceValueRequest(
        name=_require_str(params, "name"),
        value=require_json_safe(params, "value"),
    )
