"""Wire types for RemoteControlAdapter.

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
)

from .errors import ErrorCode, ErrorEnvelope, RemoteError

# Hand-maintained wire-protocol version. Bump this whenever the RPC method set,
# their params, or the event/serialization contract changes, so a live process
# advertises which contract it speaks. The GUI server reports it via the
# (no-auth) ``wire.version`` method, and the MCP server pins the version it was
# built against; ``gui_launch``/``gui_connect`` surface both so a stale process
# (one that did not reload the latest code) is immediately visible instead of
# being inferred from start times. Bump deliberately on every wire change.
# v2: added ml.list_roles / ml.create_from_role (role catalog) and
#     context.rename_ml_module / context.rename_ml_waveform; editor.open dropped
#     its discriminator param (from_name-only).
# v3: removed device.set_value (set values via device.setup updates={"value":..});
#     device.connect / device.disconnect now return operation_id (operation handle
#     parity with device.setup); device.snapshot includes the device info payload.
# v4: removed context.set_ml_module / context.set_ml_waveform (raw-dict RPC); ml
#     entries are built/edited via the editor session (create_from_role + editor.*)
#     — ADR-0011, the single ml/md write authority is ContextService.
# v5: added device.setup_spec (discover the fields settable via device.setup's
#     updates — name/type/choices/current/settable — from the live info model).
# v6: removed cfg.set_field — a tab's cfg is edited through its CfgEditorService
#     session (editor.set_field on the tab's editor_id), the same draft the form
#     attaches to, so agent + user share one model (ADR-0013 F11). tab.list_paths
#     now reads that session too (wire shape unchanged).
WIRE_VERSION = 6

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
