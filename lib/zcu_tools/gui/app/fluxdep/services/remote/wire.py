"""Wire types for the fluxdep-gui RemoteControlAdapter.

Frozen dataclasses for request / response envelopes plus the field-level
validation primitives (``require_str`` / ``require_int`` / ``optional_bool`` /
``require_object`` / ``require_json_safe``) that strictly coerce raw wire
scalars before any ``object`` flows into the ``Controller`` or its services.

This layer is transport-pure: it knows field rules but not the fluxdep domain
shapes they assemble into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .errors import ErrorCode, ErrorEnvelope, RemoteError

# Two independent hand-maintained versions reported by the no-auth
# ``wire.version`` handshake (which also surfaces the MCP server's own
# MCP_VERSION). Only WIRE_VERSION is *compared*; the code revisions are
# *reported* so an agent can eyeball whether a reload took effect:
#
#   WIRE_VERSION — the mcp<->RPC *interface contract* (RPC method set, their
#     params, event/serialization shape). The MCP server pins it; a mismatch
#     means the two sides speak different protocols → hard MISMATCH. Bump ONLY
#     on a contract change.
#   GUI_VERSION  — this GUI code's *revision*. Reported, never compared. Bump on
#     any meaningful GUI change you want to be able to spot a reload of,
#     INCLUDING pure-internal logic changes that don't touch the wire.
WIRE_VERSION = 1

# GUI code revision (see header). Bump on any meaningful GUI change you want a
# stale-process check to flag; independent of WIRE_VERSION.
GUI_VERSION = 1

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
    rid = require_str(raw, "id")
    method = require_str(raw, "method")
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


def require_str(params: Mapping[str, object], key: str) -> str:
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


def require_int(params: Mapping[str, object], key: str) -> int:
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


def optional_bool(params: Mapping[str, object], key: str, default: bool) -> bool:
    val = params.get(key)
    if val is None:
        return default
    if not isinstance(val, bool):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            f"'{key}' must be a boolean if present, got {type(val).__name__}",
        )
    return val


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
