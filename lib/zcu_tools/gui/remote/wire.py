"""Shared wire types for the GUI RemoteControlAdapter (app-agnostic).

Frozen dataclasses for request / response envelopes plus the field-level
validation primitives (``require_str`` / ``require_int`` / ``optional_bool``)
that strictly coerce raw wire scalars before any ``Any`` flows into
``Controller`` or domain services.

This layer is transport-pure: it knows field rules but not the domain shapes
they assemble into. Each GUI app owns its per-app wire/code version constants
(``WIRE_VERSION`` / ``GUI_VERSION``) in its own ``wire_version.py`` — this
shared module carries only the envelope + field-primitive mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

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
