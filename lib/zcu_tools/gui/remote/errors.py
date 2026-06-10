"""Typed error envelope for RemoteControlAdapter.

Wire format:

    {"id": "...", "ok": false, "error": {"code": "<code>", "message": "..."}}

Codes are a closed enum; new codes require updating both ends of the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCode(str, Enum):
    UNKNOWN_METHOD = "unknown_method"
    INVALID_PARAMS = "invalid_params"
    CONTROLLER_ERROR = "controller_error"
    PRECONDITION_FAILED = "precondition_failed"
    TIMEOUT = "timeout"
    UNAUTHORIZED = "unauthorized"
    BUSY = "busy"
    INTERNAL = "internal"
    SHUTTING_DOWN = "shutting_down"


class RemoteError(Exception):
    """Raised inside handlers / coercion to short-circuit with a typed code.

    ``reason`` is an optional stable machine-readable sub-tag carried alongside
    the closed ``code`` enum (e.g. ``code=precondition_failed`` +
    ``reason="no_run_result"``), so agents can branch on it without parsing the
    human ``message`` or widening the closed code set.

    ``data`` is an optional structured payload for the few errors that carry
    machine-readable detail beyond a tag — e.g. a stale-version guard naming the
    resource keys that moved (``data={"stale": ["tab:X:cfg", "context"]}``), so
    the mcp layer can translate them into agent language. It must stay free of
    RPC<->mcp bookkeeping the agent should not see (no version numbers).
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        reason: str = "",
        data: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.reason = reason
        self.data = data


@dataclass(frozen=True)
class ErrorEnvelope:
    code: str
    message: str
    reason: str = ""
    data: dict | None = None

    @classmethod
    def from_remote_error(cls, exc: RemoteError) -> ErrorEnvelope:
        return cls(
            code=exc.code.value,
            message=exc.message,
            reason=exc.reason,
            data=exc.data,
        )

    def to_wire(self) -> dict:
        wire: dict = {"code": self.code, "message": self.message}
        if self.reason:
            wire["reason"] = self.reason
        if self.data:
            wire["data"] = self.data
        return wire
