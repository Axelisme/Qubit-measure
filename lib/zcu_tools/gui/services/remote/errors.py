"""Typed error envelope for RemoteControlService.

Wire format:

    {"id": "...", "ok": false, "error": {"code": "<code>", "message": "..."}}

Codes are a closed enum; new codes require updating both ends of the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
    """Raised inside handlers / coercion to short-circuit with a typed code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ErrorEnvelope:
    code: str
    message: str

    @classmethod
    def from_remote_error(cls, exc: RemoteError) -> "ErrorEnvelope":
        return cls(code=exc.code.value, message=exc.message)

    def to_wire(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}
