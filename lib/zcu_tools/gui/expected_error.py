"""Typed, transport-independent failures that callers can correct."""

from __future__ import annotations

from enum import Enum


class ExpectedErrorCategory(str, Enum):
    """Closed classification for user-correctable failures."""

    INVALID_INPUT = "invalid_input"
    FAILED_PRECONDITION = "failed_precondition"


class ExpectedError(Exception):
    """Nominal opt-in marker for a failure a caller can correct."""

    category: ExpectedErrorCategory
    reason_code: str = ""


class _ExpectedRuntimeError(RuntimeError, ExpectedError):
    def __init__(self, message: str, *, reason_code: str = "") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class InvalidInputError(_ExpectedRuntimeError):
    """The caller must change the request input before retrying."""

    category = ExpectedErrorCategory.INVALID_INPUT


class FailedPreconditionError(_ExpectedRuntimeError):
    """The caller must first change session, resource, or environment state."""

    category = ExpectedErrorCategory.FAILED_PRECONDITION
