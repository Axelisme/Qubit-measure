"""Public contracts for owner-sequenced, versioned cfg resources.

Editing handles identify one resource for its entire lifetime. Implementations
must detach inputs and observations, publish batches atomically, and reject
mutations and acceptance during notification. App composition owns creation,
revocation, and the allocation of the narrower editing/acceptance capabilities.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

from .binding.observation import CfgNodeObservation
from .model import DirectValue, EvalValue

CfgPath: TypeAlias = tuple[str, ...]
CfgInput: TypeAlias = (
    None
    | bool
    | int
    | float
    | complex
    | str
    | DirectValue
    | EvalValue
    | Sequence["CfgInput"]
    | Mapping[str, "CfgInput"]
)


class CfgInputReason(str, Enum):
    MALFORMED_INPUT = "malformed_input"
    UNKNOWN_PATH = "unknown_path"
    READONLY = "readonly"
    UNSUPPORTED_MODE = "unsupported_mode"
    INVALID_VALUE = "invalid_value"
    CAPTURE_SYNTAX = "capture_syntax"


class CfgPreconditionReason(str, Enum):
    STALE_REVISION = "stale_revision"
    RESOURCE_GONE = "resource_gone"
    MUTATION_BLOCKED = "mutation_blocked"
    REENTRANT_MUTATION = "reentrant_mutation"
    CAPTURE_UNAVAILABLE = "capture_unavailable"
    NOT_VALID = "not_valid"


class CfgInputError(InvalidInputError):
    def __init__(
        self,
        reason: CfgInputReason,
        message: str,
        *,
        path: CfgPath | None = None,
        edit_index: int | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason.value)
        self.reason = reason
        self.path = path
        self.edit_index = edit_index


class CfgPreconditionError(FailedPreconditionError):
    def __init__(
        self,
        reason: CfgPreconditionReason,
        message: str,
        *,
        path: CfgPath | None = None,
        edit_index: int | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason.value)
        self.reason = reason
        self.path = path
        self.edit_index = edit_index


class CfgId(str):
    def __new__(cls, value: object) -> CfgId:
        if not isinstance(value, str) or not value:
            raise CfgInputError(
                CfgInputReason.MALFORMED_INPUT, "cfg_id must be a nonempty string"
            )
        return super().__new__(cls, value)


class CfgRevision(int):
    def __new__(cls, value: object) -> CfgRevision:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CfgInputError(
                CfgInputReason.MALFORMED_INPUT,
                "revision must be a nonnegative integer, not bool",
            )
        return super().__new__(cls, value)


@dataclass(frozen=True)
class CfgRef:
    cfg_id: CfgId
    revision: CfgRevision


@dataclass(frozen=True)
class CfgEdit:
    path: CfgPath
    value: CfgInput


class CfgStatus(str, Enum):
    VALID = "Valid"
    INVALID = "Invalid"
    UNAVAILABLE = "Unavailable"


@dataclass(frozen=True)
class SourceRevision:
    source_id: str
    revision: CfgRevision


SourceBasis: TypeAlias = tuple[SourceRevision, ...]


@dataclass(frozen=True)
class CfgDiagnostic:
    path: CfgPath
    reason: str
    message: str


@dataclass(frozen=True)
class CfgObservation:
    ref: CfgRef
    status: CfgStatus
    tree: CfgNodeObservation
    source_basis: SourceBasis
    diagnostics: tuple[CfgDiagnostic, ...]


@dataclass(frozen=True)
class AcceptedConfig:
    ref: CfgRef
    source_basis: SourceBasis
    values: dict[str, object]


class CfgStaleError(CfgPreconditionError):
    def __init__(self, expected: CfgRef, actual: CfgRef) -> None:
        super().__init__(
            CfgPreconditionReason.STALE_REVISION,
            f"Expected cfg revision {expected.revision}, current is {actual.revision}",
        )
        self.expected = expected
        self.actual = actual


class CfgEditing(Protocol):
    def observe(self) -> CfgObservation: ...

    def watch(
        self, callback: Callable[[CfgObservation], None]
    ) -> Callable[[], None]: ...

    def edit(
        self, expected_revision: CfgRevision, edits: tuple[CfgEdit, ...]
    ) -> CfgObservation: ...

    def reset(self, expected_revision: CfgRevision) -> CfgObservation: ...

    def refresh(self, expected_revision: CfgRevision) -> CfgObservation: ...


class CfgAcceptance(Protocol):
    def accept(self, expected_revision: CfgRevision) -> AcceptedConfig: ...
