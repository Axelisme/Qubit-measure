"""Fixed-source provenance and owner-sequenced measure Form failures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.state import DEVICE_SET_VERSION_KEY, SessionState


class CfgPreconditionReason(str, Enum):
    """Wire-stable reasons a Form resource cannot accept a command."""

    RESOURCE_GONE = "resource_gone"
    MUTATION_BLOCKED = "mutation_blocked"
    REENTRANT_MUTATION = "reentrant_mutation"
    CAPTURE_UNAVAILABLE = "capture_unavailable"
    NOT_VALID = "not_valid"


class CfgPreconditionError(FailedPreconditionError):
    """Reject a Form command without changing its inputs or publication.

    reason classifies the failed owner precondition. message describes the
    failure to the caller. path is a tuple of field segments, or None when the
    failure applies to the resource. edit_index is the zero-based batch index,
    or None when no particular edit caused the failure.
    """

    def __init__(
        self,
        reason: CfgPreconditionReason,
        message: str,
        *,
        path: tuple[str, ...] | None = None,
        edit_index: int | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason.value)
        self.reason = reason
        self.path = path
        self.edit_index = edit_index


@dataclass(frozen=True)
class SourceRevision:
    """One published source revision captured on the owner sequence.

    source_id is "context", "devices:__set__", or "device:<name>". revision is its
    nonnegative version counter, with zero denoting an unpublished source.
    """

    source_id: str
    revision: int


# Ordered revision evidence for the sources used to open or Reset a Form.
SourceBasis: TypeAlias = tuple[SourceRevision, ...]


def source_basis(state: SessionState) -> SourceBasis:
    """Read context/device-set/cache revision provenance without loading sources.

    state supplies published versions and device identities on the owner
    sequence. The set revision distinguishes removal/recreation when a
    per-device revision starts over. This neither polls nor mutates devices.
    """
    versions = state.version.snapshot()
    return (
        SourceRevision("context", int(versions.get("context", 0))),
        SourceRevision(
            DEVICE_SET_VERSION_KEY,
            int(versions.get(DEVICE_SET_VERSION_KEY, 0)),
        ),
        *(
            SourceRevision(
                f"device:{device.name}",
                int(versions.get(f"device:{device.name}", 0)),
            )
            for device in state.list_devices()
        ),
    )
