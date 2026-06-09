"""Session ↔ app seam — ports the session services depend on.

The session services (connection / device / context / ...) never reach for a
concrete app collaborator; they depend on the narrow Protocols here and the app
injects its concrete implementation. This keeps the session core app-agnostic and
free of a back-edge to any ``gui.app.*`` package.

This module holds the **exclusion seam**: the session-core operation-kind
vocabulary (``OperationKind``), the conflict error, and the ``ExclusionGate``
port. Each app keeps its own concrete ``OperationGate`` (the conflict *policy*)
and adds its own app-specific kinds (measure: ``run``; autofluxdep: a sweep kind);
the port is keyed by the kind's wire string so a session service can name a
session kind without the gate's full vocabulary leaking here (ADR-0019, decision
3 of the session-core extraction).

More driven-adapter ports (driver factory / project IO / progress transport) join
this module as the session services move in.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Protocol


class OperationKind(str, Enum):
    """Session-core hardware operation kinds (the ones session services emit).

    A ``str``-valued enum: a member IS its wire string, so it passes straight
    through the str-keyed :class:`ExclusionGate`. App-specific kinds (measure's
    ``run``, a future sweep kind) live in the app and are added to the app's gate
    policy — they are deliberately absent here.
    """

    SOC_CONNECT = "soc_connect"
    DEVICE_CONNECT = "device_connect"
    DEVICE_DISCONNECT = "device_disconnect"
    DEVICE_SETUP = "device_setup"


class OperationConflictError(RuntimeError):
    """Raised when a hardware operation conflicts with an active operation."""


class ExclusionGate(Protocol):
    """The hardware mutual-exclusion seam a session service depends on.

    ``kind`` is the operation kind's wire string (an ``OperationKind`` member
    passes directly, being a ``str``). The concrete gate (app-owned) holds the
    conflict policy across both session kinds and the app's own kinds; a session
    service only ever names session kinds through this port.
    """

    def ensure_can_start(self, kind: str) -> None:
        """Fail-fast: raise ``OperationConflictError`` if an active lease
        conflicts with ``kind`` (called before the handle is opened)."""
        ...

    def register(
        self,
        token: int,
        kind: str,
        *,
        owner_id: str,
        resource_id: Optional[str] = None,
    ) -> None:
        """Add an active exclusion lease under ``token`` (after ensure_can_start)."""
        ...

    def release(self, token: int) -> None:
        """Free the lease held by ``token`` on the terminal path."""
        ...

    def is_device_mutating(self, name: str) -> bool:
        """True while a device-mutation lease (connect/disconnect/setup) for
        ``name`` is active — guards a snapshot read against a racing mutation."""
        ...
