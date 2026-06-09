"""OperationGate — hardware mutual-exclusion, the Exclusion facet (ADR-0019).

Owns only the "can this hardware operation start right now" concern: the set of
active leases and the conflict rules. It is execution- and handle-agnostic — the
async handle (poll / await / cancel / operation_id) lives in the sibling
``OperationHandles`` leaf, and the off-main execution in ``BackgroundService``.

A hardware op (run / device mutation / soc connect) composes both: it mints a
token + handle in ``OperationHandles``, then ``register``s an exclusion here
under the *same* token; the terminal path settles the handle then ``release``s
the exclusion. An analyze / interactive op takes no exclusion at all (it never
conflicts) — so it no longer fakes a lease just to obtain a handle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OperationKind(Enum):
    RUN = "run"
    SOC_CONNECT = "soc_connect"
    DEVICE_CONNECT = "device_connect"
    DEVICE_DISCONNECT = "device_disconnect"
    DEVICE_SETUP = "device_setup"


class OperationConflictError(RuntimeError):
    """Raised when a hardware operation conflicts with an active operation."""


_DEVICE_MUTATIONS = frozenset(
    {
        OperationKind.DEVICE_CONNECT,
        OperationKind.DEVICE_DISCONNECT,
        OperationKind.DEVICE_SETUP,
    }
)


@dataclass(frozen=True)
class _ActiveLease:
    """An active exclusion lease (the value side of the token-keyed active map)."""

    kind: OperationKind
    owner_id: str
    resource_id: Optional[str] = None


class OperationGate:
    """Mutual exclusion among active hardware operations (ADR-0019).

    Keyed by the operation token minted in ``OperationHandles`` (one token
    identifies an operation across both leaves). ``ensure_can_start`` is the
    fail-fast guard the domain service calls *before* opening a handle (so a
    conflict never leaks a half-built operation); ``register`` adds the active
    lease once the handle exists; ``release`` frees it on the terminal path.
    """

    def __init__(self) -> None:
        self._active: dict[int, _ActiveLease] = {}

    def ensure_can_start(self, kind: OperationKind) -> None:
        """Fail-fast guard: raise ``OperationConflictError`` if an active lease
        conflicts with ``kind``. Called before the handle is created, so a
        conflict aborts without leaving a half-built operation behind."""
        conflicting = next(
            (
                lease
                for lease in self._active.values()
                if self._conflicts(lease.kind, kind)
            ),
            None,
        )
        if conflicting is not None:
            raise OperationConflictError(
                f"Cannot start {kind.value}: {conflicting.kind.value} is active "
                f"for {conflicting.owner_id!r}"
            )

    def register(
        self,
        token: int,
        kind: OperationKind,
        *,
        owner_id: str,
        resource_id: Optional[str] = None,
    ) -> None:
        """Add an active exclusion lease under ``token``. Precondition:
        ``ensure_can_start(kind)`` has passed (single-threaded, so no race
        between the check and here)."""
        if not owner_id:
            raise ValueError("owner_id must not be empty")
        self._active[token] = _ActiveLease(kind, owner_id, resource_id)

    def release(self, token: int) -> None:
        """Free the hardware: remove the active lease (frees it for the next
        operation). Raises if the token holds no active lease (double release /
        releasing a handle-only op that never registered)."""
        if token not in self._active:
            raise RuntimeError(f"Operation lease {token} is not active")
        del self._active[token]

    def has_active(self, kind: OperationKind) -> bool:
        return any(lease.kind is kind for lease in self._active.values())

    def is_device_mutating(self, name: str) -> bool:
        return any(
            lease.kind in _DEVICE_MUTATIONS and lease.resource_id == name
            for lease in self._active.values()
        )

    @staticmethod
    def _conflicts(existing: OperationKind, requested: OperationKind) -> bool:
        if existing is OperationKind.RUN:
            return (
                requested is OperationKind.RUN
                or requested is OperationKind.SOC_CONNECT
                or requested in _DEVICE_MUTATIONS
            )
        if requested is OperationKind.RUN:
            return (
                existing is OperationKind.SOC_CONNECT or existing in _DEVICE_MUTATIONS
            )
        if existing is OperationKind.SOC_CONNECT:
            return requested is OperationKind.SOC_CONNECT
        if existing in _DEVICE_MUTATIONS:
            return requested in _DEVICE_MUTATIONS
        return False
