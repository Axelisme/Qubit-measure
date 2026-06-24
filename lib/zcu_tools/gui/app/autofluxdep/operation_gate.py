"""OperationGate — autofluxdep-gui's hardware mutual-exclusion policy (ADR-0019).

Owns only the "can this hardware operation start right now" concern: the set of
active leases and the conflict rules. It is execution- and handle-agnostic — the
async handle (poll / await / cancel / operation_id) lives in the shared
``OperationHandles`` leaf, and the off-main execution in autofluxdep's
``BackgroundRunner``.

The conflict *policy* is per-app (session-core extraction, decision 3): the shared
session services name session kinds through the ``ExclusionGate`` port
(``gui/session/ports``), and this concrete gate spans those session kinds **and**
autofluxdep's own ``RUN`` kind (the flux sweep). RUN blocks every interactive
hardware op (soc connect / device mutation) and vice-versa — during a sweep the
worker drives the flux device directly (per-point flux-set), so an interactive
device mutation must not race it. The gate keys leases by the kind's wire string,
so a session kind and ``RUN`` — which live in separate enums — compare uniformly
without sharing one enum.

A hardware op (sweep / device mutation / soc connect) composes both: it mints a
token + handle in ``OperationHandles``, then ``register``s an exclusion here under
the *same* token; the terminal path settles the handle then ``release``s the
exclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from zcu_tools.gui.session.ports import (
    ExclusionGate,
    OperationConflictError,
)
from zcu_tools.gui.session.ports import (
    OperationKind as SessionOpKind,
)

__all__ = ["OperationKind", "OperationConflictError", "OperationGate"]


class OperationKind(str, Enum):
    """autofluxdep-gui's app-specific operation kinds (added to the session ones)."""

    RUN = "run"  # a flux sweep over the node graph


# Conflict vocabulary as wire strings (session kinds + autofluxdep's RUN), so
# leases from the two separate enums compare uniformly.
_RUN = OperationKind.RUN.value
_SOC_CONNECT = SessionOpKind.SOC_CONNECT.value
_DEVICE_MUTATIONS = frozenset(
    {
        SessionOpKind.DEVICE_CONNECT.value,
        SessionOpKind.DEVICE_DISCONNECT.value,
        SessionOpKind.DEVICE_SETUP.value,
    }
)


def _norm(kind: str) -> str:
    """The kind's wire string (an ``Enum`` member -> its value, else as-is)."""
    return kind.value if isinstance(kind, Enum) else str(kind)


@dataclass(frozen=True)
class _ActiveLease:
    """An active exclusion lease (the value side of the token-keyed active map)."""

    kind: str  # wire string of the OperationKind
    owner_id: str
    resource_id: str | None = None


class OperationGate(ExclusionGate):
    """Mutual exclusion among active hardware operations (ADR-0019).

    Keyed by the operation token minted in ``OperationHandles`` (one token
    identifies an operation across both leaves). ``ensure_can_start`` is the
    fail-fast guard the domain service calls *before* opening a handle (so a
    conflict never leaks a half-built operation); ``register`` adds the active
    lease once the handle exists; ``release`` frees it on the terminal path.
    """

    def __init__(self) -> None:
        self._active: dict[int, _ActiveLease] = {}

    def ensure_can_start(self, kind: str, *, resource_id: str | None = None) -> None:
        """Fail-fast guard: raise ``OperationConflictError`` if an active lease
        conflicts with ``kind``. Called before the handle is created, so a
        conflict aborts without leaving a half-built operation behind.

        ``resource_id`` scopes device-mutation conflicts: two device mutations
        conflict only when they target the same device (so different devices can
        be set up concurrently). Global kinds (RUN / soc connect) ignore it."""
        requested = _norm(kind)
        conflicting = next(
            (
                lease
                for lease in self._active.values()
                if self._conflicts(lease, requested, resource_id)
            ),
            None,
        )
        if conflicting is not None:
            raise OperationConflictError(
                f"Cannot start {requested}: {conflicting.kind} is active "
                f"for {conflicting.owner_id!r}"
            )

    def register(
        self,
        token: int,
        kind: str,
        *,
        owner_id: str,
        resource_id: str | None = None,
    ) -> None:
        """Add an active exclusion lease under ``token``. Precondition:
        ``ensure_can_start(kind)`` has passed (single-threaded, so no race
        between the check and here)."""
        if not owner_id:
            raise ValueError("owner_id must not be empty")
        self._active[token] = _ActiveLease(_norm(kind), owner_id, resource_id)

    def release(self, token: int) -> None:
        """Free the hardware: remove the active lease (frees it for the next
        operation). Raises if the token holds no active lease (double release /
        releasing a handle-only op that never registered)."""
        if token not in self._active:
            raise RuntimeError(f"Operation lease {token} is not active")
        del self._active[token]

    def has_active(self, kind: str) -> bool:
        return any(lease.kind == _norm(kind) for lease in self._active.values())

    def is_device_mutating(self, name: str) -> bool:
        return any(
            lease.kind in _DEVICE_MUTATIONS and lease.resource_id == name
            for lease in self._active.values()
        )

    @staticmethod
    def _conflicts(
        existing: _ActiveLease, requested: str, requested_resource: str | None
    ) -> bool:
        existing_kind = existing.kind
        if existing_kind == _RUN:
            return (
                requested == _RUN
                or requested == _SOC_CONNECT
                or requested in _DEVICE_MUTATIONS
            )
        if requested == _RUN:
            return existing_kind == _SOC_CONNECT or existing_kind in _DEVICE_MUTATIONS
        if existing_kind == _SOC_CONNECT:
            return requested == _SOC_CONNECT
        if existing_kind in _DEVICE_MUTATIONS:
            # Resource-aware: two device mutations conflict only on the SAME
            # device — different devices each own an independent driver + lock +
            # VISA session, so concurrent setup is safe (phase C). A None
            # requested_resource (defensive) is treated as conflicting with any
            # in-flight device mutation rather than silently allowed.
            return requested in _DEVICE_MUTATIONS and (
                requested_resource is None or existing.resource_id == requested_resource
            )
        return False
