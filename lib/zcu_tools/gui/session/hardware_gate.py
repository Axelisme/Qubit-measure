"""Shared hardware mutual-exclusion gate for measurement-session apps."""

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

__all__ = ["RunBlocksHardwareGate"]


_SOC_CONNECT = SessionOpKind.SOC_CONNECT.value
_DEVICE_MUTATIONS = frozenset(
    {
        SessionOpKind.DEVICE_CONNECT.value,
        SessionOpKind.DEVICE_DISCONNECT.value,
        SessionOpKind.DEVICE_SETUP.value,
    }
)


def _norm(kind: str) -> str:
    """Return the kind's wire string."""

    return kind.value if isinstance(kind, Enum) else str(kind)


@dataclass(frozen=True)
class _ActiveLease:
    """An active exclusion lease keyed by an operation token."""

    kind: str
    owner_id: str
    resource_id: str | None = None


class RunBlocksHardwareGate(ExclusionGate):
    """Mutual exclusion among RUN, SoC connect, and device mutations.

    Measurement-session apps own their app-local RUN enum value, but the hardware
    policy is shared: RUN blocks RUN, SoC connect, and any device mutation;
    device mutations conflict only with mutations of the same device.
    """

    def __init__(self, *, run_kind: str) -> None:
        self._run = _norm(run_kind)
        self._active: dict[int, _ActiveLease] = {}

    def ensure_can_start(self, kind: str, *, resource_id: str | None = None) -> None:
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
        if not owner_id:
            raise ValueError("owner_id must not be empty")
        self._active[token] = _ActiveLease(_norm(kind), owner_id, resource_id)

    def release(self, token: int) -> None:
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

    def _conflicts(
        self,
        existing: _ActiveLease,
        requested: str,
        requested_resource: str | None,
    ) -> bool:
        existing_kind = existing.kind
        if existing_kind == self._run:
            return (
                requested == self._run
                or requested == _SOC_CONNECT
                or requested in _DEVICE_MUTATIONS
            )
        if requested == self._run:
            return existing_kind == _SOC_CONNECT or existing_kind in _DEVICE_MUTATIONS
        if existing_kind == _SOC_CONNECT:
            return requested == _SOC_CONNECT
        if existing_kind in _DEVICE_MUTATIONS:
            return requested in _DEVICE_MUTATIONS and (
                requested_resource is None or existing.resource_id == requested_resource
            )
        return False
