from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OperationKind(Enum):
    RUN = "run"
    SOC_CONNECT = "soc_connect"
    DEVICE_CONNECT = "device_connect"
    DEVICE_DISCONNECT = "device_disconnect"
    DEVICE_SETUP = "device_setup"
    DEVICE_SET_VALUE = "device_set_value"


class OperationConflictError(RuntimeError):
    """Raised when a hardware operation conflicts with an active operation."""


@dataclass(frozen=True)
class OperationLease:
    token: int
    kind: OperationKind
    owner_id: str
    resource_id: str | None = None


_DEVICE_MUTATIONS = frozenset(
    {
        OperationKind.DEVICE_CONNECT,
        OperationKind.DEVICE_DISCONNECT,
        OperationKind.DEVICE_SETUP,
        OperationKind.DEVICE_SET_VALUE,
    }
)


class OperationGate:
    """Own hardware operation exclusion and lease lifetime validation."""

    def __init__(self) -> None:
        self._next_token = 1
        self._active: dict[int, OperationLease] = {}

    def acquire(
        self,
        kind: OperationKind,
        *,
        owner_id: str,
        resource_id: str | None = None,
    ) -> OperationLease:
        if not owner_id:
            raise ValueError("owner_id must not be empty")
        candidate = OperationLease(
            token=self._next_token,
            kind=kind,
            owner_id=owner_id,
            resource_id=resource_id,
        )
        conflicting = next(
            (
                lease
                for lease in self._active.values()
                if self._conflicts(lease.kind, candidate.kind)
            ),
            None,
        )
        if conflicting is not None:
            raise OperationConflictError(
                f"Cannot start {kind.value}: {conflicting.kind.value} is active "
                f"for {conflicting.owner_id!r}"
            )
        self._next_token += 1
        self._active[candidate.token] = candidate
        return candidate

    def release(self, lease: OperationLease) -> None:
        active = self._active.get(lease.token)
        if active != lease:
            raise RuntimeError(f"Operation lease {lease.token} is not active")
        del self._active[lease.token]

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
