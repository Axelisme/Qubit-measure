"""Shared hardware mutual-exclusion gate for measurement-session apps."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from zcu_tools.gui.event_bus import BaseEventBus, OriginKind
from zcu_tools.gui.session.events import GateChangedPayload, GatePresence
from zcu_tools.gui.session.ports import (
    ExclusionGate,
    OperationConflictError,
)
from zcu_tools.gui.session.ports import (
    OperationKind as SessionOpKind,
)

__all__ = ["GatePresence", "RunBlocksHardwareGate"]


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
    origin_kind: OriginKind
    note: str
    since: float
    resource_id: str | None = None


class RunBlocksHardwareGate(ExclusionGate):
    """Mutual exclusion among RUN, SoC connect, and device mutations.

    Measurement-session apps own their app-local RUN enum value, but the hardware
    policy is shared: RUN blocks RUN, SoC connect, and any device mutation;
    device mutations conflict only with mutations of the same device.
    """

    def __init__(
        self,
        *,
        run_kind: str,
        bus: BaseEventBus,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._run = _norm(run_kind)
        self._bus = bus
        self._clock = clock
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
        origin_kind: OriginKind,
        note: str,
        resource_id: str | None = None,
    ) -> None:
        if not owner_id:
            raise ValueError("owner_id must not be empty")
        if not note.strip():
            raise ValueError("gate presence note must not be blank")
        self._active[token] = _ActiveLease(
            kind=_norm(kind),
            owner_id=owner_id,
            origin_kind=origin_kind,
            note=note,
            since=self._clock(),
            resource_id=resource_id,
        )
        self._emit_changed()

    def release(self, token: int) -> None:
        if token not in self._active:
            raise RuntimeError(f"Operation lease {token} is not active")
        del self._active[token]
        self._emit_changed()

    def snapshot(self) -> tuple[GatePresence, ...]:
        """Return a read-only presence projection without raw monotonic epochs."""

        now = self._clock()
        return tuple(
            GatePresence(
                kind=lease.kind,
                origin_kind=lease.origin_kind,
                note=lease.note,
                active_for_seconds=max(0.0, now - lease.since),
            )
            for lease in self._active.values()
        )

    def _emit_changed(self) -> None:
        self._bus.emit(GateChangedPayload(active=self.snapshot()))

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
