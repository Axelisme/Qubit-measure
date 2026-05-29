from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

# Upper bound on how many released-token completion Events are retained so that
# ``await_release`` can return immediately for an operation that finished before
# the caller began waiting. LRU-evicted past this — eviction only degrades a
# late waiter to "treat as already-done" (still correct, just non-blocking).
_DONE_EVENT_LIMIT = 32


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
        # Per-token completion Event: created on acquire, set on release. A
        # blocking off-main handler (e.g. device.wait_setup) can wait on it
        # thread-safely without touching any main-thread-owned state.
        self._events: dict[int, threading.Event] = {}
        # Released tokens' already-set Events, retained briefly (LRU) so a
        # caller that begins awaiting after release still returns immediately.
        self._done: OrderedDict[int, threading.Event] = OrderedDict()

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
        self._events[candidate.token] = threading.Event()
        return candidate

    def release(self, lease: OperationLease) -> None:
        active = self._active.get(lease.token)
        if active != lease:
            raise RuntimeError(f"Operation lease {lease.token} is not active")
        del self._active[lease.token]
        evt = self._events.pop(lease.token, None)
        if evt is not None:
            evt.set()
            self._done[lease.token] = evt
            while len(self._done) > _DONE_EVENT_LIMIT:
                self._done.popitem(last=False)

    def await_release(self, token: int, timeout: float) -> bool:
        """Block until the operation holding ``token`` is released.

        Thread-safe; intended for off-main blocking handlers. Returns True once
        released (or if already released / unknown — a token with no live or
        retained Event is treated as already-done, so callers never hang on an
        operation that finished before they began waiting). Returns False on
        timeout while the operation is still active.
        """
        evt = self._events.get(token) or self._done.get(token)
        if evt is None:
            return True
        return evt.wait(timeout=timeout)

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
