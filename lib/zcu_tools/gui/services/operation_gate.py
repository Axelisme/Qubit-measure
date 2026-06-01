from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

# Upper bound on how many settled operations are retained so that
# ``await_outcome`` can return immediately for an operation that finished before
# the caller began waiting. LRU-evicted past this — eviction only degrades a
# late waiter to "treat as already-done" (still correct, just non-blocking and
# losing the recorded outcome, which then reads as a default 'finished').
_DONE_EVENT_LIMIT = 32


class OperationKind(Enum):
    RUN = "run"
    SOC_CONNECT = "soc_connect"
    DEVICE_CONNECT = "device_connect"
    DEVICE_DISCONNECT = "device_disconnect"
    DEVICE_SETUP = "device_setup"


class OperationConflictError(RuntimeError):
    """Raised when a hardware operation conflicts with an active operation."""


OperationStatus = Literal["pending", "finished", "failed", "cancelled"]


@dataclass(frozen=True)
class OperationOutcome:
    """The terminal result of an async operation, as a neutral value.

    Carries only success/failure/cancellation + an error message — never a
    result payload (run results / soc handles are read via their own snapshot /
    query paths, not through the operation handle). The gate never interprets
    ``status``; it is carried like ``owner_id``.
    """

    status: OperationStatus
    error: Optional[str] = None


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
    }
)


class _OperationExclusion:
    """Mutual exclusion among active hardware operations.

    Owns only the "can this start right now" concern: the set of active leases
    and the conflict rules. Released immediately when an operation completes so
    the hardware is freed for the next operation.
    """

    def __init__(self) -> None:
        self._active: dict[int, OperationLease] = {}

    def register(self, lease: OperationLease) -> None:
        """Add an active lease (the caller has already checked conflicts)."""
        self._active[lease.token] = lease

    def conflict_for(self, kind: OperationKind) -> Optional[OperationLease]:
        return next(
            (
                lease
                for lease in self._active.values()
                if self._conflicts(lease.kind, kind)
            ),
            None,
        )

    def remove(self, lease: OperationLease) -> None:
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


class _OperationRegistry:
    """Async-operation handles, keyed by token.

    Owns the "what is this operation's lifecycle" concern, exposing the three
    async verbs over a single token: ``await_outcome`` (off-main blocking wait),
    ``poll`` (non-blocking), and ``cancel`` (async stop request). The completion
    Event lets a blocking off-main handler (e.g. operation.await) wait thread-
    safely without touching main-thread-owned state. The optional ``stop_event``
    is the worker's own cancellation flag, passed in at ``create`` time: cancel
    sets it (a pure data handle, never a callback), and the worker self-
    translates "stopped" into a ``cancelled`` outcome. Settled tokens are
    retained briefly (LRU) so a late waiter still returns.
    """

    def __init__(self) -> None:
        # Live (pending) operations: token -> not-yet-set completion Event.
        self._events: dict[int, threading.Event] = {}
        # Live operations' worker stop_event (None when the operation has no
        # cancellation point, e.g. a blocking connect — cancel is a no-op there).
        self._stop_events: dict[int, Optional[threading.Event]] = {}
        # Settled operations, retained briefly (LRU) so a caller awaiting after
        # settle still returns the outcome immediately. The Event stays set.
        self._done: OrderedDict[int, tuple[threading.Event, OperationOutcome]] = (
            OrderedDict()
        )

    def create(self, token: int, stop_event: Optional[threading.Event]) -> None:
        self._events[token] = threading.Event()
        self._stop_events[token] = stop_event

    def settle(self, token: int, outcome: OperationOutcome) -> None:
        """Mark the operation terminal: store outcome, set Event, retain (LRU)."""
        evt = self._events.pop(token, None)
        self._stop_events.pop(token, None)
        if evt is None:
            return  # never created, or already settled
        evt.set()
        self._done[token] = (evt, outcome)
        while len(self._done) > _DONE_EVENT_LIMIT:
            self._done.popitem(last=False)

    def cancel(self, token: int) -> None:
        """Request the operation stop (set its stop_event); returns immediately.

        Async notification, not a wait: the caller polls/awaits for the actual
        terminal outcome. A no-op for an unknown/settled token or one with no
        stop_event (a connect has no cancellation point — it runs to completion
        and shutdown falls back to a timeout force-close).
        """
        stop_event = self._stop_events.get(token)
        if stop_event is not None:
            stop_event.set()

    def cancel_all(self) -> list[int]:
        """Cancel every live operation; return their tokens (for poll/await)."""
        tokens = list(self._events.keys())
        for token in tokens:
            self.cancel(token)
        return tokens

    def await_outcome(self, token: int, timeout: float) -> Optional[OperationOutcome]:
        """Block until the token settles; return its outcome.

        Thread-safe; for off-main blocking handlers. A token with no live or
        retained Event is treated as already-done (returns a default 'finished'
        outcome) so callers never hang on an operation that finished before they
        began waiting. Returns None only on timeout while still pending.
        """
        live = self._events.get(token)
        if live is not None:
            if not live.wait(timeout=timeout):
                return None
            # Woken by settle(), which moved the token into _done with its outcome.
            retained = self._done.get(token)
            return retained[1] if retained is not None else OperationOutcome("finished")
        retained = self._done.get(token)
        if retained is not None:
            return retained[1]
        # Unknown / evicted: treat as already finished.
        return OperationOutcome("finished")

    def poll(self, token: int) -> Optional[OperationOutcome]:
        """Non-blocking: outcome if settled (or unknown), None if still pending."""
        if token in self._events:
            return None
        retained = self._done.get(token)
        if retained is not None:
            return retained[1]
        return OperationOutcome("finished")


class OperationGate:
    """Facade combining hardware exclusion and async-operation handles.

    A single token identifies an operation across both concerns: ``acquire``
    mints it and registers the lease (exclusion) + the handle (registry);
    ``release`` removes the lease immediately (frees hardware) and settles the
    handle (retained briefly for late awaiters). Exclusion and handle lifecycle
    are deliberately separate objects — this facade is the only place they meet.
    """

    def __init__(self) -> None:
        self._next_token = 1
        self._exclusion = _OperationExclusion()
        self._registry = _OperationRegistry()

    def acquire(
        self,
        kind: OperationKind,
        *,
        owner_id: str,
        resource_id: str | None = None,
        stop_event: Optional[threading.Event] = None,
    ) -> OperationLease:
        if not owner_id:
            raise ValueError("owner_id must not be empty")
        conflicting = self._exclusion.conflict_for(kind)
        if conflicting is not None:
            raise OperationConflictError(
                f"Cannot start {kind.value}: {conflicting.kind.value} is active "
                f"for {conflicting.owner_id!r}"
            )
        lease = OperationLease(
            token=self._next_token,
            kind=kind,
            owner_id=owner_id,
            resource_id=resource_id,
        )
        self._next_token += 1
        self._exclusion.register(lease)
        # The stop_event is the worker's own cancellation flag (or None when the
        # operation has no cancellation point); the registry holds it so cancel()
        # can set it without owning a callback into the worker.
        self._registry.create(lease.token, stop_event)
        return lease

    def release(self, lease: OperationLease, outcome: OperationOutcome) -> None:
        """Free the hardware (exclusion) and settle the handle (registry)."""
        self._exclusion.remove(lease)
        self._registry.settle(lease.token, outcome)

    def await_outcome(self, token: int, timeout: float) -> Optional[OperationOutcome]:
        """Block until ``token`` settles; outcome, or None on timeout."""
        return self._registry.await_outcome(token, timeout)

    def poll(self, token: int) -> Optional[OperationOutcome]:
        """Non-blocking outcome (None while still pending)."""
        return self._registry.poll(token)

    def cancel(self, token: int) -> None:
        """Request the operation stop (set its stop_event); returns immediately."""
        self._registry.cancel(token)

    def cancel_all(self) -> list[int]:
        """Cancel every live operation; return their tokens (for poll/await)."""
        return self._registry.cancel_all()

    def has_active(self, kind: OperationKind) -> bool:
        return self._exclusion.has_active(kind)

    def is_device_mutating(self, name: str) -> bool:
        return self._exclusion.is_device_mutating(name)
