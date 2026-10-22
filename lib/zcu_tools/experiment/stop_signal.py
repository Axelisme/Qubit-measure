"""Cancellation and partial-result errors shared by experiment run callers."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

ErrorStatus: TypeAlias = Literal["interrupted", "failed"]


@dataclass(frozen=True)
class CancelResult:
    """Acknowledged scope request, not a promise that work has drained.

    cancelled means the owner accepted the request, not that work drained.
    Token owners atomically match/set the target. Workflow bare script Stop
    instead records and latches intent before signaling a still-live host;
    acceptance survives host exit before owner finalization. reason is scope
    cancellation requested, scope not active, sequence mismatch, or the recipe
    admission refusal no run in progress. ambiguous refuses multiple matches.
    active_path is the unique live path at the check, stopping at the first fork.
    candidates lists full matching paths only for an ambiguous refusal.
    """

    cancelled: bool
    reason: str
    active_path: str
    candidates: list[str] = field(default_factory=list)


class ScheduleOutcomeError(RuntimeError):
    """Exception wrapper for a Schedule failure that returned partial data."""

    def __init__(
        self,
        status: ErrorStatus,
        reason: str,
        exception: BaseException | None,
    ) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.exception = exception


class StopSignal:
    """Thread-safe scope tree separating cancellation from completion facts.

    A supplied event is borrowed as this token\'s stop bit; descendants own
    independent events. Scope owns names, selectors and birth numbers. Completed
    child facts travel upward on release without stopping the parent, unless
    the child was created with boundary=True.
    """

    def __init__(self, event: threading.Event | None = None) -> None:
        self._boundary = True
        self._completed_partial = False
        self._event = event if event is not None else threading.Event()
        self._error: ScheduleOutcomeError | None = None
        self._lock = threading.RLock()
        self._parent: StopSignal | None = None
        self._children: dict[StopSignal, None] = {}

    @property
    def partial_result(self) -> bool:
        """Read this token's stop bit or a released child's partial completion.

        Child completion does not set this token's event. Active child facts
        are delivered only when that child releases.
        """
        with self._lock:
            return self.is_set() or self._completed_partial

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        """Set this scope and all attached descendants, never its parent."""
        with self._lock:
            self._event.set()
            for child in self._children:
                child.set()

    def child(self, *, boundary: bool = True) -> StopSignal:
        """Attach a scope inheriting this scope's stop/error, including at birth.

        boundary=True prevents completion facts from reaching the parent,
        as required for operations. False delivers first error and partial facts
        on release. Stops travel down immediately. Call release after work drains.
        """
        with self._lock:
            child = StopSignal()
            child._boundary = boundary
            child._lock = self._lock
            child._parent = self
            child._error = self._error
            if self.is_set():
                child.set()
            self._children[child] = None
            return child

    def release(self) -> None:
        """Deliver completion facts once, then detach without stopping the parent.

        Preserve the parent's first error and accumulate partial completion.
        An operation's caller consumes its facts, so releasing an operation
        never hands them to its iteration. Repeated release is a no-op.
        Future parent stops cannot reach a detached scope.
        """
        with self._lock:
            parent = self._parent
            if parent is not None:
                if not self._boundary:
                    if parent._error is None:
                        parent._error = self._error
                    parent._completed_partial |= self.partial_result
                parent._children.pop(self)
                self._parent = None

    def clear_stop(self) -> None:
        with self._lock:
            self._error = None
        self._event.clear()

    @property
    def event(self) -> threading.Event:
        return self._event

    @property
    def error(self) -> ScheduleOutcomeError | None:
        with self._lock:
            return self._error

    def set_error(
        self,
        status: ErrorStatus,
        reason: str,
        exception: BaseException | None,
    ) -> None:
        """Record this scope's first error and stop it and its descendants.

        Ancestor and sibling stop/error state is unchanged. The stored error
        retains status, reason and exception for raise_if_error().
        """
        error = ScheduleOutcomeError(status, reason, exception)
        with self._lock:
            if self._error is None:
                self._error = error
            self.set()

    def raise_if_error(self) -> None:
        error = self.error
        if error is None:
            return
        if error.exception is not None:
            raise error from error.exception
        raise error
