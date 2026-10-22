"""Cancellation and partial-result errors shared by experiment run callers."""

from __future__ import annotations

import threading
from typing import Literal, TypeAlias

ErrorStatus: TypeAlias = Literal["interrupted", "failed"]


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
    """Run-owned cancellation and first-error signal shared by all execution stages."""

    def __init__(self, event: threading.Event | None = None) -> None:
        self._event = event if event is not None else threading.Event()
        self._error: ScheduleOutcomeError | None = None
        self._lock = threading.RLock()
        self._parent: StopSignal | None = None
        self._children: set[StopSignal] = set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        """Set this scope and all attached descendants, never its parent."""
        with self._lock:
            self._event.set()
            for child in self._children:
                child.set()

    def child(self) -> StopSignal:
        """Attach a scope inheriting this scope's stop/error, including at birth.

        Stops travel down; errors stay on the scope that records them.
        Call release when the operation drains so completed work is not retained
        by the execution tree.
        """
        with self._lock:
            child = StopSignal()
            child._lock = self._lock
            child._parent = self
            child._error = self._error
            if self.is_set():
                child.set()
            self._children.add(child)
            return child

    def release(self) -> None:
        """Detach a completed scope; future parent stops no longer reach it."""
        with self._lock:
            if self._parent is not None:
                self._parent._children.discard(self)
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
