"""Cancellation and partial-result errors shared by experiment run callers."""

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
        self._lock = threading.Lock()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()

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
        error = ScheduleOutcomeError(status, reason, exception)
        with self._lock:
            if self._error is None:
                self._error = error
        self._event.set()

    def raise_if_error(self) -> None:
        error = self.error
        if error is None:
            return
        if error.exception is not None:
            raise error from error.exception
        raise error
