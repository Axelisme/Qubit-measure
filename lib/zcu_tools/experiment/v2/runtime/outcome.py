"""Completion-state values and error descriptions for host runtime scopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

RunStatus: TypeAlias = Literal["completed", "stopped", "interrupted", "failed"]


@dataclass(frozen=True)
class ScheduleOutcome:
    """Immutable completion state and original cause for a Schedule run.

    status is completed, stopped, interrupted, or failed. reason is None for
    completion, otherwise a human-readable explanation of that status.
    exception is the original captured cause, or None when none was observed.
    A stopped outcome may retain an acquire/build Exception caught after stop
    was requested; its status/reason and StopSignal.error remain cancellation,
    and no retry follows. The first such cause is retained even if the schedule
    was already marked stopped. Pure cancellation, including the producer's
    first-round stopped-partial signal, has no exception. interrupted retains
    KeyboardInterrupt; failed retains the exhausted failure cause.
    """

    status: RunStatus = "completed"
    reason: str | None = None
    exception: BaseException | None = None

    @property
    def is_partial(self) -> bool:
        """Return whether status differs from completed, regardless of buffer size."""
        return self.status != "completed"


def exception_reason(exc: BaseException) -> str:
    """Return exc's class name, appending its nonempty string message.

    Errors raised by exc.__str__ propagate unchanged.
    """
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__
