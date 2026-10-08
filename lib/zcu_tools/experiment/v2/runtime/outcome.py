"""Completion-state values and error descriptions for host runtime scopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

RunStatus: TypeAlias = Literal["completed", "stopped", "interrupted", "failed"]


@dataclass(frozen=True)
class ScheduleOutcome:
    """Describe a completed or partial Schedule run.

    status is completed, stopped, interrupted, or failed. reason is a readable
    explanation, or None when no reason was recorded. exception retains the
    interruption/failure cause, or None when no exception was recorded. A stopped
    outcome may retain the first acquire/build exception observed after stop;
    status and StopSignal.error still represent cancellation, without retry.
    Pure cancellation, including first-round stopped-partial acquisition, has
    no exception.
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
