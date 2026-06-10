"""Run-lifecycle event definitions for measure-gui.

The run domain owns the ``RunEvent`` enum and all run-lifecycle payloads.
Import from this module (``events.run``) rather than the old ``event_bus``
flat module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional

from zcu_tools.gui.event_bus import BasePayload


class RunEvent(str, Enum):
    """Run-lifecycle event names (wire names are the enum values)."""

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"


@dataclass(frozen=True)
class _RunPayload(BasePayload):
    """Base for all run-domain EventBus payloads. Subclasses set ``EVENT``."""

    EVENT: ClassVar[RunEvent]


@dataclass(frozen=True)
class RunStartedPayload(_RunPayload):
    """Payload for RUN_STARTED: a run began on ``tab_id`` (the run lock is now
    held by it)."""

    EVENT: ClassVar[RunEvent] = RunEvent.RUN_STARTED
    tab_id: str


@dataclass(frozen=True)
class RunFinishedPayload(_RunPayload):
    """Payload for RUN_FINISHED: the run on ``tab_id`` reached a terminal state
    (the run lock is released). ``outcome`` distinguishes success / failure /
    cancellation; ``error_message`` is set only on failure."""

    EVENT: ClassVar[RunEvent] = RunEvent.RUN_FINISHED
    tab_id: str
    outcome: str  # 'finished' | 'failed' | 'cancelled'
    error_message: str | None = None
