"""Run-lifecycle event definitions for autofluxdep-gui.

The run domain owns the ``RunEvent`` enum and all run-lifecycle payloads.
Note: these are DIFFERENT from app/main's run payloads — autofluxdep run
payloads carry no ``tab_id``; keep them separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from zcu_tools.gui.event_bus import BasePayload


class RunEvent(str, Enum):
    """Run-lifecycle event names for autofluxdep (wire names are the enum values)."""

    RUN_STARTED = "run_started"
    NODE_ENTERED = "node_entered"
    POINT_DONE = "point_done"
    RUN_PAUSE_REQUESTED = "run_pause_requested"
    RUN_PAUSED = "run_paused"
    RUN_CONTINUED = "run_continued"
    RUN_FINISHED = "run_finished"
    RUN_STOPPED = "run_stopped"
    RUN_FAILED = "run_failed"


@dataclass(frozen=True)
class _RunPayload(BasePayload):
    """Base for run-domain EventBus payloads. Subclasses set ``EVENT``."""

    EVENT: ClassVar[RunEvent]


@dataclass(frozen=True)
class RunStartedPayload(_RunPayload):
    """Payload for RUN_STARTED."""

    EVENT: ClassVar[RunEvent] = RunEvent.RUN_STARTED


@dataclass(frozen=True)
class NodeEnteredPayload(_RunPayload):
    """Payload for NODE_ENTERED: a provider started running (auto-follow)."""

    EVENT: ClassVar[RunEvent] = RunEvent.NODE_ENTERED
    name: str
    idx: int


@dataclass(frozen=True)
class PointDonePayload(_RunPayload):
    """Payload for POINT_DONE: a flux point completed (advance global progress)."""

    EVENT: ClassVar[RunEvent] = RunEvent.POINT_DONE
    idx: int


@dataclass(frozen=True)
class RunPauseRequestedPayload(_RunPayload):
    """Payload for RUN_PAUSE_REQUESTED."""

    EVENT: ClassVar[RunEvent] = RunEvent.RUN_PAUSE_REQUESTED


@dataclass(frozen=True)
class RunPausedPayload(_RunPayload):
    """Payload for RUN_PAUSED: boundary pause after a committed flux point."""

    EVENT: ClassVar[RunEvent] = RunEvent.RUN_PAUSED
    next_flux_idx: int


@dataclass(frozen=True)
class RunContinuedPayload(_RunPayload):
    """Payload for RUN_CONTINUED: a paused session started a new segment."""

    EVENT: ClassVar[RunEvent] = RunEvent.RUN_CONTINUED
    next_flux_idx: int


@dataclass(frozen=True)
class RunFinishedPayload(_RunPayload):
    """Payload for RUN_FINISHED."""

    EVENT: ClassVar[RunEvent] = RunEvent.RUN_FINISHED


@dataclass(frozen=True)
class RunStoppedPayload(_RunPayload):
    """Payload for RUN_STOPPED."""

    EVENT: ClassVar[RunEvent] = RunEvent.RUN_STOPPED


@dataclass(frozen=True)
class RunFailedPayload(_RunPayload):
    """Payload for RUN_FAILED: a Node's ``produce`` raised mid-sweep.

    Carries the human-readable error so the UI can surface it. A failed run is a
    terminal state distinct from a cooperative stop: the sweep aborted on an
    unexpected error (e.g. an unconfigured Node Fast-Failing), not the user's
    Stop. The run still unlocks the UI exactly like a stop/finish."""

    EVENT: ClassVar[RunEvent] = RunEvent.RUN_FAILED
    message: str
