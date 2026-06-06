"""EventBus — autofluxdep-gui internal events.

The publish/subscribe mechanism lives in :mod:`zcu_tools.gui.event_bus`; this
module supplies the autofluxdep event enum and payloads and re-exports the shared
``EventBus``. All emits and subscribes happen on the main thread (no Qt
dependency). Each payload carries its own event tag (``EVENT`` ClassVar) and the
payload type alone determines the event, so a payload can never be paired with
the wrong event.

Workflow-editing events drive the node list; setup/run events drive the
edit↔run state switch and the progress display.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from zcu_tools.gui.event_bus import BaseEventBus, BasePayload


class EventType(str, Enum):
    """Internal event identifiers for the autofluxdep workflow + run pipeline."""

    WORKFLOW_CHANGED = "workflow_changed"  # nodes added/removed/reordered/renamed
    FLUX_CHANGED = "flux_changed"
    SETUP_DONE = "setup_done"  # resources built → Run enabled
    # run lifecycle (drive edit↔run UI + progress)
    RUN_STARTED = "run_started"
    NODE_ENTERED = "node_entered"  # a provider started running — auto-follow
    POINT_DONE = "point_done"  # a flux point completed — advance global progress
    RUN_FINISHED = "run_finished"
    RUN_STOPPED = "run_stopped"


@dataclass(frozen=True)
class Payload(BasePayload):
    """Base for all autofluxdep EventBus payloads. Subclasses set ``EVENT``."""

    EVENT: ClassVar[EventType]


@dataclass(frozen=True)
class WorkflowChangedPayload(Payload):
    EVENT: ClassVar[EventType] = EventType.WORKFLOW_CHANGED
    name: str | None = None  # the affected node, or None for whole-list edits


@dataclass(frozen=True)
class FluxChangedPayload(Payload):
    EVENT: ClassVar[EventType] = EventType.FLUX_CHANGED
    count: int  # number of flux points now set


@dataclass(frozen=True)
class SetupDonePayload(Payload):
    EVENT: ClassVar[EventType] = EventType.SETUP_DONE


@dataclass(frozen=True)
class RunStartedPayload(Payload):
    EVENT: ClassVar[EventType] = EventType.RUN_STARTED


@dataclass(frozen=True)
class NodeEnteredPayload(Payload):
    EVENT: ClassVar[EventType] = EventType.NODE_ENTERED
    name: str
    idx: int


@dataclass(frozen=True)
class PointDonePayload(Payload):
    EVENT: ClassVar[EventType] = EventType.POINT_DONE
    idx: int


@dataclass(frozen=True)
class RunFinishedPayload(Payload):
    EVENT: ClassVar[EventType] = EventType.RUN_FINISHED


@dataclass(frozen=True)
class RunStoppedPayload(Payload):
    EVENT: ClassVar[EventType] = EventType.RUN_STOPPED


EventBus = BaseEventBus
