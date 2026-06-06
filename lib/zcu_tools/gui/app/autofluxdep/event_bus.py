"""EventBus for autofluxdep-gui.

Same typed enum + subscribe/emit pattern as fluxdep/dispersive (ADR-0013).
Workflow-editing events drive the node list; setup/run events drive the
edit↔run state switch and the progress display.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing_extensions import Any, Callable


class EventType(str, Enum):
    PROJECT_CHANGED = "project_changed"
    WORKFLOW_CHANGED = "workflow_changed"  # nodes added/removed/reordered
    FLUX_CHANGED = "flux_changed"
    SETUP_DONE = "setup_done"  # resources built → Run enabled
    # run lifecycle (drive edit↔run UI + progress)
    RUN_STARTED = "run_started"
    NODE_ENTERED = "node_entered"  # payload: (node_name, flux_idx) — auto-follow
    POINT_DONE = "point_done"  # payload: flux_idx (advance global progress)
    RUN_FINISHED = "run_finished"
    RUN_STOPPED = "run_stopped"


@dataclass(frozen=True)
class Event:
    type: EventType
    payload: Any = None


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[EventType, list[Callable[[Event], None]]] = {}

    def subscribe(self, etype: EventType, handler: Callable[[Event], None]) -> None:
        self._subs.setdefault(etype, []).append(handler)

    def emit(self, event: Event) -> None:
        for handler in self._subs.get(event.type, []):
            handler(event)
