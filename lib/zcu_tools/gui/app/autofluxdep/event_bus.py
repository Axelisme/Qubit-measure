"""EventBus for autofluxdep-gui (skeleton).

Same typed enum + subscribe/emit pattern as fluxdep/dispersive (ADR-0013). Only
the workflow-editing events exist in the skeleton; run-progress events
(PointStarted, NodeCompleted, ...) are Phase B/C additions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing_extensions import Any, Callable


class EventType(str, Enum):
    PROJECT_CHANGED = "project_changed"
    WORKFLOW_CHANGED = "workflow_changed"  # nodes added/removed/reordered
    FLUX_CHANGED = "flux_changed"


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
