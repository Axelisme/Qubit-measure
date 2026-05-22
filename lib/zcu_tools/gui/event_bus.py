"""EventBus — lightweight publish/subscribe for GUI-internal events.

All emits and subscribes happen on the main thread.  No Qt dependency.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class GuiEvent(str, Enum):
    """Supported event names for EventBus."""

    MD_CHANGED = "md_changed"  # MetaDict attribute was set or deleted
    CONTEXT_CHANGED = "context_changed"  # active ExpContext switched
    RUN_STATE_CHANGED = "run_state_changed"  # run started or finished


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[GuiEvent, list[Callable[[], None]]] = {}

    def subscribe(self, event: GuiEvent, cb: Callable[[], None]) -> None:
        if not isinstance(event, GuiEvent):
            raise TypeError(f"event must be a GuiEvent, got {type(event)}")
        self._subs.setdefault(event, []).append(cb)

    def unsubscribe(self, event: GuiEvent, cb: Callable[[], None]) -> None:
        lst = self._subs.get(event, [])
        try:
            lst.remove(cb)
        except ValueError:
            pass

    def emit(self, event: GuiEvent) -> None:
        if not isinstance(event, GuiEvent):
            raise TypeError(f"event must be a GuiEvent, got {type(event)}")
        for cb in list(self._subs.get(event, [])):
            try:
                cb()
            except Exception:
                logger.warning(
                    "EventBus: exception in subscriber for %r", event, exc_info=True
                )
