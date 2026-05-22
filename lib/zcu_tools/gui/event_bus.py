"""EventBus — lightweight publish/subscribe for GUI-internal events.

All emits and subscribes happen on the main thread.  No Qt dependency.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class GuiEvent(str, Enum):
    """Supported event names for EventBus."""

    # Data layer
    MD_CHANGED = "md_changed"  # MetaDict attribute was set or deleted
    CONTEXT_CHANGED = (
        "context_changed"  # active ExpContext switched (project load or switch)
    )

    # Controller / Tab layer
    TAB_ADDED = "tab_added"  # payload: (tab_id, adapter_name)
    TAB_CLOSED = "tab_closed"  # payload: (tab_id)
    TAB_CONTENT_CHANGED = "tab_content_changed"  # payload: (tab_id)
    RUN_STATE_CHANGED = "run_state_changed"  # global run state changed

    # UI / Panel layer
    PREDICTOR_CHANGED = "predictor_changed"  # predictor state or values changed
    INSPECT_CHANGED = "inspect_changed"  # metadata/library changed via inspect dialog


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[GuiEvent, list[Callable[..., None]]] = {}

    def subscribe(self, event: GuiEvent, cb: Callable[..., None]) -> None:
        if not isinstance(event, GuiEvent):
            raise TypeError(f"event must be a GuiEvent, got {type(event)}")
        self._subs.setdefault(event, []).append(cb)

    def unsubscribe(self, event: GuiEvent, cb: Callable[..., None]) -> None:
        lst = self._subs.get(event, [])
        try:
            lst.remove(cb)
        except ValueError:
            pass

    def emit(self, event: GuiEvent, *args: Any, **kwargs: Any) -> None:
        if not isinstance(event, GuiEvent):
            raise TypeError(f"event must be a GuiEvent, got {type(event)}")
        for cb in list(self._subs.get(event, [])):
            try:
                cb(*args, **kwargs)
            except Exception:
                logger.warning(
                    "EventBus: exception in subscriber for %r", event, exc_info=True
                )
