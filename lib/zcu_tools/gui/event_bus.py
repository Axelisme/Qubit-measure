"""EventBus — lightweight publish/subscribe for GUI-internal events.

All emits and subscribes happen on the main thread.  No Qt dependency.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Supported event names (informational; not enforced at runtime).
# "md_changed"       — MetaDict attribute was set or deleted
# "context_changed"  — active ExpContext switched (use_context / new_context)
# "run_state_changed"— run started or finished


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[], None]]] = {}

    def subscribe(self, event: str, cb: Callable[[], None]) -> None:
        self._subs.setdefault(event, []).append(cb)

    def unsubscribe(self, event: str, cb: Callable[[], None]) -> None:
        lst = self._subs.get(event, [])
        try:
            lst.remove(cb)
        except ValueError:
            pass

    def emit(self, event: str) -> None:
        for cb in list(self._subs.get(event, [])):
            try:
                cb()
            except Exception:
                logger.warning(
                    "EventBus: exception in subscriber for %r", event, exc_info=True
                )
