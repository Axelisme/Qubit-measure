"""Shared EventBus — lightweight publish/subscribe for GUI app internal events.

All emits and subscribes happen on the main thread. No Qt dependency.

Each payload carries its own event tag (``EVENT`` ClassVar) and
``subscribe`` / ``emit`` are parameterised by the payload *type* — the payload
type alone determines the event, so a payload can never be paired with the
wrong event. Apps subclass :class:`BasePayload` with their own event enum (each
narrowing ``EVENT`` to that concrete enum) and alias :class:`BaseEventBus`.

This module owns only the *mechanism*; it knows nothing of any concrete event,
payload, or app. main-gui and autofluxdep-gui deliberately use different
schemes (``@overload`` keyed on an event enum / a single ``Event`` wrapper) and
do not build on this base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, ClassVar, TypeVar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BasePayload:
    """Base for all EventBus payloads.

    App-level subclasses narrow ``EVENT`` to their concrete event enum
    (e.g. ``EVENT: ClassVar[FluxDepEvent]``).
    """

    EVENT: ClassVar[Enum]


P = TypeVar("P", bound=BasePayload)


class BaseEventBus:
    """Main-thread publish/subscribe keyed by payload type.

    Subscribing to a payload *type* (not the event enum) makes the callback's
    payload type infer automatically — ``subscribe(SomePayload, cb)`` gives
    ``cb`` a ``SomePayload`` argument with no overloads. The app event enum is
    retained for wire serialisation (``payload.EVENT``).
    """

    def __init__(self) -> None:
        self._subs: dict[type[BasePayload], list[Callable[..., None]]] = {}

    def subscribe(self, payload_type: type[P], cb: Callable[[P], None]) -> None:
        self._subs.setdefault(payload_type, []).append(cb)

    def unsubscribe(self, payload_type: type[P], cb: Callable[[P], None]) -> None:
        subs = self._subs.get(payload_type)
        if subs and cb in subs:
            subs.remove(cb)

    def emit(self, payload: BasePayload) -> None:
        """Dispatch ``payload`` to every subscriber of its concrete type.

        A subscriber raising does not stop the others (logged).
        """
        for cb in list(self._subs.get(type(payload), ())):
            try:
                cb(payload)
            except Exception:  # noqa: BLE001 — one bad subscriber must not break the rest
                logger.exception(
                    "EventBus subscriber for %s raised", type(payload).__name__
                )
