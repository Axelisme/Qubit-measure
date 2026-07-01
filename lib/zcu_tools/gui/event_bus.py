"""Shared EventBus — lightweight publish/subscribe for GUI app internal events.

All emits and subscribes happen on the main thread. No Qt dependency.

Each payload carries its own event tag (``EVENT`` ClassVar) and
``subscribe`` / ``emit`` are parameterised by the payload *type* — the payload
type alone determines the event, so a payload can never be paired with the
wrong event. Apps subclass :class:`BasePayload` with their own event enum (each
narrowing ``EVENT`` to that concrete enum) and alias :class:`BaseEventBus`.

This module owns only the *mechanism*; it knows nothing of any concrete event,
payload, or app. All four GUI apps — main, fluxdep, dispersive, autofluxdep —
build on this base, each supplying its own event enum + payloads and aliasing
``EventBus = BaseEventBus``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Generic, TypeVar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BasePayload:
    """Base for all EventBus payloads.

    App-level subclasses narrow ``EVENT`` to their concrete event enum
    (e.g. ``EVENT: ClassVar[FluxDepEvent]``).
    """

    EVENT: ClassVar[Enum]


P = TypeVar("P", bound=BasePayload)


@dataclass
class EventSubscription(Generic[P]):
    """Handle returned by ``BaseEventBus.subscribe`` for lifecycle cleanup."""

    _bus: BaseEventBus
    _payload_type: type[P]
    _cb: Callable[[P], None]
    _active: bool = True

    def unsubscribe(self) -> None:
        """Remove this subscription once; safe to call repeatedly."""
        if not self._active:
            return
        self._active = False
        self._bus.unsubscribe(self._payload_type, self._cb)


class EventSubscriptions:
    """Small owner-side group for EventBus subscription cleanup."""

    def __init__(self) -> None:
        self._handles: list[EventSubscription[Any]] = []

    def add(self, handle: EventSubscription[P]) -> EventSubscription[P]:
        self._handles.append(handle)
        return handle

    def subscribe(
        self, bus: BaseEventBus, payload_type: type[P], cb: Callable[[P], None]
    ) -> EventSubscription[P]:
        return self.add(bus.subscribe(payload_type, cb))

    def unsubscribe_all(self, *_args: object) -> None:
        """Unsubscribe every handle; accepts Qt signal arguments."""
        handles = self._handles
        self._handles = []
        for handle in reversed(handles):
            handle.unsubscribe()

    def __len__(self) -> int:
        return len(self._handles)


class BaseEventBus:
    """Main-thread publish/subscribe keyed by payload type.

    Subscribing to a payload *type* (not the event enum) makes the callback's
    payload type infer automatically — ``subscribe(SomePayload, cb)`` gives
    ``cb`` a ``SomePayload`` argument with no overloads. The app event enum is
    retained for wire serialisation (``payload.EVENT``).
    """

    def __init__(self) -> None:
        self._subs: dict[type[BasePayload], list[Callable[..., None]]] = {}

    def subscribe(
        self, payload_type: type[P], cb: Callable[[P], None]
    ) -> EventSubscription[P]:
        self._subs.setdefault(payload_type, []).append(cb)
        return EventSubscription(self, payload_type, cb)

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
