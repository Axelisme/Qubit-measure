"""EventBus — lightweight publish/subscribe for fluxdep-gui internal events.

All emits and subscribes happen on the main thread. No Qt dependency.

Mechanism copied in spirit from measure-gui's EventBus, but with a leaner typing
scheme: instead of N×3 per-event overloads, each payload carries its own event
tag (``EVENT`` ClassVar) and ``subscribe``/``emit`` are parameterised by the
payload type. The payload type alone determines the event — there is no way to
pair a payload with the wrong event.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, ClassVar, TypeVar

logger = logging.getLogger(__name__)


class FluxDepEvent(str, Enum):
    """Internal event identifiers for the fluxdep analysis pipeline."""

    SPECTRUM_ADDED = "spectrum_added"  # a spectrum was loaded into the collection
    SPECTRUM_REMOVED = "spectrum_removed"  # a spectrum was removed
    SPECTRUM_CHANGED = "spectrum_changed"  # a spectrum's alignment/points changed
    ACTIVE_SPECTRUM_CHANGED = "active_spectrum_changed"  # the active spectrum switched
    SELECTION_CHANGED = "selection_changed"  # cross-spectrum selection mask changed
    PROJECT_CHANGED = "project_changed"  # the project info (chip/qub/paths) changed


@dataclass(frozen=True)
class Payload:
    """Base for all EventBus payloads. Subclasses set ``EVENT``."""

    EVENT: ClassVar[FluxDepEvent]


@dataclass(frozen=True)
class SpectrumAddedPayload(Payload):
    EVENT: ClassVar[FluxDepEvent] = FluxDepEvent.SPECTRUM_ADDED
    name: str


@dataclass(frozen=True)
class SpectrumRemovedPayload(Payload):
    EVENT: ClassVar[FluxDepEvent] = FluxDepEvent.SPECTRUM_REMOVED
    name: str


@dataclass(frozen=True)
class SpectrumChangedPayload(Payload):
    EVENT: ClassVar[FluxDepEvent] = FluxDepEvent.SPECTRUM_CHANGED
    name: str


@dataclass(frozen=True)
class ActiveSpectrumChangedPayload(Payload):
    EVENT: ClassVar[FluxDepEvent] = FluxDepEvent.ACTIVE_SPECTRUM_CHANGED
    name: str | None


@dataclass(frozen=True)
class SelectionChangedPayload(Payload):
    EVENT: ClassVar[FluxDepEvent] = FluxDepEvent.SELECTION_CHANGED


@dataclass(frozen=True)
class ProjectChangedPayload(Payload):
    EVENT: ClassVar[FluxDepEvent] = FluxDepEvent.PROJECT_CHANGED


P = TypeVar("P", bound=Payload)


class EventBus:
    """Main-thread publish/subscribe keyed by payload type.

    Subscribing to a payload *type* (not the event enum) makes the callback's
    payload type infer automatically — ``subscribe(SpectrumAddedPayload, cb)``
    gives ``cb`` a ``SpectrumAddedPayload`` argument with no overloads. The
    FluxDepEvent enum is retained for wire serialisation (``payload.EVENT``).
    """

    def __init__(self) -> None:
        self._subs: dict[type[Payload], list[Callable[..., None]]] = {}

    def subscribe(self, payload_type: type[P], cb: Callable[[P], None]) -> None:
        self._subs.setdefault(payload_type, []).append(cb)

    def unsubscribe(self, payload_type: type[P], cb: Callable[[P], None]) -> None:
        subs = self._subs.get(payload_type)
        if subs and cb in subs:
            subs.remove(cb)

    def emit(self, payload: Payload) -> None:
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
