"""EventBus — lightweight publish/subscribe for dispersive-fit-gui internal events.

All emits and subscribes happen on the main thread. No Qt dependency.

Each payload carries its own event tag (``EVENT`` ClassVar) and ``subscribe`` /
``emit`` are parameterised by the payload type — the payload type alone
determines the event, so a payload can never be paired with the wrong event.
(Mechanism copied from fluxdep-gui's EventBus.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, ClassVar, TypeVar

logger = logging.getLogger(__name__)


class DispersiveEvent(str, Enum):
    """Internal event identifiers for the dispersive analysis pipeline."""

    PROJECT_CHANGED = "project_changed"  # project info (chip/qub/paths) changed
    FIT_INPUTS_LOADED = "fit_inputs_loaded"  # fluxdep_fit read in from params.json
    ONETONE_LOADED = "onetone_loaded"  # a one-tone spectrum was loaded
    PREPROCESS_CHANGED = "preprocess_changed"  # the preprocessing result changed
    DISP_FIT_CHANGED = (
        "disp_fit_changed"  # the g/bare_rf tuning inputs or result changed
    )


@dataclass(frozen=True)
class Payload:
    """Base for all EventBus payloads. Subclasses set ``EVENT``."""

    EVENT: ClassVar[DispersiveEvent]


@dataclass(frozen=True)
class ProjectChangedPayload(Payload):
    EVENT: ClassVar[DispersiveEvent] = DispersiveEvent.PROJECT_CHANGED


@dataclass(frozen=True)
class FitInputsLoadedPayload(Payload):
    EVENT: ClassVar[DispersiveEvent] = DispersiveEvent.FIT_INPUTS_LOADED
    has_inputs: bool


@dataclass(frozen=True)
class OnetoneLoadedPayload(Payload):
    EVENT: ClassVar[DispersiveEvent] = DispersiveEvent.ONETONE_LOADED
    name: str


@dataclass(frozen=True)
class PreprocessChangedPayload(Payload):
    EVENT: ClassVar[DispersiveEvent] = DispersiveEvent.PREPROCESS_CHANGED


@dataclass(frozen=True)
class DispFitChangedPayload(Payload):
    EVENT: ClassVar[DispersiveEvent] = DispersiveEvent.DISP_FIT_CHANGED
    has_result: bool = False


P = TypeVar("P", bound=Payload)


class EventBus:
    """Main-thread publish/subscribe keyed by payload type.

    Subscribing to a payload *type* (not the event enum) makes the callback's
    payload type infer automatically. The DispersiveEvent enum is retained for
    wire serialisation (``payload.EVENT``).
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
