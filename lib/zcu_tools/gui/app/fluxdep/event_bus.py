"""EventBus — fluxdep-gui internal events.

The publish/subscribe mechanism lives in :mod:`zcu_tools.gui.event_bus`; this
module supplies the fluxdep event enum and payloads and re-exports the shared
``EventBus``. All emits and subscribes happen on the main thread (no Qt
dependency). Each payload carries its own event tag (``EVENT`` ClassVar) and the
payload type alone determines the event, so a payload can never be paired with
the wrong event.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from zcu_tools.gui.event_bus import BaseEventBus, BasePayload


class FluxDepEvent(str, Enum):
    """Internal event identifiers for the fluxdep analysis pipeline."""

    SPECTRUM_ADDED = "spectrum_added"  # a spectrum was loaded into the collection
    SPECTRUM_REMOVED = "spectrum_removed"  # a spectrum was removed
    SPECTRUM_CHANGED = "spectrum_changed"  # a spectrum's alignment/points changed
    ACTIVE_SPECTRUM_CHANGED = "active_spectrum_changed"  # the active spectrum switched
    SELECTION_CHANGED = "selection_changed"  # cross-spectrum selection mask changed
    PROJECT_CHANGED = "project_changed"  # the project info (chip/qub/paths) changed
    FIT_CHANGED = "fit_changed"  # the database-search fit inputs or result changed


@dataclass(frozen=True)
class Payload(BasePayload):
    """Base for all fluxdep EventBus payloads. Subclasses set ``EVENT``."""

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


@dataclass(frozen=True)
class FitChangedPayload(Payload):
    EVENT: ClassVar[FluxDepEvent] = FluxDepEvent.FIT_CHANGED
    has_result: bool = False


EventBus = BaseEventBus
