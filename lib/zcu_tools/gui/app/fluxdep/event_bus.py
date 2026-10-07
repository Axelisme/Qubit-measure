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
from typing import ClassVar, Literal

from zcu_tools.gui.event_bus import BaseEventBus, BasePayload


class FluxDepEvent(str, Enum):
    """Internal event identifiers for the fluxdep analysis pipeline."""

    SPECTRUM_ADDED = "spectrum_added"  # a spectrum was loaded into the collection
    SPECTRUM_REMOVED = "spectrum_removed"  # a spectrum was removed
    SPECTRUM_CHANGED = "spectrum_changed"  # a spectrum's alignment/points changed
    ACTIVE_SPECTRUM_CHANGED = "active_spectrum_changed"  # the active spectrum switched
    SELECTION_CHANGED = "selection_changed"  # cross-spectrum selection mask changed
    PROJECT_CHANGED = "project_changed"  # the project info (chip/qub/paths) changed
    SEARCH_CHANGED = "search_changed"  # app-owned search activity changed
    FIT_CHANGED = "fit_changed"  # the database-search fit inputs or result changed
    INTERACTIVE_CHANGED = "interactive_changed"  # a live interactive context changed


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


@dataclass(frozen=True)
class SearchChangedPayload(Payload):
    """Search handle projection: token is its id; status is pending or terminal;
    error is the failure reason or None. Numeric results stay with the owner.
    """

    EVENT: ClassVar[FluxDepEvent] = FluxDepEvent.SEARCH_CHANGED
    token: int
    status: Literal["pending", "finished", "failed", "cancelled"]
    error: str | None = None


InteractiveKind = Literal["line", "onetone", "twotone", "selection"]


@dataclass(frozen=True)
class InteractiveChangedPayload(Payload):
    """Committed interactive lifecycle fact, independent of presentation.

    context_id is the positive owner-lifetime identity, never an edit revision.
    kind identifies the domain picker; spectrum_name is None for joint selection.
    phase records installation, committed state/info update, or closed input.
    EventMeta carries the submitting origin separately.
    """

    EVENT: ClassVar[FluxDepEvent] = FluxDepEvent.INTERACTIVE_CHANGED
    context_id: int
    kind: InteractiveKind
    spectrum_name: str | None
    phase: Literal["opened", "updated", "closed"]


EventBus = BaseEventBus
