"""EventBus — dispersive-fit-gui internal events.

The publish/subscribe mechanism lives in :mod:`zcu_tools.gui.event_bus`; this
module supplies the dispersive event enum and payloads and re-exports the shared
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
class Payload(BasePayload):
    """Base for all dispersive EventBus payloads. Subclasses set ``EVENT``."""

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


EventBus = BaseEventBus
