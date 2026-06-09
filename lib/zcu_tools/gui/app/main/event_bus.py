"""EventBus — measure-gui internal events.

The publish/subscribe mechanism lives in :mod:`zcu_tools.gui.event_bus`; this
module supplies the measure-gui event enum and payloads and re-exports the shared
``EventBus``. All emits and subscribes happen on the main thread (no Qt
dependency). Each payload carries its own event tag (``EVENT`` ClassVar) and the
payload type alone determines the event, so a payload can never be paired with
the wrong event.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from zcu_tools.gui.event_bus import BaseEventBus, BasePayload

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import SocCfgHandle, SocHandle
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


# ---------------------------------------------------------------------------
# GuiEvent enum
# ---------------------------------------------------------------------------


class GuiEvent(str, Enum):
    """Supported event names for EventBus."""

    # Data layer
    MD_CHANGED = "md_changed"  # MetaDict attribute was set or deleted
    ML_CHANGED = "ml_changed"  # ModuleLibrary contents changed
    CONTEXT_SWITCHED = "context_switched"  # active md/ml context switched
    SOC_CHANGED = "soc_changed"  # soc/soccfg connection changed

    # Controller / Tab layer
    TAB_ADDED = "tab_added"  # payload: TabAddedPayload
    TAB_CLOSED = "tab_closed"  # payload: TabClosedPayload
    TAB_CONTENT_CHANGED = "tab_content_changed"  # payload: TabContentChangedPayload
    TAB_INTERACTION_CHANGED = (
        "tab_interaction_changed"  # payload: TabInteractionChangedPayload
    )
    RUN_STARTED = "run_started"  # payload: RunStartedPayload
    RUN_FINISHED = "run_finished"  # payload: RunFinishedPayload

    # UI / Panel layer
    PREDICTOR_CHANGED = "predictor_changed"  # predictor state or values changed

    # Device layer
    DEVICE_CHANGED = "device_changed"  # device registered or dropped
    DEVICE_SETUP_STARTED = "device_setup_started"  # payload: DeviceSetupStartedPayload
    DEVICE_SETUP_FINISHED = (
        "device_setup_finished"  # payload: DeviceSetupFinishedPayload
    )


# ---------------------------------------------------------------------------
# Payload base class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Payload(BasePayload):
    """Base for all measure-gui EventBus payloads. Subclasses set ``EVENT``."""

    EVENT: ClassVar[GuiEvent]


# ---------------------------------------------------------------------------
# Concrete payload types — one per GuiEvent member
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MdChangedPayload(Payload):
    """Payload for MD_CHANGED: a MetaDict attribute was set or deleted."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.MD_CHANGED
    md: "MetaDict"


@dataclass(frozen=True)
class ContextSwitchedPayload(Payload):
    """Payload for CONTEXT_SWITCHED: active md/ml context switched."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.CONTEXT_SWITCHED
    md: "MetaDict"
    ml: "ModuleLibrary"


@dataclass(frozen=True)
class MlChangedPayload(Payload):
    """Payload for ML_CHANGED: ModuleLibrary contents changed."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.ML_CHANGED
    ml: "ModuleLibrary"


@dataclass(frozen=True)
class SocChangedPayload(Payload):
    """Payload for SOC_CHANGED: soc/soccfg connection changed."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.SOC_CHANGED
    soc: "SocHandle | None"
    soccfg: "SocCfgHandle | None"
    is_mock: bool = False


@dataclass(frozen=True)
class TabAddedPayload(Payload):
    """Payload for TAB_ADDED."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.TAB_ADDED
    tab_id: str
    adapter_name: str


@dataclass(frozen=True)
class TabClosedPayload(Payload):
    """Payload for TAB_CLOSED."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.TAB_CLOSED
    tab_id: str


@dataclass(frozen=True)
class TabContentChangedPayload(Payload):
    """Payload for TAB_CONTENT_CHANGED."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.TAB_CONTENT_CHANGED
    tab_id: str


@dataclass(frozen=True)
class TabInteractionChangedPayload(Payload):
    """Payload for TAB_INTERACTION_CHANGED."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.TAB_INTERACTION_CHANGED
    tab_id: str


@dataclass(frozen=True)
class RunStartedPayload(Payload):
    """Payload for RUN_STARTED: a run began on ``tab_id`` (the run lock is now
    held by it)."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.RUN_STARTED
    tab_id: str


@dataclass(frozen=True)
class RunFinishedPayload(Payload):
    """Payload for RUN_FINISHED: the run on ``tab_id`` reached a terminal state
    (the run lock is released). ``outcome`` distinguishes success / failure /
    cancellation; ``error_message`` is set only on failure."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.RUN_FINISHED
    tab_id: str
    outcome: str  # 'finished' | 'failed' | 'cancelled'
    error_message: str | None = None


@dataclass(frozen=True)
class PredictorChangedPayload(Payload):
    """Payload for PREDICTOR_CHANGED: predictor state or values changed."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.PREDICTOR_CHANGED


@dataclass(frozen=True)
class DeviceChangedPayload(Payload):
    """Payload for DEVICE_CHANGED: a device was registered or dropped."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.DEVICE_CHANGED
    name: str | None = None


@dataclass(frozen=True)
class DeviceSetupStartedPayload(Payload):
    """Payload for DEVICE_SETUP_STARTED: a setup began on device ``name`` (its
    progress is pollable via operation.progress, by operation_id)."""

    EVENT: ClassVar[GuiEvent] = GuiEvent.DEVICE_SETUP_STARTED
    name: str


@dataclass(frozen=True)
class DeviceSetupFinishedPayload(Payload):
    """Payload for DEVICE_SETUP_FINISHED: the setup on device ``name`` reached a
    terminal state. ``outcome`` ∈ finished/failed/cancelled; ``error_message``
    set only on failure. (Mirrors RUN_STARTED / RUN_FINISHED.)"""

    EVENT: ClassVar[GuiEvent] = GuiEvent.DEVICE_SETUP_FINISHED
    name: str
    outcome: str  # 'finished' | 'failed' | 'cancelled'
    error_message: str | None = None


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


EventBus = BaseEventBus
