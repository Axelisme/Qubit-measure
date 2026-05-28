"""EventBus — lightweight publish/subscribe for GUI-internal events.

All emits and subscribes happen on the main thread.  No Qt dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal, overload

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import SocCfgHandle, SocHandle
    from zcu_tools.gui.services.device import DeviceSetupSnapshot
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Payload base class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Payload:
    """Base class for all EventBus payloads."""


# ---------------------------------------------------------------------------
# Concrete payload types — one per GuiEvent member
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MdChangedPayload(Payload):
    """Payload for MD_CHANGED: a MetaDict attribute was set or deleted."""

    md: "MetaDict"


@dataclass(frozen=True)
class ContextSwitchedPayload(Payload):
    """Payload for CONTEXT_SWITCHED: active md/ml context switched."""

    md: "MetaDict"
    ml: "ModuleLibrary"


@dataclass(frozen=True)
class MlChangedPayload(Payload):
    """Payload for ML_CHANGED: ModuleLibrary contents changed."""

    ml: "ModuleLibrary"


@dataclass(frozen=True)
class SocChangedPayload(Payload):
    """Payload for SOC_CHANGED: soc/soccfg connection changed."""

    soc: "SocHandle | None"
    soccfg: "SocCfgHandle | None"
    is_mock: bool = False


@dataclass(frozen=True)
class TabAddedPayload(Payload):
    """Payload for TAB_ADDED."""

    tab_id: str
    adapter_name: str


@dataclass(frozen=True)
class TabClosedPayload(Payload):
    """Payload for TAB_CLOSED."""

    tab_id: str


@dataclass(frozen=True)
class TabContentChangedPayload(Payload):
    """Payload for TAB_CONTENT_CHANGED."""

    tab_id: str


@dataclass(frozen=True)
class TabInteractionChangedPayload(Payload):
    """Payload for TAB_INTERACTION_CHANGED."""

    tab_id: str


@dataclass(frozen=True)
class RunLockChangedPayload(Payload):
    """Payload for RUN_LOCK_CHANGED."""

    running_tab_id: str | None


@dataclass(frozen=True)
class RunFinishedPayload(Payload):
    """Payload for RUN_FINISHED: a run completed successfully."""

    tab_id: str


@dataclass(frozen=True)
class RunFailedPayload(Payload):
    """Payload for RUN_FAILED: a run ended with an error."""

    tab_id: str
    error_message: str


@dataclass(frozen=True)
class RunProgressChangedPayload(Payload):
    """Payload for RUN_PROGRESS_CHANGED: progress bar stepped N times."""

    tab_id: str


@dataclass(frozen=True)
class PredictorChangedPayload(Payload):
    """Payload for PREDICTOR_CHANGED: predictor state or values changed."""


@dataclass(frozen=True)
class DeviceChangedPayload(Payload):
    """Payload for DEVICE_CHANGED: a device was registered or dropped."""

    name: str | None = None


@dataclass(frozen=True)
class DeviceSetupChangedPayload(Payload):
    """Payload for DEVICE_SETUP_CHANGED: active setup progress/lifecycle changed."""

    active_setup: "DeviceSetupSnapshot | None"


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
    RUN_LOCK_CHANGED = "run_lock_changed"  # payload: RunLockChangedPayload
    RUN_FINISHED = "run_finished"  # payload: RunFinishedPayload
    RUN_FAILED = "run_failed"  # payload: RunFailedPayload
    RUN_PROGRESS_CHANGED = "run_progress_changed"  # payload: RunProgressChangedPayload

    # UI / Panel layer
    PREDICTOR_CHANGED = "predictor_changed"  # predictor state or values changed

    # Device layer
    DEVICE_CHANGED = "device_changed"  # device registered or dropped
    DEVICE_SETUP_CHANGED = "device_setup_changed"  # active setup/progress changed


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

# Mapping from GuiEvent to its payload type (for overload resolution)
_EventPayloadMap = {
    GuiEvent.MD_CHANGED: MdChangedPayload,
    GuiEvent.ML_CHANGED: MlChangedPayload,
    GuiEvent.CONTEXT_SWITCHED: ContextSwitchedPayload,
    GuiEvent.SOC_CHANGED: SocChangedPayload,
    GuiEvent.TAB_ADDED: TabAddedPayload,
    GuiEvent.TAB_CLOSED: TabClosedPayload,
    GuiEvent.TAB_CONTENT_CHANGED: TabContentChangedPayload,
    GuiEvent.TAB_INTERACTION_CHANGED: TabInteractionChangedPayload,
    GuiEvent.RUN_LOCK_CHANGED: RunLockChangedPayload,
    GuiEvent.RUN_FINISHED: RunFinishedPayload,
    GuiEvent.RUN_FAILED: RunFailedPayload,
    GuiEvent.RUN_PROGRESS_CHANGED: RunProgressChangedPayload,
    GuiEvent.PREDICTOR_CHANGED: PredictorChangedPayload,
    GuiEvent.DEVICE_CHANGED: DeviceChangedPayload,
    GuiEvent.DEVICE_SETUP_CHANGED: DeviceSetupChangedPayload,
}


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[GuiEvent, list[Callable[[Any], None]]] = {}

    # ------------------------------------------------------------------
    # subscribe overloads
    # ------------------------------------------------------------------

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.MD_CHANGED]",
        cb: Callable[[MdChangedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.CONTEXT_SWITCHED]",
        cb: Callable[[ContextSwitchedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.ML_CHANGED]",
        cb: Callable[[MlChangedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.SOC_CHANGED]",
        cb: Callable[[SocChangedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.TAB_ADDED]",
        cb: Callable[[TabAddedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.TAB_CLOSED]",
        cb: Callable[[TabClosedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.TAB_CONTENT_CHANGED]",
        cb: Callable[[TabContentChangedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.TAB_INTERACTION_CHANGED]",
        cb: Callable[[TabInteractionChangedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.RUN_LOCK_CHANGED]",
        cb: Callable[[RunLockChangedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.PREDICTOR_CHANGED]",
        cb: Callable[[PredictorChangedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.DEVICE_CHANGED]",
        cb: Callable[[DeviceChangedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.DEVICE_SETUP_CHANGED]",
        cb: Callable[[DeviceSetupChangedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.RUN_FINISHED]",
        cb: Callable[[RunFinishedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.RUN_FAILED]",
        cb: Callable[[RunFailedPayload], None],
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event: "Literal[GuiEvent.RUN_PROGRESS_CHANGED]",
        cb: Callable[[RunProgressChangedPayload], None],
    ) -> None: ...

    def subscribe(self, event: GuiEvent, cb: Callable[[Any], None]) -> None:
        if not isinstance(event, GuiEvent):
            raise TypeError(f"event must be a GuiEvent, got {type(event)}")
        self._subs.setdefault(event, []).append(cb)

    # ------------------------------------------------------------------
    # unsubscribe overloads
    # ------------------------------------------------------------------

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.MD_CHANGED]",
        cb: Callable[[MdChangedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.CONTEXT_SWITCHED]",
        cb: Callable[[ContextSwitchedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.ML_CHANGED]",
        cb: Callable[[MlChangedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.SOC_CHANGED]",
        cb: Callable[[SocChangedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.TAB_ADDED]",
        cb: Callable[[TabAddedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.TAB_CLOSED]",
        cb: Callable[[TabClosedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.TAB_CONTENT_CHANGED]",
        cb: Callable[[TabContentChangedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.TAB_INTERACTION_CHANGED]",
        cb: Callable[[TabInteractionChangedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.RUN_LOCK_CHANGED]",
        cb: Callable[[RunLockChangedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.PREDICTOR_CHANGED]",
        cb: Callable[[PredictorChangedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.DEVICE_CHANGED]",
        cb: Callable[[DeviceChangedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.DEVICE_SETUP_CHANGED]",
        cb: Callable[[DeviceSetupChangedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.RUN_FINISHED]",
        cb: Callable[[RunFinishedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.RUN_FAILED]",
        cb: Callable[[RunFailedPayload], None],
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event: "Literal[GuiEvent.RUN_PROGRESS_CHANGED]",
        cb: Callable[[RunProgressChangedPayload], None],
    ) -> None: ...

    def unsubscribe(self, event: GuiEvent, cb: Callable[[Any], None]) -> None:
        lst = self._subs.get(event, [])
        try:
            lst.remove(cb)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # emit overloads
    # ------------------------------------------------------------------

    @overload
    def emit(
        self, event: "Literal[GuiEvent.MD_CHANGED]", payload: MdChangedPayload
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.CONTEXT_SWITCHED]",
        payload: ContextSwitchedPayload,
    ) -> None: ...

    @overload
    def emit(
        self, event: "Literal[GuiEvent.ML_CHANGED]", payload: MlChangedPayload
    ) -> None: ...

    @overload
    def emit(
        self, event: "Literal[GuiEvent.SOC_CHANGED]", payload: SocChangedPayload
    ) -> None: ...

    @overload
    def emit(
        self, event: "Literal[GuiEvent.TAB_ADDED]", payload: TabAddedPayload
    ) -> None: ...

    @overload
    def emit(
        self, event: "Literal[GuiEvent.TAB_CLOSED]", payload: TabClosedPayload
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.TAB_CONTENT_CHANGED]",
        payload: TabContentChangedPayload,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.TAB_INTERACTION_CHANGED]",
        payload: TabInteractionChangedPayload,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.RUN_LOCK_CHANGED]",
        payload: RunLockChangedPayload,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.PREDICTOR_CHANGED]",
        payload: PredictorChangedPayload,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.DEVICE_CHANGED]",
        payload: DeviceChangedPayload,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.DEVICE_SETUP_CHANGED]",
        payload: DeviceSetupChangedPayload,
    ) -> None: ...

    @overload
    def emit(
        self, event: "Literal[GuiEvent.RUN_FINISHED]", payload: RunFinishedPayload
    ) -> None: ...

    @overload
    def emit(
        self, event: "Literal[GuiEvent.RUN_FAILED]", payload: RunFailedPayload
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.RUN_PROGRESS_CHANGED]",
        payload: RunProgressChangedPayload,
    ) -> None: ...

    def emit(self, event: GuiEvent, payload: Payload) -> None:
        if not isinstance(event, GuiEvent):
            raise TypeError(f"event must be a GuiEvent, got {type(event)}")
        expected_payload = _EventPayloadMap[event]
        if not isinstance(payload, expected_payload):
            raise TypeError(
                f"{event.value} expects {expected_payload.__name__}, "
                f"got {type(payload).__name__}"
            )
        exceptions: list[BaseException] = []
        for cb in list(self._subs.get(event, [])):
            try:
                cb(payload)
            except Exception as exc:
                exceptions.append(exc)
                logger.exception(
                    "EventBus subscriber failed: event=%s callback=%r",
                    event.value,
                    cb,
                )
        if exceptions:
            raise exceptions[0]
