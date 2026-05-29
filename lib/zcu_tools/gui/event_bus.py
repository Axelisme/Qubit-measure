"""EventBus — lightweight publish/subscribe for GUI-internal events.

All emits and subscribes happen on the main thread.  No Qt dependency.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, overload

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
    """Payload for RUN_LOCK_CHANGED.

    Emitted both when a run *starts* (running_tab_id set, outcome=None) and when
    it *ends* (running_tab_id=None, outcome in {finished, failed, cancelled}).
    Folding the terminal outcome here lets a subscriber distinguish success /
    failure / cancellation from a single event stream.
    """

    running_tab_id: str | None
    tab_id: str | None = None
    outcome: str | None = None  # 'finished' | 'failed' | 'cancelled' | None
    error_message: str | None = None


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
    GuiEvent.PREDICTOR_CHANGED: PredictorChangedPayload,
    GuiEvent.DEVICE_CHANGED: DeviceChangedPayload,
    GuiEvent.DEVICE_SETUP_CHANGED: DeviceSetupChangedPayload,
}


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[GuiEvent, list[Callable[[Any], None]]] = {}
        # Origin of the change currently being emitted — a neutral string the
        # bus never interprets ("agent" for RPC-client-driven changes, "" for
        # UI / system). A subscriber that cares (only the remote service's
        # change-buffer) reads ``current_origin`` synchronously inside its
        # callback, since emit runs the callbacks on the same call stack that
        # set it. Set either by ``acting_as`` (a sync scope around an RPC
        # handler) or per-emit via ``emit(..., origin=...)`` (async terminals
        # that carry the originating operation's lease origin).
        self._cur_origin: str = ""

    @property
    def current_origin(self) -> str:
        """Origin of the change being emitted right now (synchronous read)."""
        return self._cur_origin

    @contextmanager
    def acting_as(self, origin: str) -> "Generator[None, None, None]":
        """Mark synchronous emits inside this scope as caused by ``origin``."""
        prev = self._cur_origin
        self._cur_origin = origin
        try:
            yield
        finally:
            self._cur_origin = prev

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
        self,
        event: "Literal[GuiEvent.MD_CHANGED]",
        payload: MdChangedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.CONTEXT_SWITCHED]",
        payload: ContextSwitchedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.ML_CHANGED]",
        payload: MlChangedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.SOC_CHANGED]",
        payload: SocChangedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.TAB_ADDED]",
        payload: TabAddedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.TAB_CLOSED]",
        payload: TabClosedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.TAB_CONTENT_CHANGED]",
        payload: TabContentChangedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.TAB_INTERACTION_CHANGED]",
        payload: TabInteractionChangedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.RUN_LOCK_CHANGED]",
        payload: RunLockChangedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.PREDICTOR_CHANGED]",
        payload: PredictorChangedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.DEVICE_CHANGED]",
        payload: DeviceChangedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: "Literal[GuiEvent.DEVICE_SETUP_CHANGED]",
        payload: DeviceSetupChangedPayload,
        *,
        origin: str = ...,
    ) -> None: ...

    def emit(self, event: GuiEvent, payload: Payload, *, origin: str = "") -> None:
        if not isinstance(event, GuiEvent):
            raise TypeError(f"event must be a GuiEvent, got {type(event)}")
        expected_payload = _EventPayloadMap[event]
        if not isinstance(payload, expected_payload):
            raise TypeError(
                f"{event.value} expects {expected_payload.__name__}, "
                f"got {type(payload).__name__}"
            )
        # Override the current origin only when explicitly passed (async
        # terminal emits carry their operation's lease origin). When omitted,
        # the value set by an enclosing ``acting_as`` scope (or "") stands.
        prev_origin = self._cur_origin
        if origin:
            self._cur_origin = origin
        exceptions: list[BaseException] = []
        try:
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
        finally:
            self._cur_origin = prev_origin
        if exceptions:
            raise exceptions[0]
