"""EventBus — lightweight publish/subscribe for GUI-internal events.

All emits and subscribes happen on the main thread.  No Qt dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal, overload

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


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
class ContextChangedPayload(Payload):
    """Payload for CONTEXT_CHANGED: active ExpContext switched."""

    md: "MetaDict"
    ml: "ModuleLibrary"


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
class PredictorChangedPayload(Payload):
    """Payload for PREDICTOR_CHANGED: predictor state or values changed."""


@dataclass(frozen=True)
class InspectChangedPayload(Payload):
    """Payload for INSPECT_CHANGED: metadata/library changed via inspect dialog."""

    md: "MetaDict | None" = None


# ---------------------------------------------------------------------------
# GuiEvent enum
# ---------------------------------------------------------------------------


class GuiEvent(str, Enum):
    """Supported event names for EventBus."""

    # Data layer
    MD_CHANGED = "md_changed"  # MetaDict attribute was set or deleted
    CONTEXT_CHANGED = (
        "context_changed"  # active ExpContext switched (project load or switch)
    )

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
    INSPECT_CHANGED = "inspect_changed"  # metadata/library changed via inspect dialog


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

# Mapping from GuiEvent to its payload type (for overload resolution)
_EventPayloadMap = {
    GuiEvent.MD_CHANGED: MdChangedPayload,
    GuiEvent.CONTEXT_CHANGED: ContextChangedPayload,
    GuiEvent.TAB_ADDED: TabAddedPayload,
    GuiEvent.TAB_CLOSED: TabClosedPayload,
    GuiEvent.TAB_CONTENT_CHANGED: TabContentChangedPayload,
    GuiEvent.TAB_INTERACTION_CHANGED: TabInteractionChangedPayload,
    GuiEvent.RUN_LOCK_CHANGED: RunLockChangedPayload,
    GuiEvent.PREDICTOR_CHANGED: PredictorChangedPayload,
    GuiEvent.INSPECT_CHANGED: InspectChangedPayload,
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
        event: "Literal[GuiEvent.CONTEXT_CHANGED]",
        cb: Callable[[ContextChangedPayload], None],
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
        event: "Literal[GuiEvent.INSPECT_CHANGED]",
        cb: Callable[[InspectChangedPayload], None],
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
        event: "Literal[GuiEvent.CONTEXT_CHANGED]",
        cb: Callable[[ContextChangedPayload], None],
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
        event: "Literal[GuiEvent.INSPECT_CHANGED]",
        cb: Callable[[InspectChangedPayload], None],
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
        self, event: "Literal[GuiEvent.CONTEXT_CHANGED]", payload: ContextChangedPayload
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
        self, event: "Literal[GuiEvent.INSPECT_CHANGED]", payload: InspectChangedPayload
    ) -> None: ...

    def emit(self, event: GuiEvent, payload: Payload) -> None:
        if not isinstance(event, GuiEvent):
            raise TypeError(f"event must be a GuiEvent, got {type(event)}")
        for cb in list(self._subs.get(event, [])):
            try:
                cb(payload)
            except Exception:
                logger.warning(
                    "EventBus: exception in subscriber for %r", event, exc_info=True
                )
