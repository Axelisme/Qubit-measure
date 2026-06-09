"""EventBus — measure-gui experiment-surface events.

The publish/subscribe mechanism lives in :mod:`zcu_tools.gui.event_bus`; the
session-core events (md/ml/context/soc/predictor/device) live in
:mod:`zcu_tools.gui.session.events`. This module supplies only measure-gui's own
experiment-surface events (tabs + runs) and re-exports the shared ``EventBus``.
All emits and subscribes happen on the main thread (no Qt dependency). Each
payload carries its own event tag (``EVENT`` ClassVar) and the payload type alone
determines the event, so a payload can never be paired with the wrong event.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from zcu_tools.gui.event_bus import BaseEventBus, BasePayload

# ---------------------------------------------------------------------------
# GuiEvent enum
# ---------------------------------------------------------------------------


class GuiEvent(str, Enum):
    """Supported experiment-surface event names for EventBus."""

    # Controller / Tab layer
    TAB_ADDED = "tab_added"  # payload: TabAddedPayload
    TAB_CLOSED = "tab_closed"  # payload: TabClosedPayload
    TAB_CONTENT_CHANGED = "tab_content_changed"  # payload: TabContentChangedPayload
    TAB_INTERACTION_CHANGED = (
        "tab_interaction_changed"  # payload: TabInteractionChangedPayload
    )
    RUN_STARTED = "run_started"  # payload: RunStartedPayload
    RUN_FINISHED = "run_finished"  # payload: RunFinishedPayload


# ---------------------------------------------------------------------------
# Payload base class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Payload(BasePayload):
    """Base for all measure-gui experiment EventBus payloads. Subclasses set
    ``EVENT``."""

    EVENT: ClassVar[GuiEvent]


# ---------------------------------------------------------------------------
# Concrete payload types — one per GuiEvent member
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


EventBus = BaseEventBus
