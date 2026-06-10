"""Tab-domain event definitions for measure-gui.

The tab domain owns the ``TabEvent`` enum and all tab-lifecycle payloads.
Import from this module (``events.tab``) rather than the old ``event_bus``
flat module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from zcu_tools.gui.event_bus import BasePayload


class TabEvent(str, Enum):
    """Tab-lifecycle event names (wire names are the enum values)."""

    TAB_ADDED = "tab_added"
    TAB_CLOSED = "tab_closed"
    TAB_CONTENT_CHANGED = "tab_content_changed"
    TAB_INTERACTION_CHANGED = "tab_interaction_changed"


@dataclass(frozen=True)
class _TabPayload(BasePayload):
    """Base for all tab-domain EventBus payloads. Subclasses set ``EVENT``."""

    EVENT: ClassVar[TabEvent]


@dataclass(frozen=True)
class TabAddedPayload(_TabPayload):
    """Payload for TAB_ADDED."""

    EVENT: ClassVar[TabEvent] = TabEvent.TAB_ADDED
    tab_id: str
    adapter_name: str


@dataclass(frozen=True)
class TabClosedPayload(_TabPayload):
    """Payload for TAB_CLOSED."""

    EVENT: ClassVar[TabEvent] = TabEvent.TAB_CLOSED
    tab_id: str


@dataclass(frozen=True)
class TabContentChangedPayload(_TabPayload):
    """Payload for TAB_CONTENT_CHANGED."""

    EVENT: ClassVar[TabEvent] = TabEvent.TAB_CONTENT_CHANGED
    tab_id: str


@dataclass(frozen=True)
class TabInteractionChangedPayload(_TabPayload):
    """Payload for TAB_INTERACTION_CHANGED."""

    EVENT: ClassVar[TabEvent] = TabEvent.TAB_INTERACTION_CHANGED
    tab_id: str
