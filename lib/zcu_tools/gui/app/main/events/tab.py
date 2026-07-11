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


class TabInteractionFact(str, Enum):
    """Closed domain facts carried by the interaction event envelope."""

    RUN_START_REJECTED = "run_start_rejected"
    PRIMARY_ANALYZE_STARTED = "primary_analyze_started"
    PRIMARY_ANALYZE_SUCCEEDED = "primary_analyze_succeeded"
    PRIMARY_ANALYZE_FAILED = "primary_analyze_failed"
    PRIMARY_ANALYZE_CANCELLED = "primary_analyze_cancelled"
    PRIMARY_ANALYZE_START_REJECTED = "primary_analyze_start_rejected"
    POST_ANALYZE_STARTED = "post_analyze_started"
    POST_ANALYZE_SUCCEEDED = "post_analyze_succeeded"
    POST_ANALYZE_FAILED = "post_analyze_failed"
    POST_ANALYZE_START_REJECTED = "post_analyze_start_rejected"
    SAVE_STARTED = "save_started"
    SAVE_SUCCEEDED = "save_succeeded"
    SAVE_FAILED = "save_failed"
    ANALYZE_PARAMS_CHANGED = "analyze_params_changed"
    POST_ANALYZE_PARAMS_CHANGED = "post_analyze_params_changed"
    SAVE_PATHS_CHANGED = "save_paths_changed"


class TabContentFact(str, Enum):
    """Closed facts for committed tab-owned result content."""

    RUN_RESULT_COMMITTED = "run_result_committed"
    LOADED_RESULT_COMMITTED = "loaded_result_committed"
    PRIMARY_ANALYSIS_COMMITTED = "primary_analysis_committed"
    POST_ANALYSIS_COMMITTED = "post_analysis_committed"


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
    fact: TabContentFact


@dataclass(frozen=True)
class TabInteractionChangedPayload(_TabPayload):
    """Payload for TAB_INTERACTION_CHANGED."""

    EVENT: ClassVar[TabEvent] = TabEvent.TAB_INTERACTION_CHANGED
    tab_id: str
    fact: TabInteractionFact
