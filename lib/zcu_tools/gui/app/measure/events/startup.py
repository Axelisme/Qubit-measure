"""Startup progress and terminal facts, published on the application owner."""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from zcu_tools.gui.event_bus import BasePayload


class StartupEvent(str, Enum):
    """Wire name for the script runner's progress/terminal publication."""

    CHANGED = "startup_changed"


@dataclass(frozen=True)
class StartupChangedPayload(BasePayload):
    """Current script activity and human-readable status.

    running is True until owner terminal delivery. message is progress or the
    terminal summary. error is None unless an unhandled script error occurred;
    then it includes script path, line when available, and the original cause.
    """

    EVENT: ClassVar[StartupEvent] = StartupEvent.CHANGED
    running: bool
    message: str
    error: str | None = None
