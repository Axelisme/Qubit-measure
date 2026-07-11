"""In-process completion facts that carry terminal detail absent from wire events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

from zcu_tools.gui.event_bus import BasePayload


class CompletionEvent(str, Enum):
    ANALYZE_FAILED = "analyze_failed_detail"
    SAVE_FINISHED = "save_finished_detail"


@dataclass(frozen=True)
class AnalyzeFailedPayload(BasePayload):
    EVENT: ClassVar[CompletionEvent] = CompletionEvent.ANALYZE_FAILED
    tab_id: str
    stage: Literal["primary", "post"]
    error_message: str


@dataclass(frozen=True)
class SaveFinishedPayload(BasePayload):
    EVENT: ClassVar[CompletionEvent] = CompletionEvent.SAVE_FINISHED
    tab_id: str
    data_path: str
    image_path: str | None
    data_error: str | None = None
    image_error: str | None = None


__all__ = ["AnalyzeFailedPayload", "CompletionEvent", "SaveFinishedPayload"]
