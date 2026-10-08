"""In-process completion facts that carry terminal detail absent from wire events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

from zcu_tools.experiment.run_saving import SaveReport
from zcu_tools.gui.event_bus import BasePayload


class CompletionEvent(str, Enum):
    ANALYZE_FAILED = "analyze_failed_detail"
    SAVE_DATA_FINISHED = "save_data_finished_detail"
    SAVE_ARTIFACTS_FINISHED = "save_artifacts_finished_detail"


@dataclass(frozen=True)
class AnalyzeFailedPayload(BasePayload):
    EVENT: ClassVar[CompletionEvent] = CompletionEvent.ANALYZE_FAILED
    tab_id: str
    stage: Literal["primary", "post"]
    error_message: str


@dataclass(frozen=True)
class SaveDataFinishedPayload(BasePayload):
    """Latest native data terminal fact, delivered after State and handle settle.

    tab_id identifies the owning tab; data_path is the canonical reservation.
    error is the operation diagnostic, including failed output paths. report
    contains every attempted output, or None for preparation/submission/ledger
    failure that prevented the shared saver returning a report.
    """

    EVENT: ClassVar[CompletionEvent] = CompletionEvent.SAVE_DATA_FINISHED
    tab_id: str
    data_path: str
    error: str | None = None
    report: SaveReport | None = None


@dataclass(frozen=True)
class SaveArtifactsFinishedPayload(BasePayload):
    EVENT: ClassVar[CompletionEvent] = CompletionEvent.SAVE_ARTIFACTS_FINISHED
    tab_id: str
    error: str | None = None


__all__ = [
    "AnalyzeFailedPayload",
    "CompletionEvent",
    "SaveDataFinishedPayload",
    "SaveArtifactsFinishedPayload",
]
