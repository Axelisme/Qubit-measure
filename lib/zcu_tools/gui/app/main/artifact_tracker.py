"""Tab-owned artifact status shared by application save flows and presentation.

The tracker is Qt-free and process-local. Its signature uses result identity,
current draft path, and the data comment; only a successful terminal save records
an actual path. A loaded result has no successful save in this session.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ArtifactKind(str, Enum):
    DATA = "data"
    ANALYSIS = "analysis"
    POST_ANALYSIS = "post_analysis"


class SaveStatus(str, Enum):
    NO_RESULT = "no_result"
    NOT_SAVED = "not_saved"
    UNSAVED_CHANGES = "unsaved_changes"
    SAVED = "saved"


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    kind: ArtifactKind
    status: SaveStatus
    default_path: str | None
    last_saved_path: str | None
    is_saveable: bool


class ArtifactTracker:
    """One session's status owner; callers supply current owner-thread facts.

    ``observe`` compares result object identity (not ``id()``) and the draft
    signature. ``started`` captures that signature. ``succeeded`` promotes only
    the pending signature and the actual output path; ``failed`` clears pending
    without erasing an earlier success. A later draft/result change can make a
    successfully saved artifact ``UNSAVED_CHANGES``. The caller may reset all
    successful baselines on a new loaded-result session.
    """

    def observe(
        self,
        kind: ArtifactKind,
        *,
        result: object | None,
        has_figure: bool,
        path: str | None,
        comment: str = "",
    ) -> ArtifactSnapshot:
        raise NotImplementedError

    def started(self, kind: ArtifactKind) -> None:
        raise NotImplementedError

    def succeeded(self, kind: ArtifactKind, actual_path: str) -> None:
        raise NotImplementedError

    def failed(self, kind: ArtifactKind) -> None:
        raise NotImplementedError

    def reset_for_load(self) -> None:
        """Forget previous successes before publishing a loaded result."""
        raise NotImplementedError
