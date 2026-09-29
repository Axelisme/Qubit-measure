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


@dataclass(slots=True)
class _ArtifactRecord:
    result: object | None = None
    revision: int = 0
    current: tuple[object, ...] | None = None
    pending: tuple[object, ...] | None = None
    saved: tuple[object, ...] | None = None
    last_saved_path: str | None = None


class ArtifactTracker:
    """One session's status owner; callers supply current owner-thread facts.

    ``observe`` compares result object identity (not ``id()``) and the draft
    signature. ``started`` captures that signature. ``succeeded`` promotes only
    the pending signature and the actual output path; ``failed`` clears pending
    without erasing an earlier success. A later draft/result change can make a
    successfully saved artifact ``UNSAVED_CHANGES``. The caller may reset all
    successful baselines on a new loaded-result session.
    """

    def __init__(self) -> None:
        self._records = {kind: _ArtifactRecord() for kind in ArtifactKind}

    def observe(
        self,
        kind: ArtifactKind,
        *,
        result: object | None,
        has_figure: bool,
        path: str | None,
        comment: str = "",
    ) -> ArtifactSnapshot:
        rec = self._records[kind]
        if rec.result is not result:
            rec.result = result
            rec.revision += 1
        rec.current = (
            (rec.revision, path, comment)
            if kind is ArtifactKind.DATA
            else (rec.revision, path)
        )
        if result is None:
            status = SaveStatus.NO_RESULT
        elif rec.saved is None:
            status = SaveStatus.NOT_SAVED
        elif rec.current == rec.saved:
            status = SaveStatus.SAVED
        else:
            status = SaveStatus.UNSAVED_CHANGES
        return ArtifactSnapshot(
            kind=kind,
            status=status,
            default_path=path,
            last_saved_path=rec.last_saved_path,
            is_saveable=result is not None
            and (kind is ArtifactKind.DATA or has_figure),
        )

    def started(self, kind: ArtifactKind) -> None:
        rec = self._records[kind]
        if rec.current is None:
            raise RuntimeError(f"Cannot start {kind.value} save before observation")
        if rec.pending is not None:
            raise RuntimeError(f"{kind.value} save already pending")
        rec.pending = rec.current

    def succeeded(self, kind: ArtifactKind, actual_path: str) -> None:
        rec = self._records[kind]
        if rec.pending is None:
            raise RuntimeError(f"No pending {kind.value} save to complete")
        rec.saved = rec.pending
        rec.last_saved_path = actual_path
        rec.pending = None

    def failed(self, kind: ArtifactKind) -> None:
        self._records[kind].pending = None

    def reset_for_load(self) -> None:
        """Forget previous successes before publishing a loaded result."""
        self._records = {kind: _ArtifactRecord() for kind in ArtifactKind}
