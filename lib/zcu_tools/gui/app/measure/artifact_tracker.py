"""Tab-owned artifact status shared by application save flows and presentation.

The tracker is Qt-free and process-local. Data tracks result identity, draft path,
and comment. Images track successful saves of the result/figure pair, not content
or destination edits. A loaded result has no successful save in this session.
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

    @property
    def needs_save(self) -> bool:
        return self.is_saveable and self.status in (
            SaveStatus.NOT_SAVED,
            SaveStatus.UNSAVED_CHANGES,
        )


@dataclass(slots=True)
class _ArtifactRecord:
    result: object | None = None
    figure: object | None = None
    revision: int = 0
    current: tuple[object, ...] | None = None
    pending: tuple[object, ...] | None = None
    saved: tuple[object, ...] | None = None
    last_saved_path: str | None = None


class ArtifactTracker:
    """One session's status owner; callers supply current owner-thread facts.

    ``observe`` compares object identity (not ``id()``). Data also tracks its
    draft signature; images only track which result/figure pair was saved.
    ``started`` captures that identity so a late success cannot mark a replacement
    image saved. ``failed`` preserves an earlier success. Image edits and path
    changes never invalidate that success. Load resets all successful baselines.
    """

    def __init__(self) -> None:
        self._records = {kind: _ArtifactRecord() for kind in ArtifactKind}

    def observe(
        self,
        kind: ArtifactKind,
        *,
        result: object | None,
        figure: object | None,
        path: str | None,
        comment: str = "",
    ) -> ArtifactSnapshot:
        rec = self._records[kind]
        if rec.result is not result or rec.figure is not figure:
            rec.result = result
            rec.figure = figure
            rec.revision += 1
        rec.current = (
            (rec.revision, path, comment)
            if kind is ArtifactKind.DATA
            else (rec.revision,)
        )
        if result is None:
            status = SaveStatus.NO_RESULT
        elif rec.saved is None:
            status = SaveStatus.NOT_SAVED
        elif rec.current == rec.saved:
            status = SaveStatus.SAVED
        elif kind is ArtifactKind.DATA:
            status = SaveStatus.UNSAVED_CHANGES
        else:
            status = SaveStatus.NOT_SAVED
        return ArtifactSnapshot(
            kind=kind,
            status=status,
            default_path=path,
            last_saved_path=(
                rec.last_saved_path
                if kind is ArtifactKind.DATA or rec.current == rec.saved
                else None
            ),
            is_saveable=result is not None
            and (kind is ArtifactKind.DATA or figure is not None),
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
