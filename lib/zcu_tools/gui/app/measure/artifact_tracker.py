"""Owner-thread, process-local save history for named tab artifacts.

Images track successful saves of a result/Figure pair, not artist or destination
changes. Data also tracks its path and comment. Projection retains only current
artifacts; in-flight attempts retain their own record until the caller drops them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class ArtifactKind(str, Enum):
    DATA = "data"
    ANALYSIS = "analysis"
    POST_ANALYSIS = "post_analysis"


@dataclass(frozen=True, slots=True)
class ArtifactKey:
    kind: ArtifactKind
    figure_name: str | None = None

    def __post_init__(self) -> None:
        if self.kind is ArtifactKind.DATA:
            if self.figure_name is not None:
                raise ValueError("Data artifacts cannot have a figure name")
        elif not isinstance(self.figure_name, str) or not self.figure_name:
            raise ValueError("Image artifacts require a nonempty figure name")


class SaveStatus(str, Enum):
    NO_RESULT = "no_result"
    NOT_SAVED = "not_saved"
    UNSAVED_CHANGES = "unsaved_changes"
    SAVED = "saved"


@dataclass(frozen=True, slots=True)
class ArtifactObservation:
    key: ArtifactKey
    result: object | None
    figure: object | None
    path: str | None
    comment: str = ""


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    key: ArtifactKey
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
    current: tuple[object, ...] = ()
    pending: SaveAttempt | None = None
    saved: tuple[object, ...] | None = None
    last_saved_path: str | None = None
    is_saveable: bool = False


@dataclass(frozen=True, slots=True)
class SaveAttempt:
    """Opaque submission identity returned by ``ArtifactTracker.started``.

    The save operation owns this reference, including after a pane replacement.
    Settle it on the tracker owner thread and then release it.
    """

    key: ArtifactKey
    _record: _ArtifactRecord
    _signature: tuple[object, ...]

    @property
    def pending(self) -> bool:
        return self._record.pending is self

    def succeed(self, actual_path: str) -> None:
        self._ensure_pending()
        self._record.saved = self._signature
        self._record.last_saved_path = actual_path
        self._record.pending = None

    def fail(self) -> None:
        self._ensure_pending()
        self._record.pending = None

    def _ensure_pending(self) -> None:
        if not self.pending:
            raise RuntimeError("Save attempt is already settled")


class ArtifactTracker:
    """Project current artifacts and settle captured saves on one owner thread.

    ``project`` is the complete current set, in display order. Duplicate keys
    fail before mutation. Image replacement creates a new record; late completion
    can update only the old one. Data keeps its revision/path/comment semantics.
    Load resets the visible records without retargeting outstanding attempts.
    """

    def __init__(self) -> None:
        self._records: dict[ArtifactKey, _ArtifactRecord] = {}

    def project(
        self, observations: Sequence[ArtifactObservation]
    ) -> tuple[ArtifactSnapshot, ...]:
        if len({item.key for item in observations}) != len(observations):
            raise ValueError("Artifact projection requires unique keys")
        records: dict[ArtifactKey, _ArtifactRecord] = {}
        snapshots: list[ArtifactSnapshot] = []
        for item in observations:
            rec = self._records.get(item.key)
            identity_changed = rec is None or (
                rec.result is not item.result or rec.figure is not item.figure
            )
            if rec is None or (
                identity_changed and item.key.kind is not ArtifactKind.DATA
            ):
                rec = _ArtifactRecord()
            if identity_changed:
                rec.result = item.result
                rec.figure = item.figure
                rec.revision += 1
            rec.current = (
                (rec.revision, item.path, item.comment)
                if item.key.kind is ArtifactKind.DATA
                else (rec.revision,)
            )
            rec.is_saveable = item.result is not None and (
                item.key.kind is ArtifactKind.DATA or item.figure is not None
            )
            if item.result is None:
                status = SaveStatus.NO_RESULT
            elif rec.saved is None:
                status = SaveStatus.NOT_SAVED
            elif rec.current == rec.saved:
                status = SaveStatus.SAVED
            elif item.key.kind is ArtifactKind.DATA:
                status = SaveStatus.UNSAVED_CHANGES
            else:
                status = SaveStatus.NOT_SAVED
            snapshots.append(
                ArtifactSnapshot(
                    key=item.key,
                    status=status,
                    default_path=item.path,
                    last_saved_path=(
                        rec.last_saved_path
                        if item.key.kind is ArtifactKind.DATA
                        or rec.current == rec.saved
                        else None
                    ),
                    is_saveable=rec.is_saveable,
                )
            )
            records[item.key] = rec
        self._records = records
        return tuple(snapshots)

    def started(self, key: ArtifactKey) -> SaveAttempt:
        rec = self._records.get(key)
        if rec is None or not rec.is_saveable:
            raise ValueError(f"Artifact {key!r} is not saveable")
        if rec.pending is not None:
            raise RuntimeError(f"Artifact {key!r} save already pending")
        attempt = SaveAttempt(key, rec, rec.current)
        rec.pending = attempt
        return attempt

    def reset_for_load(self) -> None:
        """Forget visible successful baselines, without retargeting pending saves."""
        self._records = {}
