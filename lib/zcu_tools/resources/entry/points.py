"""Working-point views over entry-coordinated layered transactions."""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Literal

from zcu_tools.resources.document_store import DocumentStore

from . import _point_origin
from .layering import compose
from .schema import (
    ComponentSchema,
    LayeredDocument,
    PointDocument,
    PointGeneral,
    SetupDocument,
)
from .views import (
    ComponentView,
    EditView,
    GeneralView,
    stage_component,
    stage_general,
)


class PointView:
    """A bound point's working-unit values over its entry's shared setup snapshot.

    Obtain a view from ResultEntry.use_point or new_point. Component attributes
    such as Q1 read point values first, then setup values, without disk I/O.
    Unknown components or unset fields raise AttributeError. Assignments commit
    through edit(): existing fields retain their layer and new fields go to point.
    Successful field writes clear cloned_from even when the value is unchanged.
    General metadata belongs only to this point. Call refresh to observe disk edits;
    opening another view does not change which point this view addresses.
    """

    def __init__(
        self,
        store: DocumentStore[PointDocument],
        setup: DocumentStore[SetupDocument],
        source: Path,
        edit: Callable[[], AbstractContextManager[LayeredDocument]],
        refresh: Callable[[], None],
    ) -> None:
        """Bind entry-owned stores and transaction callbacks without I/O.

        store and setup are validated working-unit stores for the same entry.
        source is store's point.yaml path under that entry's points/<label>.
        edit returns an independent layered draft context that reloads, validates
        and commits both layers. refresh reloads both stores or raises without
        publishing invalid snapshots. ResultEntry supplies these callbacks.
        Construction neither calls them nor validates the supplied dependencies.
        """
        self._store = store
        self._setup = setup
        self._source = source
        _point_origin.register(self, source)
        self._edit_document = edit
        self._refresh_document = refresh

    def _snapshot(self) -> LayeredDocument:
        return compose(
            self._setup.snapshot(), self._store.snapshot(), self._source, complete=False
        )

    @property
    def general(self) -> GeneralView:
        """Return this point's created_at, optional description and YAML ext view.

        Reads use the cached point snapshot. Writable fields commit through the
        same layered transaction as edit(); metadata does not fall back to setup.
        """
        return GeneralView(lambda: self._store.snapshot().general, self._edit_general)

    @contextmanager
    def _edit_general(self) -> Generator[PointGeneral]:
        with self._edit_document() as draft, stage_general(draft) as candidate:
            yield candidate

    @property
    def description(self) -> str | None:
        """Return the cached point description, or None when it is unset."""
        return self._store.snapshot().general.description

    @description.setter
    def description(self, value: str | None) -> None:
        """Commit text or None to this point through the edit() transaction."""
        with self.edit() as draft:
            draft.description = value

    def __getattr__(self, name: str) -> ComponentView:
        if name not in self._setup.snapshot().components:
            raise AttributeError(f"Unknown component {name!r}")
        return ComponentView(
            lambda: self._snapshot().components[name],
            lambda field: self._edit_component(name, field),
            name,
        )

    @contextmanager
    def _edit_component(self, name: str, field: str) -> Generator[ComponentSchema]:
        with (
            self._edit_document() as draft,
            stage_component(draft, name, field) as candidate,
        ):
            yield candidate

    @contextmanager
    def edit(self) -> Generator[EditView]:
        """Yield a working-unit draft and commit it on normal context exit.

        Reload both layers on entry. Existing component fields write to their
        owning layer; new fields and general metadata write to point. Validate
        required fields and references in every point's combined view before
        committing. A body exception, invalid value or ConflictError aborts all
        writes and keeps cached snapshots. Ordinary replace failure restores
        completed writes; failed restoration raises PartialCommitError with the
        affected paths. There is no multi-file power-loss guarantee.
        """
        with self._edit_document() as draft:
            yield EditView(draft)

    def refresh(self) -> None:
        """Reload this point and the shared setup snapshot from disk.

        Validate their combined required fields, references and canonical values.
        Missing files, incompatible headers, duplicate leaves or invalid values
        raise without publishing either new snapshot. No parameter files are committed.
        """
        self._refresh_document()

    def move(self, path: str, *, to: Literal["setup", "point"]) -> None:
        """Move a dotted component field and its source metadata between layers.

        path is a logical path such as Q1.t1 or Q1.wiring.flux_ch. to names the
        destination layer, opposite the layer holding the value. Nested mappings
        carry descendant metadata too. kind cannot move. Invalid paths, kind or
        an occupied destination raise ValueError; an unknown component or absent
        source value raises AttributeError. The move commits through edit() and
        inherits its validation, conflict and recovery behavior.
        """
        if to not in ("setup", "point"):
            raise ValueError(f"{to!r}: expected setup or point")
        with self._edit_document() as draft:
            draft.moves.append((path, to))
