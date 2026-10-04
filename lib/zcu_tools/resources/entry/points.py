"""Working-point views over entry-coordinated layered transactions."""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Literal

from zcu_tools.resources.document_store import DocumentStore

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
    def __init__(
        self,
        store: DocumentStore[PointDocument],
        setup: DocumentStore[SetupDocument],
        source: Path,
        edit: Callable[[], AbstractContextManager[LayeredDocument]],
        refresh: Callable[[], None],
    ) -> None:
        self._store = store
        self._setup = setup
        self._source = source
        self._edit_document = edit
        self._refresh_document = refresh

    def _snapshot(self) -> LayeredDocument:
        return compose(
            self._setup.snapshot(), self._store.snapshot(), self._source, complete=False
        )

    @property
    def general(self) -> GeneralView:
        return GeneralView(lambda: self._store.snapshot().general, self._edit_general)

    @contextmanager
    def _edit_general(self) -> Generator[PointGeneral]:
        with self._edit_document() as draft, stage_general(draft) as candidate:
            yield candidate

    @property
    def description(self) -> str | None:
        return self._store.snapshot().general.description

    @description.setter
    def description(self, value: str | None) -> None:
        with self.edit() as draft:
            draft.description = value

    def __getattr__(self, name: str) -> ComponentView:
        if name not in self._setup.snapshot().components:
            raise AttributeError(f"Unknown component {name!r}")
        return ComponentView(
            lambda: self._snapshot().components[name],
            lambda: self._edit_component(name),
            name,
        )

    @contextmanager
    def _edit_component(self, name: str) -> Generator[ComponentSchema]:
        with self._edit_document() as draft, stage_component(draft, name) as candidate:
            yield candidate

    @contextmanager
    def edit(self) -> Generator[EditView]:
        with self._edit_document() as draft:
            yield EditView(draft)

    def clone_source(self, entry: Path) -> Path:
        """Resources-only identity check for an entry-owned clone."""
        if self._source.resolve().parents[2] != entry.resolve():
            raise ValueError("Cross-entry cloning is not supported")
        return self._source.parent

    def refresh(self) -> None:
        self._refresh_document()

    def move(self, path: str, *, to: Literal["setup", "point"]) -> None:
        if to not in ("setup", "point"):
            raise ValueError(f"{to!r}: expected setup or point")
        with self._edit_document() as draft:
            draft.moves.append((path, to))
