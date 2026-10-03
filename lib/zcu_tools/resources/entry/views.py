"""Setup and transaction views over one typed document store."""

from collections.abc import Generator
from contextlib import contextmanager

from zcu_tools.resources.document_store import DocumentStore

from .schema import SetupDocument


class EditView:
    def __init__(self, draft: SetupDocument) -> None:
        self._draft = draft

    @property
    def description(self) -> str | None:
        return self._draft.general.description

    @description.setter
    def description(self, value: str | None) -> None:
        self._draft.general.description = value


class SetupView:
    def __init__(self, store: DocumentStore[SetupDocument]) -> None:
        self._store = store

    @property
    def description(self) -> str | None:
        return self._store.snapshot().general.description

    @description.setter
    def description(self, value: str | None) -> None:
        with self.edit() as draft:
            draft.description = value

    @contextmanager
    def edit(self) -> Generator[EditView]:
        with self._store.edit() as draft:
            yield EditView(draft)

    def refresh(self) -> None:
        self._store.refresh()
