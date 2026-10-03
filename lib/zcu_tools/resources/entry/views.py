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
        raise NotImplementedError("Draft description access is not implemented")

    @description.setter
    def description(self, value: str | None) -> None:
        raise NotImplementedError("Draft description writes are not implemented")


class SetupView:
    def __init__(self, store: DocumentStore[SetupDocument]) -> None:
        self._store = store

    @property
    def description(self) -> str | None:
        raise NotImplementedError("Setup description access is not implemented")

    @description.setter
    def description(self, value: str | None) -> None:
        raise NotImplementedError("Setup description writes are not implemented")

    @contextmanager
    def edit(self) -> Generator[EditView]:
        with self._store.edit() as draft:
            yield EditView(draft)

    def refresh(self) -> None:
        self._store.refresh()
