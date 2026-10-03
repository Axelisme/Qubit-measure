"""Setup and transaction views over one typed document store."""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from pydantic import TypeAdapter

from zcu_tools.format_version import YamlValue
from zcu_tools.resources.document_store import DocumentStore

from .registry import component_registry
from .schema import ComponentSchema, SetupDocument


class ComponentView:
    _model: Callable[[], ComponentSchema]
    _edit: Callable[[], AbstractContextManager[ComponentSchema]]

    def __init__(
        self,
        model: Callable[[], ComponentSchema],
        edit: Callable[[], AbstractContextManager[ComponentSchema]],
    ) -> None:
        self._model = model
        self._edit = edit

    @property
    def kind(self) -> str:
        return self._model().kind

    def __getattr__(self, name: str) -> YamlValue:
        return TypeAdapter(YamlValue).validate_python(getattr(self._model(), name))

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            with self._edit() as draft:
                setattr(draft, name, value)


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
    def __init__(self, store: DocumentStore[SetupDocument], source: Path) -> None:
        self._store = store
        self._source = source

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

    def add_component(self, name: str, *, kind: str, **fields: YamlValue) -> None:
        model = component_registry.get(kind, source=self._source, component=name)
        with self._store.edit() as draft:
            if name in draft.components:
                raise ValueError(f"Component {name!r} already exists")
            draft.components[name] = model.model_validate({"kind": kind, **fields})

    def __getattr__(self, name: str) -> ComponentView:
        if name not in self._store.snapshot().components:
            raise AttributeError(f"Unknown component {name!r}")
        return ComponentView(
            lambda: self._store.snapshot().components[name],
            lambda: self._edit_component(name),
        )

    @contextmanager
    def _edit_component(self, name: str) -> Generator[ComponentSchema]:
        with self._store.edit() as draft:
            yield draft.components[name]
