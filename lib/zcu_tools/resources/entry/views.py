"""Setup and transaction views over one typed document store."""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.document_store import DocumentStore

from .registry import component_registry
from .schema import (
    ComponentSchema,
    SetupDocument,
    SetupGeneral,
    WiringSchema,
    validate_component_name,
)

type _FieldNode = BaseModel | YamlMap


class FieldView:
    _model: Callable[[], _FieldNode]
    _edit: Callable[[], AbstractContextManager[_FieldNode]]
    _path: str

    def __init__(
        self,
        model: Callable[[], _FieldNode],
        edit: Callable[[], AbstractContextManager[_FieldNode]],
        path: str,
    ) -> None:
        self._model = model
        self._edit = edit
        self._path = path

    def __getattr__(self, name: str) -> YamlValue:
        try:
            return self[name]
        except KeyError as cause:
            raise AttributeError(f"{self._path}.{name}: field is not set") from cause

    def __getitem__(self, name: str) -> YamlValue:
        model = self._model()
        if isinstance(model, BaseModel):
            component_registry.check_fields(type(model), {name: None}, path=self._path)
            if name not in model.model_fields_set:
                raise AttributeError(f"{self._path}.{name}: field is not set")
            return TypeAdapter(YamlValue).validate_python(getattr(model, name))
        return model[name]

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __setitem__(self, name: str, value: object) -> None:
        with self._edit() as draft:
            if isinstance(draft, BaseModel):
                component_registry.check_fields(
                    type(draft), {name: value}, path=self._path
                )
                values = draft.model_dump(exclude_unset=True)
                values[name] = value
                validated = type(draft).model_validate(values)
                setattr(draft, name, getattr(validated, name))
            else:
                draft[name] = TypeAdapter(YamlValue).validate_python(value)


class GeneralView(FieldView):
    _extension: FieldView

    def __init__(
        self,
        model: Callable[[], SetupGeneral],
        edit: Callable[[], AbstractContextManager[SetupGeneral]],
    ) -> None:
        super().__init__(model, edit, "general")

        @contextmanager
        def edit_extension() -> Generator[YamlMap]:
            with edit() as draft:
                yield draft.ext

        self._extension = FieldView(lambda: model().ext, edit_extension, "general.ext")

    @property
    def ext(self) -> FieldView:
        return self._extension


class ComponentView:
    _model: Callable[[], ComponentSchema]
    _edit: Callable[[], AbstractContextManager[ComponentSchema]]
    _path: str

    def __init__(
        self,
        model: Callable[[], ComponentSchema],
        edit: Callable[[], AbstractContextManager[ComponentSchema]],
        path: str,
    ) -> None:
        self._model = model
        self._edit = edit
        self._path = path

    @property
    def ext(self) -> FieldView:
        return FieldView(lambda: self._model().ext, self._edit_ext, f"{self._path}.ext")

    @contextmanager
    def _edit_ext(self) -> Generator[YamlMap]:
        with self._edit() as draft:
            yield draft.ext

    @property
    def wiring(self) -> FieldView:
        return FieldView(
            lambda: self._model().wiring, self._edit_wiring, f"{self._path}.wiring"
        )

    @contextmanager
    def _edit_wiring(self) -> Generator[WiringSchema]:
        with self._edit() as draft:
            yield draft.wiring

    @property
    def kind(self) -> str:
        return self._model().kind

    def __getattr__(self, name: str) -> YamlValue:
        model = self._model()
        component_registry.check_fields(model.kind, {name: None}, path=self._path)
        if name not in model.model_fields_set:
            raise AttributeError(f"{self._path}.{name}: field is not set")
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            value = value.model_dump(exclude_unset=True)
        return TypeAdapter(YamlValue).validate_python(value)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            with self._edit() as draft:
                component_registry.check_fields(
                    draft.kind, {name: value}, path=self._path
                )
                values = draft.model_dump(exclude_unset=True)
                values[name] = value
                validated = type(draft).model_validate(values)
                setattr(draft, name, getattr(validated, name))


class EditView:
    def __init__(self, draft: SetupDocument) -> None:
        self._draft = draft

    @property
    def general(self) -> GeneralView:
        return GeneralView(lambda: self._draft.general, self._edit_general)

    @contextmanager
    def _edit_general(self) -> Generator[SetupGeneral]:
        yield self._draft.general

    def set(self, path: str, value: YamlValue) -> None:
        parts = path.split(".")
        if len(parts) < 2 or not all(parts):
            raise ValueError(f"{path!r}: expected a dotted field path")
        node: _FieldNode
        if parts[0] == "general":
            node = self._draft.general
        else:
            if parts[0] not in self._draft.components:
                raise AttributeError(f"Unknown component {parts[0]!r}")
            node = self._draft.components[parts[0]]
        for index, name in enumerate(parts[1:-1], start=1):
            parent_path = ".".join(parts[:index])
            if isinstance(node, BaseModel):
                component_registry.check_fields(
                    type(node), {name: None}, path=parent_path
                )
                child = getattr(node, name)
            else:
                child = node[name]
            if not isinstance(child, (BaseModel, dict)):
                raise AttributeError(f"{parent_path}.{name}: not a field container")
            node = child

        @contextmanager
        def edit_node() -> Generator[_FieldNode]:
            yield node

        FieldView(lambda: node, edit_node, ".".join(parts[:-1]))[parts[-1]] = value

    @property
    def description(self) -> str | None:
        return self._draft.general.description

    @description.setter
    def description(self, value: str | None) -> None:
        self._draft.general.description = value

    def __getattr__(self, name: str) -> ComponentView:
        if name not in self._draft.components:
            raise AttributeError(f"Unknown component {name!r}")
        return ComponentView(
            lambda: self._draft.components[name],
            lambda: self._edit_component(name),
            name,
        )

    @contextmanager
    def _edit_component(self, name: str) -> Generator[ComponentSchema]:
        yield self._draft.components[name]


class SetupView:
    def __init__(self, store: DocumentStore[SetupDocument], source: Path) -> None:
        self._store = store
        self._source = source

    @property
    def general(self) -> GeneralView:
        return GeneralView(lambda: self._store.snapshot().general, self._edit_general)

    @contextmanager
    def _edit_general(self) -> Generator[SetupGeneral]:
        with self._store.edit() as draft:
            yield draft.general

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
        validate_component_name(name, source=self._source)
        model = component_registry.partial_model(
            kind, source=self._source, component=name
        )
        component_registry.check_fields(kind, fields, path=name)
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
            name,
        )

    @contextmanager
    def _edit_component(self, name: str) -> Generator[ComponentSchema]:
        with self._store.edit() as draft:
            yield draft.components[name]
