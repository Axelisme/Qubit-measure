"""Setup and transaction views over one typed document store.

Copy incoming values before assignment: notebook models can validate assignment.
"""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import overload

from pydantic import BaseModel, TypeAdapter

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.document_store import DocumentStore

from .registry import component_registry
from .schema import (
    ComponentSchema,
    LayeredDocument,
    PointGeneral,
    SetupDocument,
    SetupGeneral,
    WiringSchema,
    validate_component_name,
)

type _FieldNode = BaseModel | YamlMap


@contextmanager
def stage_component(
    draft: SetupDocument | LayeredDocument, name: str, field: str
) -> Generator[ComponentSchema]:
    """Yield an independent working-unit component candidate for a draft edit.

    draft is a mutable setup or layered draft. name identifies an existing
    component; an unknown name raises KeyError. field is the accepted relative
    dotted path within it, such as t1 or wiring.ch, used for source bookkeeping.
    Normal exit validates supplied fields, replaces draft's component, and clears
    cloned_from on that path and descendants even for an unchanged value.
    A body exception or validation failure leaves the draft and sources intact.
    No complete required-field check, layer routing or disk I/O occurs here.
    """
    candidate = draft.components[name].model_copy(deep=True)
    yield candidate
    # The registered partial model runs field validators, not full invariants.
    draft.components[name] = type(candidate).model_validate(
        candidate.model_dump(exclude_unset=True, warnings=False)
    )
    path = f"{name}.{field}"
    for key, metadata in draft.provenance.items():
        if key == path or key.startswith(f"{path}."):
            metadata.pop("cloned_from", None)


@overload
def stage_general(draft: SetupDocument) -> AbstractContextManager[SetupGeneral]: ...


@overload
def stage_general(draft: LayeredDocument) -> AbstractContextManager[PointGeneral]: ...


@contextmanager
def stage_general(
    draft: SetupDocument | LayeredDocument,
) -> Generator[SetupGeneral | PointGeneral]:
    """Yield independent metadata, accepting it into draft on normal exit.

    draft is a mutable setup or layered draft. The candidate is SetupGeneral for
    setup, PointGeneral for a layered point. Successful validation replaces only
    draft.general; a body exception or ValidationError leaves it unchanged.
    This does no I/O and does not enforce entry identity or cross-layer rules.
    """
    candidate = draft.general.model_copy(deep=True)
    yield candidate
    fields = candidate.model_dump(exclude_unset=True, warnings=False)
    if isinstance(draft, SetupDocument):
        draft.general = SetupGeneral.model_validate(fields)
    else:
        draft.general = PointGeneral.model_validate(fields)


class FieldView:
    _model: Callable[[], _FieldNode]
    _edit: Callable[[str], AbstractContextManager[_FieldNode]]
    _path: str

    def __init__(
        self,
        model: Callable[[], _FieldNode],
        edit: Callable[[str], AbstractContextManager[_FieldNode]],
        path: str,
    ) -> None:
        """Bind a field container to its snapshot and validated edit callbacks.

        model returns the readable container. edit(field_name) stages one write
        and raises on rejection without changing that snapshot. path is the
        logical container path, such as Q1.wiring, used in errors.
        """
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
        with self._edit(name) as draft:
            if isinstance(draft, BaseModel):
                component_registry.check_fields(
                    type(draft), {name: value}, path=self._path
                )
                setattr(draft, name, deepcopy(value))
            else:
                draft[name] = TypeAdapter(YamlValue).validate_python(value)


class GeneralView(FieldView):
    _general_model: Callable[[], SetupGeneral | PointGeneral]
    _extension: FieldView

    def __init__(
        self,
        model: Callable[[], SetupGeneral | PointGeneral],
        edit: Callable[[], AbstractContextManager[SetupGeneral | PointGeneral]],
    ) -> None:
        super().__init__(model, lambda _name: edit(), "general")
        self._general_model = model

        @contextmanager
        def edit_extension(_name: str) -> Generator[YamlMap]:
            with edit() as draft:
                yield draft.ext
                draft.model_fields_set.add("ext")

        self._extension = FieldView(lambda: model().ext, edit_extension, "general.ext")

    @property
    def ext(self) -> FieldView:
        return self._extension

    @property
    def description(self) -> str | None:
        return self._general_model().description

    @description.setter
    def description(self, value: str | None) -> None:
        self["description"] = value


class ComponentView:
    _model: Callable[[], ComponentSchema]
    _edit: Callable[[str], AbstractContextManager[ComponentSchema]]
    _path: str

    def __init__(
        self,
        model: Callable[[], ComponentSchema],
        edit: Callable[[str], AbstractContextManager[ComponentSchema]],
        path: str,
    ) -> None:
        """Bind a component snapshot to an atomic, validated field edit.

        model supplies the readable model. edit(relative_field_path) yields
        a detached candidate, validates it on normal exit, then accepts the write
        into its owning document. Rejection leaves the pre-write draft intact.
        path is the component name used in logical paths, for example Q1.
        Nested wiring/ext writes pass their relative dotted paths to edit.
        The callback owner controls persistence and transaction boundaries.
        """
        self._model = model
        self._edit = edit
        self._path = path

    @property
    def ext(self) -> FieldView:
        return FieldView(lambda: self._model().ext, self._edit_ext, f"{self._path}.ext")

    @contextmanager
    def _edit_ext(self, field: str) -> Generator[YamlMap]:
        with self._edit(f"ext.{field}") as draft:
            yield draft.ext
            draft.model_fields_set.add("ext")

    @property
    def wiring(self) -> FieldView:
        return FieldView(
            lambda: self._model().wiring, self._edit_wiring, f"{self._path}.wiring"
        )

    @contextmanager
    def _edit_wiring(self, field: str) -> Generator[WiringSchema]:
        with self._edit(f"wiring.{field}") as draft:
            yield draft.wiring
            draft.model_fields_set.add("wiring")

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
            with self._edit(name) as draft:
                component_registry.check_fields(
                    draft.kind, {name: value}, path=self._path
                )
                setattr(draft, name, deepcopy(value))


class EditView:
    def __init__(self, draft: SetupDocument | LayeredDocument) -> None:
        self._draft = draft

    @property
    def general(self) -> GeneralView:
        return GeneralView(lambda: self._draft.general, self._edit_general)

    @contextmanager
    def _edit_general(self) -> Generator[SetupGeneral | PointGeneral]:
        with stage_general(self._draft) as candidate:
            yield candidate

    def set(self, path: str, value: YamlValue) -> None:
        parts = path.split(".")
        if len(parts) < 2 or not all(parts):
            raise ValueError(f"{path!r}: expected a dotted field path")
        edit_root: AbstractContextManager[BaseModel]
        if parts[0] == "general":
            edit_root = self._edit_general()
        else:
            if parts[0] not in self._draft.components:
                raise AttributeError(f"Unknown component {parts[0]!r}")
            edit_root = self._edit_component(parts[0], ".".join(parts[1:]))
        with edit_root as root:
            node: _FieldNode = root
            for index, name in enumerate(parts[1:-1], start=1):
                parent_path = ".".join(parts[:index])
                if isinstance(node, BaseModel):
                    component_registry.check_fields(
                        type(node), {name: None}, path=parent_path
                    )
                    child = getattr(node, name)
                    node.model_fields_set.add(name)
                else:
                    child = node[name]
                if not isinstance(child, (BaseModel, dict)):
                    raise AttributeError(f"{parent_path}.{name}: not a field container")
                node = child

            @contextmanager
            def edit_node(_name: str) -> Generator[_FieldNode]:
                yield node

            FieldView(lambda: node, edit_node, ".".join(parts[:-1]))[parts[-1]] = value

    @property
    def description(self) -> str | None:
        return self._draft.general.description

    @description.setter
    def description(self, value: str | None) -> None:
        self.general.description = value

    def __getattr__(self, name: str) -> ComponentView:
        if name not in self._draft.components:
            raise AttributeError(f"Unknown component {name!r}")
        return ComponentView(
            lambda: self._draft.components[name],
            lambda field: self._edit_component(name, field),
            name,
        )

    @contextmanager
    def _edit_component(self, name: str, field: str) -> Generator[ComponentSchema]:
        with stage_component(self._draft, name, field) as candidate:
            yield candidate


class SetupView:
    def __init__(
        self,
        store: DocumentStore[SetupDocument],
        source: Path,
        edit: Callable[[], AbstractContextManager[SetupDocument]],
    ) -> None:
        self._store = store
        self._source = source
        self._edit_document = edit

    @property
    def general(self) -> GeneralView:
        return GeneralView(lambda: self._store.snapshot().general, self._edit_general)

    @contextmanager
    def _edit_general(self) -> Generator[SetupGeneral]:
        with self._edit_document() as draft, stage_general(draft) as candidate:
            yield candidate

    @property
    def description(self) -> str | None:
        return self._store.snapshot().general.description

    @description.setter
    def description(self, value: str | None) -> None:
        with self.edit() as draft:
            draft.description = value

    @contextmanager
    def edit(self) -> Generator[EditView]:
        with self._edit_document() as draft:
            yield EditView(draft)

    def refresh(self) -> None:
        self._store.refresh()

    def add_component(self, name: str, *, kind: str, **fields: YamlValue) -> None:
        validate_component_name(name, source=self._source)
        model = component_registry.partial_model(
            kind, source=self._source, component=name
        )
        component_registry.check_fields(kind, fields, path=name)
        with self._edit_document() as draft:
            if name in draft.components:
                raise ValueError(f"Component {name!r} already exists")
            draft.components[name] = model.model_validate({"kind": kind, **fields})

    def __getattr__(self, name: str) -> ComponentView:
        if name not in self._store.snapshot().components:
            raise AttributeError(f"Unknown component {name!r}")
        return ComponentView(
            lambda: self._store.snapshot().components[name],
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
