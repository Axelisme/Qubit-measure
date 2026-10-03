"""Component model declarations, independent of experiment definitions."""

from collections.abc import Mapping, Sequence
from difflib import get_close_matches
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin

from pydantic import BaseModel

from zcu_tools.resources.document_store import FieldPath, UnitSpec

from .errors import UnknownFieldError, UnknownKindError
from .schema import ComponentSchema, ResonatorSchema


def _model_units(model: type[BaseModel]) -> dict[FieldPath, UnitSpec]:
    result: dict[FieldPath, UnitSpec] = {}
    for name, field in model.model_fields.items():
        for metadata in field.metadata:
            if isinstance(metadata, UnitSpec):
                annotation = field.annotation
                types = (
                    get_args(annotation)
                    if get_origin(annotation) in (Union, UnionType)
                    else (annotation,)
                )
                if not all(item in (float, int, type(None)) for item in types):
                    raise TypeError(f"Unit metadata requires a numeric field: {name}")
                metadata.validate()
                result[(name,)] = metadata
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            for path, spec in _model_units(annotation).items():
                result[(name, *path)] = spec
    return result


def _validate_component_model(model: object) -> None:
    if not isinstance(model, type) or not issubclass(model, ComponentSchema):
        raise TypeError("Registered models must derive from ComponentSchema")


def _validate_reference(model: type[BaseModel], reference: str) -> None:
    parts = reference.split(".")
    for index, name in enumerate(parts):
        field = model.model_fields.get(name)
        if field is None:
            break
        annotation = field.annotation
        if index == len(parts) - 1:
            types = tuple(
                item for item in get_args(annotation) if item is not type(None)
            )
            if annotation is str or (
                get_origin(annotation) in (Union, UnionType) and types == (str,)
            ):
                return
            break
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            model = annotation
        else:
            break
    raise ValueError(f"Invalid component reference path: {reference!r}")


class ComponentRegistry:
    def __init__(self) -> None:
        self._models: dict[str, type[ComponentSchema]] = {}
        self._references: dict[str, tuple[str, ...]] = {}
        self._units: dict[str, dict[FieldPath, UnitSpec]] = {}

    def register(
        self, kind: str, model: type[ComponentSchema], *, references: Sequence[str] = ()
    ) -> None:
        if kind in self._models:
            raise ValueError(f"Kind {kind!r} is already registered")
        _validate_component_model(model)
        reference_paths = tuple(references)
        for reference in reference_paths:
            _validate_reference(model, reference)
        units = _model_units(model)
        self._models[kind] = model
        self._references[kind] = reference_paths
        self._units[kind] = units

    def unregister(self, kind: str) -> None:
        del self._models[kind]
        del self._references[kind]
        del self._units[kind]

    def get(
        self, kind: str, *, source: Path | None = None, component: str | None = None
    ) -> type[ComponentSchema]:
        try:
            return self._models[kind]
        except KeyError as cause:
            raise UnknownKindError(
                source, component, kind, tuple(get_close_matches(kind, self._models))
            ) from cause

    def check_fields(
        self, kind: str | type[BaseModel], fields: Mapping[str, object], *, path: str
    ) -> None:
        known_fields = (self.get(kind) if isinstance(kind, str) else kind).model_fields
        for name, value in fields.items():
            if name not in known_fields:
                raise UnknownFieldError(
                    f"{path}.{name}", name, tuple(get_close_matches(name, known_fields))
                )
            annotation = known_fields[name].annotation
            if (
                isinstance(value, dict)
                and isinstance(annotation, type)
                and issubclass(annotation, BaseModel)
            ):
                self.check_fields(annotation, value, path=f"{path}.{name}")

    def units(
        self, kind: str, *, source: Path | None = None, component: str | None = None
    ) -> Mapping[FieldPath, UnitSpec]:
        self.get(kind, source=source, component=component)
        return dict(self._units[kind])


component_registry = ComponentRegistry()
component_registry.register("resonator", ResonatorSchema)
