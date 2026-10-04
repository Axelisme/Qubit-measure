"""Component model declarations, independent of experiment definitions."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin

from pydantic import (
    BaseModel,
    TypeAdapter,
    create_model,
)
from pydantic.fields import FieldInfo

from zcu_tools.format_version import YamlMap
from zcu_tools.resources.document_store import FieldPath, UnitSpec

from .errors import MissingReferenceError, UnknownFieldError, UnknownKindError
from .schema import (
    ComponentSchema,
    CurrentSourceSchema,
    FluxoniumSchema,
    JpaSchema,
    QubitSchema,
    ResonatorSchema,
    WiringSchema,
    field_annotations,
)


def _nested_model(annotation: object) -> type[BaseModel] | None:
    if get_origin(annotation) in (Union, UnionType):
        alternatives = tuple(
            item for item in get_args(annotation) if item is not type(None)
        )
        if len(alternatives) != 1:
            return None
        annotation = alternatives[0]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _model_units(model: type[BaseModel]) -> dict[FieldPath, UnitSpec]:
    result: dict[FieldPath, UnitSpec] = {}
    for name, field in model.model_fields.items():
        types, field_metadata = field_annotations(field)
        numeric = any(isinstance(item, type) and item in (float, int) for item in types)
        for metadata in field_metadata:
            if isinstance(metadata, UnitSpec):
                if not numeric or not all(
                    isinstance(item, type) and item in (float, int, type(None))
                    for item in types
                ):
                    raise TypeError(f"Unit metadata requires a numeric field: {name}")
                metadata.validate()
                if (name,) in result and result[(name,)] != metadata:
                    raise ValueError(f"Conflicting UnitSpec declarations: {name}")
                result[(name,)] = metadata
        wiring_field = WiringSchema.model_fields.get(name)
        channel_index = (
            issubclass(model, WiringSchema)
            and wiring_field is not None
            and not any(
                isinstance(metadata, UnitSpec) for metadata in wiring_field.metadata
            )
            and all(
                isinstance(item, type) and item in (int, type(None)) for item in types
            )
        )
        if numeric and (name,) not in result and not channel_index:
            raise ValueError(
                f"UnitSpec required for numeric field: {model.__name__}.{name}"
            )
        nested_model = _nested_model(field.annotation)
        if nested_model is not None:
            for path, spec in _model_units(nested_model).items():
                result[(name, *path)] = spec
    return result


def _partial_model[_Model: BaseModel](
    model: type[_Model], *, component_root: bool = False
) -> type[_Model]:
    """Relax missing fields while retaining supplied-field decorators and constraints.

    Registered models contain no model-level validators. Defaults and factories
    retain Pydantic semantics, but missing fields do not validate their defaults.
    Only this derived model changes; registry.get returns the original model.
    """
    fields: dict[str, tuple[object, FieldInfo]] = {}
    for name, field in model.model_fields.items():
        annotation = field.annotation
        nested_model = _nested_model(annotation)
        if nested_model is not None:
            partial_nested = _partial_model(nested_model)
            annotation = (
                partial_nested | None
                if type(None) in get_args(annotation)
                else partial_nested
            )
        partial_field = deepcopy(field)
        # D101: only supplied fields run validators in a partial document.
        partial_field.validate_default = False
        if field.is_required() and not (component_root and name == "kind"):
            # Missing values stay outside model_fields_set and serialized patches.
            # Supplied values retain the original non-nullable annotation.
            partial_field.default = None
        fields[name] = (annotation, partial_field)
    # Pydantic mixes field definitions and reserved options in one kwargs signature.
    return create_model(
        f"{model.__name__}Partial", __base__=model, **cast(dict[str, Any], fields)
    )


def _validate_partial_contract(model: type[BaseModel]) -> None:
    for validator in model.__pydantic_decorators__.model_validators.values():
        raise ValueError(
            f"{model.__name__}: {validator.info.mode} model validator is unsupported "
            "in partial setup; cross-field checks are not supported in this batch"
        )
    if model.model_post_init is not ComponentSchema.model_post_init:
        raise ValueError(
            f"{model.__name__}: custom model_post_init is unsupported in partial setup"
        )
    for field in model.model_fields.values():
        nested_model = _nested_model(field.annotation)
        if nested_model is not None:
            _validate_partial_contract(nested_model)


def _validate_component_model(model: object) -> None:
    if not isinstance(model, type) or not issubclass(model, ComponentSchema):
        raise TypeError("Registered models must derive from ComponentSchema")
    if model.model_config.get("extra") != "forbid":
        raise TypeError("Registered component models must use extra=forbid")


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
        nested_model = _nested_model(annotation)
        if nested_model is None:
            break
        model = nested_model
    raise ValueError(f"Invalid component reference path: {reference!r}")


class ComponentRegistry:
    def __init__(self) -> None:
        self._models: dict[str, type[ComponentSchema]] = {}
        self._references: dict[str, tuple[str, ...]] = {}
        self._units: dict[str, dict[FieldPath, UnitSpec]] = {}
        self._partial_models: dict[str, type[ComponentSchema]] = {}

    def register(
        self, kind: str, model: type[ComponentSchema], *, references: Sequence[str] = ()
    ) -> None:
        """Register a unique document kind and its component schema.

        ``model`` must extend ComponentSchema with extra=forbid and UnitSpec on
        physical numeric fields. ``references`` contains component-reference
        field paths, including nested paths. All model-level validators and custom
        model_post_init are rejected, including inherited and direct or nullable
        nested declarations. Cross-field checks are unsupported in this batch;
        field validators retain their conversions and canonical constraints.
        Defaults and default_factory retain Pydantic semantics.

        Raise ValueError for duplicates, invalid references, missing units or
        unsupported model hooks; raise TypeError for invalid model or unit
        declarations. Failure does not reserve kind or execute a trial model.
        """
        if kind in self._models:
            raise ValueError(f"Kind {kind!r} is already registered")
        _validate_component_model(model)
        _validate_partial_contract(model)
        reference_paths = tuple(references)
        for reference in reference_paths:
            _validate_reference(model, reference)
        units = _model_units(model)
        partial_model = _partial_model(model, component_root=True)
        self._models[kind] = model
        self._references[kind] = reference_paths
        self._units[kind] = units
        self._partial_models[kind] = partial_model

    def unregister(self, kind: str) -> None:
        del self._models[kind]
        del self._references[kind]
        del self._units[kind]
        del self._partial_models[kind]

    def get(
        self, kind: str, *, source: Path | None = None, component: str | None = None
    ) -> type[ComponentSchema]:
        try:
            return self._models[kind]
        except KeyError as cause:
            raise UnknownKindError(
                source, component, kind, tuple(get_close_matches(kind, self._models))
            ) from cause

    def partial_model(
        self, kind: str, *, source: Path | None = None, component: str | None = None
    ) -> type[ComponentSchema]:
        """Validate supplied setup values while deferring required-field completeness."""
        self.get(kind, source=source, component=component)
        return self._partial_models[kind]

    def validate_complete(
        self, kind: str, fields: YamlMap, *, source: Path, component: str
    ) -> ComponentSchema:
        """Validate merged working-unit fields with the original complete model.

        ``component`` prefixes field errors; ``source`` locates unknown-kind
        diagnostics. Required values and field validators use Pydantic semantics.
        Raise ValidationError for invalid fields or missing required values.
        """
        model = self.get(kind, source=source, component=component)
        adapter = TypeAdapter[dict[str, ComponentSchema]](dict[str, model])
        return adapter.validate_python({component: fields})[component]

    def check_fields(
        self, kind: str | type[BaseModel], fields: Mapping[str, object], *, path: str
    ) -> None:
        known_fields = (self.get(kind) if isinstance(kind, str) else kind).model_fields
        for name, value in fields.items():
            if name not in known_fields:
                raise UnknownFieldError(
                    f"{path}.{name}", name, tuple(get_close_matches(name, known_fields))
                )
            nested_model = _nested_model(known_fields[name].annotation)
            if isinstance(value, dict) and nested_model is not None:
                self.check_fields(nested_model, value, path=f"{path}.{name}")

    def validate_references(
        self, components: Mapping[str, ComponentSchema], *, source: Path
    ) -> None:
        """Reject supplied references that do not name a component in this document."""
        for name, component in components.items():
            self.get(component.kind, source=source, component=name)
            for field in self._references[component.kind]:
                target: object = component
                for part in field.split("."):
                    target = (
                        getattr(target, part, None)
                        if isinstance(target, BaseModel)
                        else None
                    )
                if isinstance(target, str) and target not in components:
                    raise MissingReferenceError(source, name, field, target)

    def units(
        self, kind: str, *, source: Path | None = None, component: str | None = None
    ) -> Mapping[FieldPath, UnitSpec]:
        self.get(kind, source=source, component=component)
        return dict(self._units[kind])


component_registry = ComponentRegistry()
component_registry.register("resonator", ResonatorSchema, references=("amplifier",))
component_registry.register("device/current_source", CurrentSourceSchema)
component_registry.register("amplifier/jpa", JpaSchema)
component_registry.register(
    "qubit/fluxonium", FluxoniumSchema, references=("readout", "flux_source")
)
component_registry.register(
    "qubit/transmon", QubitSchema, references=("readout", "flux_source")
)
