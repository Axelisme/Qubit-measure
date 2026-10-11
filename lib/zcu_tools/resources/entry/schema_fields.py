"""Declared field projection of registered component kinds for editors.

The projection reads only pydantic class declarations. It never builds
defaults, runs validators or model hooks, reads values or performs I/O.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from types import NoneType, UnionType
from typing import Annotated, Literal, TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .registry import component_registry
from .schema import PointDocument, SetupDocument

type FieldValueKind = Literal[
    "text", "integer", "number", "boolean", "choice", "mapping", "group", "yaml"
]


@dataclass(frozen=True)
class FieldSpec:
    """One declared field of a registered component kind.

    segments: path inside the component, one literal key per nesting level,
        for example ("wiring", "flux", "ch"). Each key is the field alias when
        the model declares one, otherwise the field name. Never empty and
        never the top-level kind field.
    value_kind: editor category of the annotation after removing None:
        "text" for str, "integer" for int, "number" for float, "boolean" for
        bool, "choice" for a Literal whose values are all str, "group" for a
        nested BaseModel whose fields follow as their own specs, "mapping" for
        dict or Mapping types, and "yaml" for everything else (lists, unions of
        several types, a model already being expanded), edited as YAML text
        and validated by the model.
    required: True when the model declares no default (pydantic is_required()).
        A field inside a group is required once that group is supplied.
    choices: the Literal values in declaration order when value_kind is
        "choice"; an empty tuple for every other value_kind.
    unit: the working-unit label from json_schema_extra["unit"], or None when
        the field declares none. A missing unit does not mean dimensionless.
    description: the pydantic Field description, or None when undeclared.
    reference_role: the registered role name when this is a top-level "text"
        field whose key equals a role registered without a via path; such a
        field names another component of that role's kind (see
        role_candidates). None for every other field.
    """

    segments: tuple[str, ...]
    value_kind: FieldValueKind
    required: bool
    choices: tuple[str, ...]
    unit: str | None
    description: str | None
    reference_role: str | None

    @property
    def path(self) -> str:
        """Return segments joined by dots, for example "wiring.flux.ch"."""
        return ".".join(self.segments)


def component_field_specs(kind: str) -> tuple[FieldSpec, ...]:
    """Project the registered model of kind into specs in declaration order.

    Expansion is depth first: a group spec comes right before the specs of its
    nested fields. A nested model already being expanded on the current path
    (a recursive model) projects as "yaml" instead of expanding again.
    Unknown kinds raise UnknownKindError from the component registry. Unit
    metadata that is not a nonempty string raises ValueError.
    """
    model = component_registry.get(kind)
    return tuple(_project(model, (), (model,)))


def role_candidates(
    document: SetupDocument | PointDocument, role: str
) -> tuple[str, ...]:
    """Return document component names whose kind matches role's kind pattern.

    Names keep document order. Unknown roles raise ValueError from the role
    registry. Only the given document is read; no I/O or validation runs.
    """
    pattern = component_registry.roles.get(role).kind
    return tuple(
        name
        for name, component in document.components.items()
        if fnmatchcase(component.kind, pattern)
    )


def _reference_role(
    prefix: tuple[str, ...], key: str, value_kind: FieldValueKind
) -> str | None:
    # Role resolution without via reads a top-level field named after the role.
    roles = component_registry.roles
    if prefix or value_kind != "text" or key not in roles.names():
        return None
    return key if roles.get(key).via is None else None


def _project(
    model: type[BaseModel],
    prefix: tuple[str, ...],
    expanding: tuple[type[BaseModel], ...],
) -> Iterator[FieldSpec]:
    for name, field in model.model_fields.items():
        if not prefix and name == "kind":
            continue
        segments = (*prefix, field.alias or name)
        annotation = _without_none(field.annotation)
        nested = _nested_model(annotation, expanding)
        value_kind = "group" if nested is not None else _value_kind(annotation)
        yield FieldSpec(
            segments=segments,
            value_kind=value_kind,
            required=field.is_required(),
            choices=_choices(annotation) if value_kind == "choice" else (),
            unit=declared_unit(field, ".".join(segments)),
            description=field.description,
            reference_role=_reference_role(prefix, segments[-1], value_kind),
        )
        if nested is not None:
            yield from _project(nested, segments, (*expanding, nested))


def _unwrap(annotation: object) -> object:
    """Resolve type aliases and Annotated wrappers to the underlying type."""
    while True:
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
        elif get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        else:
            return annotation


def _without_none(annotation: object) -> object:
    """Drop None from an optional union; other unions stay unchanged."""
    annotation = _unwrap(annotation)
    if get_origin(annotation) not in (Union, UnionType):
        return annotation
    members = [arg for arg in get_args(annotation) if arg is not NoneType]
    return _unwrap(members[0]) if len(members) == 1 else annotation


def _nested_model(
    annotation: object, expanding: tuple[type[BaseModel], ...]
) -> type[BaseModel] | None:
    if (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and annotation not in expanding
    ):
        return annotation
    return None


def _value_kind(annotation: object) -> FieldValueKind:
    scalar_kinds: tuple[tuple[type, FieldValueKind], ...] = (
        (str, "text"),
        (bool, "boolean"),
        (int, "integer"),
        (float, "number"),
    )
    for scalar, kind in scalar_kinds:
        if annotation is scalar:
            return kind
    if get_origin(annotation) is Literal and all(
        isinstance(value, str) for value in get_args(annotation)
    ):
        return "choice"
    container = get_origin(annotation) or annotation
    if isinstance(container, type) and issubclass(container, Mapping):
        return "mapping"
    return "yaml"


def _choices(annotation: object) -> tuple[str, ...]:
    return tuple(str(value) for value in get_args(annotation))


def declared_unit(field: FieldInfo, path: str) -> str | None:
    """Return a field's json_schema_extra["unit"] label, or None when undeclared.

    path only names the field in errors. A unit that is not a nonempty string
    raises ValueError. This is the one unit rule for the projection and for
    document_unit; no value is read and no conversion happens.
    """
    metadata = field.json_schema_extra
    if not isinstance(metadata, dict) or "unit" not in metadata:
        return None
    unit = metadata["unit"]
    if not isinstance(unit, str) or not unit:
        raise ValueError(f"{path!r}: unit must be a nonempty string")
    return unit
