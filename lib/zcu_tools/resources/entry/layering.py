"""Logical-layer composition and write placement, never disk patching or SI conversion."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from enum import Enum
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.document_store import FieldPath, UnitSpec

from .errors import LayerConflictError, UnknownFieldError
from .registry import component_registry
from .schema import (
    ComponentSchema,
    LayeredDocument,
    PointDocument,
    SetupDocument,
    canonical_errors,
    is_forward_minor,
)


class _Absent(Enum):
    VALUE = "absent"


def _fields(component: ComponentSchema) -> YamlMap:
    return TypeAdapter(YamlMap).validate_python(
        component.model_dump(exclude_unset=True)
    )


def _merge(setup: YamlMap, point: YamlMap, path: FieldPath, source: Path) -> YamlMap:
    result = deepcopy(setup)
    for name, value in point.items():
        if name in setup:
            original = setup[name]
            if isinstance(original, dict) and isinstance(value, dict):
                result[name] = _merge(original, value, (*path, name), source)
            else:
                raise LayerConflictError(
                    ".".join((*path, name)), source.parents[2] / "setup.yaml", source
                )
        else:
            result[name] = deepcopy(value)
    return result


def point_units(
    document: Mapping[str, YamlValue], setup: SetupDocument, source: Path
) -> Mapping[FieldPath, UnitSpec]:
    result: dict[FieldPath, UnitSpec] = {}
    components = document.get("components")
    if not isinstance(components, dict):
        return result
    for name in components:
        if name not in setup.components:
            raise AttributeError(f"{source}: unknown point component {name!r}")
        kind = setup.components[name].kind
        for path, unit in component_registry.units(kind).items():
            result[("components", name, *path)] = unit
    return result


def validate_point(document: PointDocument, setup: SetupDocument, source: Path) -> None:
    forward = is_forward_minor(
        {"format": document.format, "format_version": document.format_version},
        source=source,
    )
    for name, fields in document.components.items():
        if name not in setup.components:
            raise AttributeError(f"{source}: unknown point component {name!r}")
        if "kind" in fields:
            raise UnknownFieldError(f"{name}.kind", "kind", ())
        kind = setup.components[name].kind
        if not forward:
            component_registry.check_fields(kind, fields, path=name)
        supplied: YamlMap = {"kind": kind, **deepcopy(fields)}
        model = component_registry.partial_model(kind)
        validated = model.model_validate(supplied, extra="ignore" if forward else None)
        errors = canonical_errors(supplied, validated, (name,), source=source)
        if errors:
            raise ValidationError.from_exception_data(model.__name__, errors)
        canonical = _fields(validated)
        del canonical["kind"]
        document.components[name] = canonical


def compose(
    setup: SetupDocument, point: PointDocument, source: Path, *, complete: bool
) -> LayeredDocument:
    components: dict[str, ComponentSchema] = {}
    for name, base in setup.components.items():
        supplied = _merge(
            _fields(base), point.components.get(name, {}), (name,), source
        )
        model = (
            component_registry.get(base.kind)
            if complete
            else component_registry.partial_model(base.kind)
        )
        validated = model.model_validate(deepcopy(supplied))
        errors = canonical_errors(supplied, validated, (name,), source=source)
        if errors:
            raise ValidationError.from_exception_data(model.__name__, errors)
        components[name] = validated
    component_registry.validate_references(components, source=source)
    return LayeredDocument(
        general=point.general.model_copy(deep=True),
        components=components,
        provenance=deepcopy({**setup.provenance, **point.provenance}),
    )


def _differences(
    before: YamlMap, after: YamlMap, prefix: FieldPath = ()
) -> Iterator[tuple[FieldPath, YamlValue | _Absent]]:
    for name in dict.fromkeys((*before, *after)):
        original = before.get(name, _Absent.VALUE)
        candidate = after.get(name, _Absent.VALUE)
        if isinstance(original, dict) and isinstance(candidate, dict):
            yield from _differences(original, candidate, (*prefix, name))
        elif original != candidate:
            yield (*prefix, name), candidate


def _lookup(document: YamlMap, path: FieldPath) -> YamlValue | _Absent:
    value: YamlValue = document
    for name in path:
        if not isinstance(value, dict) or name not in value:
            return _Absent.VALUE
        value = value[name]
    return value


def _put(document: YamlMap, path: FieldPath, value: YamlValue | _Absent) -> None:
    node = document
    for name in path[:-1]:
        child = node.get(name)
        if not isinstance(child, dict):
            child = {}
            node[name] = child
        node = child
    if isinstance(value, _Absent):
        node.pop(path[-1], None)
    else:
        node[path[-1]] = deepcopy(value)


def route(
    before: LayeredDocument,
    after: LayeredDocument,
    setup: SetupDocument,
    point: PointDocument,
) -> None:
    point.general = after.general.model_copy(deep=True)
    for name, candidate in after.components.items():
        setup_fields = _fields(setup.components[name])
        point_fields = deepcopy(point.components.get(name, {}))
        for path, value in _differences(
            _fields(before.components[name]), _fields(candidate)
        ):
            destination = (
                setup_fields
                if not isinstance(_lookup(setup_fields, path), _Absent)
                else point_fields
            )
            _put(destination, path, value)
        setup.components[name] = component_registry.partial_model(
            setup.components[name].kind
        ).model_validate(setup_fields)
        if point_fields:
            point.components[name] = point_fields
