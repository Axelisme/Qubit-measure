"""Logical-layer composition and write placement, never disk patching or SI conversion."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Literal

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
    # YAML null is a value, so routing needs a separate marker for an absent leaf.
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
    """Return store unit paths for the point's setup-declared component names.

    document is the raw YAML tree in either SI or working units; values are not
    inspected or changed. setup supplies the registered kind of each component.
    source is point.yaml for diagnostics. Paths begin with components/<name> and
    include declared fields even when absent. Missing/non-mapping components
    returns an empty map; an unknown component raises AttributeError.
    """
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
    """Validate and replace supplied point fields with canonical working values.

    document is a working-unit point draft; setup supplies kinds for its names.
    source is point.yaml for header and field diagnostics. Required fields may
    remain absent. Unknown names raise AttributeError; point kind and unknown
    current-minor fields raise UnknownFieldError. Invalid values or canonical
    drift raise ValidationError; field-validator exceptions propagate.
    Newer supported minor fields are omitted from this typed projection; the
    store retains their raw YAML. A failure may leave earlier components updated,
    so callers must discard the draft. No disk I/O or snapshot publication occurs.
    """
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
    """Return an independent working-unit view combining validated layer drafts.

    setup supplies component declarations; point supplies point-local values and
    metadata. source must locate entry/points/<label>/point.yaml for diagnostics.
    Reject any leaf present in both layers with LayerConflictError. complete=True
    validates required fields against original registered models; False permits
    missing fields for editing. Both modes validate references and canonical
    values, raising their validation errors. Inputs are not changed and no I/O
    occurs. The returned draft has copied provenance and no pending moves.
    """
    components: dict[str, ComponentSchema] = {}
    for name, base in setup.components.items():
        supplied = _merge(
            _fields(base), point.components.get(name, {}), (name,), source
        )
        if complete:
            validated = component_registry.validate_complete(
                base.kind,
                deepcopy(supplied),
                source=source,
                component=name,
            )
        else:
            model = component_registry.partial_model(base.kind)
            validated = model.model_validate(deepcopy(supplied))
        errors = canonical_errors(supplied, validated, (name,), source=source)
        if errors:
            raise ValidationError.from_exception_data(type(validated).__name__, errors)
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


def _move(
    path: str,
    to: Literal["setup", "point"],
    setup: SetupDocument,
    point: PointDocument,
) -> None:
    parts = tuple(path.split("."))
    if len(parts) < 2 or not all(parts):
        raise ValueError(f"{path!r}: expected a dotted component field path")
    name, *field_parts = parts
    if name not in setup.components:
        raise AttributeError(f"Unknown component {name!r}")
    if field_parts[0] == "kind":
        raise ValueError(f"{path}: kind belongs to setup")
    setup_fields = _fields(setup.components[name])
    point_fields = deepcopy(point.components.get(name, {}))
    source_fields, destination_fields = (
        (point_fields, setup_fields) if to == "setup" else (setup_fields, point_fields)
    )
    field_path = tuple(field_parts)
    if not isinstance(_lookup(destination_fields, field_path), _Absent):
        raise ValueError(f"{path}: destination {to} already has a value")
    value = _lookup(source_fields, field_path)
    if isinstance(value, _Absent):
        raise AttributeError(f"{path}: field is not set in the source layer")
    _put(destination_fields, field_path, value)
    _put(source_fields, field_path, _Absent.VALUE)
    setup.components[name] = component_registry.partial_model(
        setup.components[name].kind
    ).model_validate(setup_fields)
    point.components[name] = point_fields
    source_meta, destination_meta = (
        (point.provenance, setup.provenance)
        if to == "setup"
        else (setup.provenance, point.provenance)
    )
    for key in tuple(source_meta):
        if key == path or key.startswith(f"{path}."):
            destination_meta[key] = source_meta.pop(key)


def route(
    before: LayeredDocument,
    after: LayeredDocument,
    setup: SetupDocument,
    point: PointDocument,
) -> None:
    """Apply a layered draft's edits to mutable working-unit layer drafts.

    before is the original combined view; after is its edited independent copy.
    setup and point must represent the same original layers. Existing fields
    retain their layer; new fields and general metadata go to point. Accepted
    source changes keep their source layer even without a value diff, such as a
    same-value rewrite clearing cloned_from. Apply after.moves last, carrying
    values and descendant provenance together.
    Invalid moves raise ValueError or AttributeError; partial-model validation
    errors propagate. No disk I/O occurs and no rollback is provided here:
    callers must discard both mutated drafts on failure.
    """
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
    for path, metadata in after.provenance.items():
        if metadata != before.provenance.get(path):
            provenance = (
                point.provenance if path in point.provenance else setup.provenance
            )
            provenance[path] = deepcopy(metadata)
    for path, destination in after.moves:
        _move(path, destination, setup, point)
