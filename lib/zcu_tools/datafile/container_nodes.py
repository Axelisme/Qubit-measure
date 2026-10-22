"""Container wire validation shared by one-shot and append I/O."""

from __future__ import annotations

import json
import re

import h5py as h5
import numpy as np
from pydantic import TypeAdapter

from .container_models import (
    Axis,
    ContainerAttrs,
    FormatError,
    JsonObject,
    JsonScalar,
    Variable,
    VariableSchema,
)
from .native_nodes import known_node

_JSON = TypeAdapter(JsonObject)
_SCALAR = TypeAdapter(JsonScalar)
_RESERVED = {"format", "format_version", "producer", "kind"}


def name_segment(name: str) -> None:
    """Raise ValueError unless name is a nonempty HDF5 path segment."""
    if not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise ValueError(f"invalid name: {name!r}")


def version_minor(version: str, major: int = 3) -> int:
    """Validate major.minor and return its minor, raising FormatError otherwise."""
    match = re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version)
    if match is None or int(match[1]) != major:
        raise FormatError(
            f"unsupported format_version {version!r}; expected major {major}"
        )
    return int(match[2])


def json_text(value: JsonObject) -> str:
    """Encode a strict finite JSON object; invalid values raise ValueError."""
    _JSON.validate_python(value, strict=True)
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def read_doc(node: h5.Group, name: str) -> JsonObject:
    """Read a local UTF-8 scalar JSON object or raise FormatError."""
    dataset = known_node(node, name)
    if not isinstance(dataset, h5.Dataset) or dataset.shape != ():
        raise FormatError(f"{node.name}/{name}: expected scalar JSON dataset")
    try:
        text = dataset.asstr()[()]
        if not isinstance(text, str):
            raise FormatError(f"{dataset.name}: expected JSON text")
        value = _JSON.validate_json(text, strict=True)
        json_text(value)
    except (ValueError, TypeError) as error:
        raise FormatError(f"{dataset.name}: invalid finite JSON: {error}") from error
    return value


def write_doc(node: h5.Group, name: str, value: JsonObject) -> None:
    """Replace the whole named document after validating its finite JSON."""
    name_segment(name)
    text = json_text(value)
    if name in node:
        del node[name]
    node.create_dataset(name, data=text, dtype=h5.string_dtype("utf-8"))


def text_attr(node: h5.Group | h5.Dataset, name: str) -> str:
    """Read a UTF-8 string attribute; missing/nontext attributes raise FormatError."""
    value = node.attrs.get(name)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise FormatError(f"{node.name}: {name}: expected text attribute")
    return value


def root_attrs(file: h5.File) -> ContainerAttrs:
    """Read a major-3 header and finite scalar extensions or raise FormatError."""
    if text_attr(file, "format") != "zcu.container":
        raise FormatError("expected format zcu.container")
    version = text_attr(file, "format_version")
    version_minor(version)
    extra: dict[str, JsonScalar] = {}
    for key, value in file.attrs.items():
        if key in _RESERVED:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, h5.Empty):
            value = None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        scalar = _SCALAR.validate_python(value, strict=True)
        json.dumps(scalar, allow_nan=False)
        extra[key] = scalar
    return ContainerAttrs(
        version,
        text_attr(file, "producer"),
        text_attr(file, "kind") if "kind" in file.attrs else None,
        extra,
    )


def write_attrs(file: h5.File, attrs: ContainerAttrs) -> None:
    """Write validated major-3 root attributes, refusing reserved extra keys."""
    version_minor(attrs.format_version)
    file.attrs.update(
        format="zcu.container",
        format_version=attrs.format_version,
        producer=attrs.producer,
    )
    if attrs.kind is not None:
        file.attrs["kind"] = attrs.kind
    for key, value in attrs.extra.items():
        if "\0" in key:
            raise FormatError("root attribute name contains NUL")
        if key in _RESERVED:
            raise FormatError(f"reserved root attribute: {key}")
        scalar = _SCALAR.validate_python(value, strict=True)
        json.dumps(scalar, allow_nan=False)
        file.attrs[key] = h5.Empty("f") if scalar is None else scalar


def validate_variable(variable: Variable) -> None:
    """Validate numeric dtype, ordered axes and shape without changing arrays."""
    shape = variable.values.shape
    if variable.values.dtype.kind not in "iufc":
        raise ValueError("variable dtype must be numeric")
    if len(shape) != len(variable.axes):
        raise ValueError("variable shape must match axes")
    names: set[str] = set()
    for size, axis in zip(shape, variable.axes, strict=True):
        name_segment(axis.name)
        if axis.name in names:
            raise ValueError(f"duplicate axis name: {axis.name}")
        names.add(axis.name)
        if axis.values.ndim != 1 or axis.values.shape[0] != size:
            raise ValueError(f"{axis.name}: axis shape does not match variable shape")
        if axis.values.dtype.kind not in "iufc":
            raise ValueError(f"{axis.name}: axis dtype must be numeric")


def write_variable(group: h5.Group, variable: Variable) -> None:
    """Write a validated variable and attach one scale per NumPy dimension."""
    values = group.create_dataset("values", data=variable.values)
    values.attrs["unit"] = variable.unit
    axes = group.create_group("axes")
    for index, axis in enumerate(variable.axes):
        dataset = axes.create_dataset(axis.name, data=axis.values)
        dataset.attrs["unit"] = axis.unit
        dataset.make_scale(axis.name)
        values.dims[index].attach_scale(dataset)


def variable_parts(
    group: h5.Group, *, arrays: bool
) -> tuple[h5.Dataset, tuple[Axis, ...]]:
    """Read local variable schema, optionally loading coordinates, never values."""
    values = known_node(group, "values")
    if not isinstance(values, h5.Dataset) or values.dtype.kind not in "iufc":
        raise FormatError(f"{group.name}: missing numeric values dataset")
    axes: list[Axis] = []
    for index, size in enumerate(values.shape):
        scales = list(values.dims[index].values())
        if len(scales) != 1:
            raise FormatError(f"{values.name}: expected one scale per dimension")
        scale = scales[0]
        if scale.name is None:
            raise FormatError(f"{values.name}: unnamed axis")
        name = scale.name.rsplit("/", 1)[-1]
        name_segment(name)
        if (
            scale.file.id != group.file.id
            or scale.shape != (size,)
            or scale.dtype.kind not in "iufc"
        ):
            raise FormatError(f"{values.name}: invalid axis shape or dtype")
        axes.append(
            Axis(
                name,
                text_attr(scale, "unit"),
                np.asarray(scale[()]) if arrays else np.empty(0, dtype=scale.dtype),
            )
        )
    if len({axis.name for axis in axes}) != len(axes):
        raise FormatError(f"{values.name}: duplicate axis names")
    return values, tuple(axes)


def variable_schema(group: h5.Group) -> VariableSchema:
    """Read a variable declaration without loading values or coordinates."""
    values, axes = variable_parts(group, arrays=False)
    return VariableSchema(values.shape, axes, values.dtype, text_attr(values, "unit"))
