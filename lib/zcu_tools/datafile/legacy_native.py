"""Long-lived native-major-2 reader, projecting representation not run types."""

from __future__ import annotations

import h5py as h5
import numpy as np
from pydantic import TypeAdapter

from .container_models import (
    Axis,
    Container,
    ContainerAttrs,
    ContainerInfo,
    FormatError,
    GroupSource,
    JsonObject,
    Variable,
    VariableSchema,
)
from .container_nodes import (
    json_text,
    read_doc,
    text_attr,
    validate_variable,
    version_minor,
)
from .native_nodes import known_group, known_node

_NAMES = TypeAdapter(list[str])


def legacy_header(file: h5.File) -> ContainerAttrs:
    """Validate legacy major 2 and return the normalized major-3 rewrite header."""
    version_minor(text_attr(file, "format_version"), 2)
    return ContainerAttrs("3.0", "zcu_tools legacy-native", "experiment-run", {})


def legacy_variable(
    group: h5.Group, *, arrays: bool
) -> tuple[h5.Dataset, tuple[Axis, ...]]:
    """Read legacy inner-first axes into NumPy dimension order."""
    signal = known_node(group, text_attr(group, "signal"))
    if not isinstance(signal, h5.Dataset) or signal.dtype.kind not in "iufc":
        raise FormatError(f"{group.name}: expected numeric signal")
    names = _NAMES.validate_json(text_attr(group, "axes"), strict=True)
    axes: list[Axis] = []
    if len(names) != signal.ndim:
        raise FormatError(f"{group.name}: axis count does not match shape")
    for index, name in enumerate(reversed(names)):
        axis = known_node(group, name)
        if (
            not isinstance(axis, h5.Dataset)
            or axis.shape != (signal.shape[index],)
            or axis.dtype.kind not in "iufc"
        ):
            raise FormatError(f"{group.name}/{name}: invalid numeric axis shape")
        scales = list(signal.dims[index].values())
        if len(scales) != 1 or scales[0].id != axis.id:
            raise FormatError(f"{signal.name}: dimension scale does not match axis")
        axes.append(
            Axis(
                name,
                text_attr(axis, "units"),
                np.asarray(axis[()]) if arrays else np.empty(0, dtype=axis.dtype),
            )
        )
    return signal, tuple(axes)


def legacy_schemas(file: h5.File) -> dict[str, VariableSchema]:
    """Read legacy variable schemas without signal or coordinate arrays."""
    result = {}
    data = known_group(file, "data")
    for name in data:
        if not isinstance(name, str):
            raise FormatError("/data: expected text child names")
        if not isinstance(data.get(name, getlink=True), h5.HardLink):
            continue
        group = data[name]
        if not isinstance(group, h5.Group) or "signal" not in group.attrs:
            continue
        signal, axes = legacy_variable(group, arrays=False)
        result[name] = VariableSchema(
            signal.shape, axes, signal.dtype, text_attr(signal, "units")
        )
    return result


def _legacy_provenance(file: h5.File) -> JsonObject:
    provenance: JsonObject = {}
    attributes = known_group(file, "provenance").attrs
    for name in (
        "software_versions",
        "git_dirty",
        "git_commit",
        "qick_version",
        "soc_fingerprint",
        "hostname",
        "simulation_id",
    ):
        value = attributes.get(name, "")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str):
            raise FormatError(f"/provenance/{name}: expected text")
        if name == "software_versions":
            provenance[name] = TypeAdapter(JsonObject).validate_json(value, strict=True)
        elif name == "git_dirty":
            if value not in {"true", "false", "unknown"}:
                raise FormatError("/provenance/git_dirty: invalid boolean")
            provenance[name] = {"true": True, "false": False, "unknown": None}[value]
        else:
            provenance[name] = value or None
    return provenance


def read_legacy_doc(file: h5.File, name: str) -> JsonObject:
    """Project one cfg/metadata doc from a header-validated native 2.x file.

    No variable schemas or arrays are read. Unknown names or missing cfg raise
    KeyError; malformed document nodes raise ValueError/FormatError.
    """
    if name == "cfg":
        cfg = known_node(file, "cfg")
        if cfg is None:
            raise KeyError(name)
        if not isinstance(cfg, h5.Dataset):
            raise FormatError("/cfg: expected dataset")
        return {
            "values": read_doc(file, "cfg"),
            "cfg_type": text_attr(cfg, "cfg_type"),
            "schema_version": text_attr(cfg, "cfg_schema_version"),
        }
    if name != "metadata":
        raise KeyError(name)
    metadata: JsonObject = {}
    for name in (
        "run_id",
        "experiment",
        "started_at",
        "finished_at",
        "completion",
    ):
        value = text_attr(file, name)
        metadata[name] = (
            None if name in {"started_at", "finished_at"} and not value else value
        )
    metadata["snapshot"] = read_doc(file, "context")
    metadata["device_snapshot"] = read_doc(file, "device") if "device" in file else None
    metadata["provenance"] = _legacy_provenance(file)
    data = known_group(file, "data")
    if "metadata" in data.attrs:
        metadata["labber"] = TypeAdapter(JsonObject).validate_json(
            text_attr(data, "metadata"), strict=True
        )
    json_text(metadata)
    return metadata


def load_legacy(file: h5.File) -> Container:
    """Project native 2.x vars and JSON documents; retain the original tree."""
    attrs = legacy_header(file)
    try:
        docs = {name: read_legacy_doc(file, name) for name in ("cfg", "metadata")}
    except KeyError as error:
        raise FormatError("/cfg: missing dataset") from error
    data = known_group(file, "data")
    result: dict[str, Variable] = {}
    for name in data:
        if not isinstance(name, str):
            raise FormatError("/data: expected text child names")
        if not isinstance(data.get(name, getlink=True), h5.HardLink):
            continue
        node = data[name]
        if not isinstance(node, h5.Group) or "signal" not in node.attrs:
            continue
        values, axes = legacy_variable(node, arrays=True)
        variable = Variable(np.asarray(values[()]), axes, text_attr(values, "units"))
        validate_variable(variable)
        result[name] = variable
    for document in docs.values():
        json_text(document)
    return Container(
        result, docs, {"legacy": GroupSource(file.id.get_file_image(), "/")}, attrs
    )


def inspect_legacy(file: h5.File) -> ContainerInfo:
    """Return normalized legacy header and schemas without loading array values."""
    return ContainerInfo(
        legacy_schemas(file), ("cfg", "metadata"), ("legacy",), legacy_header(file)
    )
