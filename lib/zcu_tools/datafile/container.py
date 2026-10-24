"""Atomic one-shot I/O for representation-only HDF5 containers."""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import h5py as h5
import numpy as np

from .container_models import (
    Container,
    ContainerInfo,
    FormatError,
    GroupSource,
    JsonObject,
    Variable,
)
from .container_nodes import (
    json_text,
    name_segment,
    root_attrs,
    text_attr,
    validate_variable,
    variable_parts,
    variable_schema,
    write_attrs,
    write_doc,
    write_variable,
)
from .container_nodes import (
    read_doc as read_document,
)
from .native_nodes import known_group, known_node


@dataclass(frozen=True)
class _DetachedContainer(Container):
    """Retain the object graph across edits, including dataclasses.replace."""

    image: bytes = field(repr=False, compare=False)


def save(path: Path, container: Container, *, replace: bool = False) -> None:
    """Write a container to exactly path through a same-directory temporary file.

    container.vars use NumPy axis order; docs contain strict finite JSON objects;
    groups copy opaque HDF5 subtrees. Names are nonempty path segments excluding
    '.', '..', slash and NUL. container.attrs preserves major-3 forward minors,
    producer/kind and finite scalar extensions. New callers specify version 3.0;
    load returns the same value type, ready for a lossless rewrite.
    Existing paths raise FileExistsError unless replace=True. Publication is
    atomic, not a cross-file transaction or power-loss guarantee. Failed writes
    preserve the destination and remove temporary files. Invalid input raises
    ValueError (FormatError for malformed headers); I/O errors propagate.
    References in groups to reshaped variables are not guaranteed to remain valid.
    """
    path = Path(path)
    if not replace and path.exists():
        raise FileExistsError(path)
    validate_contents(container.vars, container.docs, container.groups)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temp = Path(temporary)
    try:
        image = container.image if isinstance(container, _DetachedContainer) else b""
        if image:
            temp.write_bytes(image)
        with h5.File(temp, "r+" if image else "w") as file:
            _write_contents(file, container, image)
        if replace:
            os.replace(temp, path)
        else:
            os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_contents(file: h5.File, container: Container, image: bytes) -> None:
    # Editing the detached graph keeps references attached to canonical objects.
    for key in tuple(file.attrs):
        del file.attrs[key]
    write_attrs(file, container.attrs)
    variables = file.require_group("vars")
    documents = file.require_group("docs")
    opaque = file.require_group("groups")
    for namespace, contents in (
        (variables, container.vars),
        (documents, container.docs),
        (opaque, container.groups),
    ):
        for name in tuple(namespace):
            if name not in contents:
                del namespace[name]
    for name, variable in container.vars.items():
        write_variable(variables.require_group(name), variable)
    for name, doc in container.docs.items():
        write_doc(documents, name, doc)
    for name, source in container.groups.items():
        if image and source == GroupSource(image, f"/groups/{name}"):
            continue
        if name in opaque:
            del opaque[name]
        copy_group(opaque, name, source)


def validate_contents(
    variables: Mapping[str, Variable],
    docs: Mapping[str, JsonObject],
    groups: Mapping[str, GroupSource],
) -> None:
    """Validate all names and in-memory content before creating the output file."""
    for namespace in (variables, docs, groups):
        for name in namespace:
            name_segment(name)
    for variable in variables.values():
        validate_variable(variable)
    for value in docs.values():
        json_text(value)


def node_names(group: h5.Group) -> tuple[str, ...]:
    """Return validated immediate child names, refusing unnamed HDF5 objects."""
    names: list[str] = []
    for name in group:
        if not isinstance(name, str):
            raise FormatError(f"{group.name}: child name must be text")
        name_segment(name)
        names.append(name)
    return tuple(names)


def copy_group(destination: h5.Group, name: str, source: GroupSource) -> None:
    """Copy a caller-provided or detached HDF5 group, preserving its subtree."""
    stream = (
        io.BytesIO(source.source) if isinstance(source.source, bytes) else source.source
    )
    with h5.File(stream, "r") as file:
        node = file[source.node]
        if not isinstance(node, h5.Group):
            raise ValueError(f"{source.node}: group source must be an HDF5 group")
        file.copy(node, destination, name=name, expand_refs=True)


def load(path: Path) -> Container:
    """Read and close a major-3 container.

    All arrays/docs are detached, and opaque groups retain a detached file image.
    Unknown named content is retained. Known nodes must be local, never external
    links. Same-major future minors are accepted.
    Malformed content raises path-qualified FormatError; I/O errors propagate.
    """
    with h5.File(path, "r") as file:
        try:
            attrs = root_attrs(file)
            variables = known_group(file, "vars")
            documents = known_group(file, "docs")
            opaque = known_group(file, "groups")
            result: dict[str, Variable] = {}
            for name in node_names(variables):
                name_segment(name)
                values, axes = variable_parts(known_group(variables, name), arrays=True)
                result[name] = Variable(
                    np.asarray(values[()]), axes, text_attr(values, "unit")
                )
            docs = {}
            for name in node_names(documents):
                name_segment(name)
                docs[name] = read_document(documents, name)
            image = file.id.get_file_image()
            groups = {}
            for name in node_names(opaque):
                name_segment(name)
                groups[name] = GroupSource(image, f"/groups/{name}")
            return _DetachedContainer(result, docs, groups, attrs, image)
        except ValueError as error:
            raise FormatError(f"{path}: {error}") from error


def read_doc(path: Path, name: str) -> JsonObject:
    """Read one detached finite JSON object without loading vars or other docs.

    path is an exact major-3 container. name is a nonempty path segment
    excluding '.', '..', slash and NUL.
    Only the header and requested document's nodes are validated. Invalid vars
    do not prevent document reads, and success does not promise load will work.
    Missing documents raise KeyError naming path and name. Invalid names raise
    ValueError; malformed headers/documents or external links raise path-qualified
    FormatError. File I/O errors propagate. The file is closed before returning.
    """
    name_segment(name)
    with h5.File(path, "r") as file:
        try:
            root_attrs(file)
            documents = known_group(file, "docs")
            if known_node(documents, name) is None:
                raise KeyError(name)
            return read_document(documents, name)
        except KeyError as error:
            raise KeyError(f"{path}: document {name!r} not found") from error
        except ValueError as error:
            raise FormatError(f"{path}: {error}") from error


def inspect(path: Path) -> ContainerInfo:
    """Read header and var/doc names and schemas without loading any arrays.

    Coordinate arrays are empty in returned axes. Documents are not decoded, so
    inspect is not a promise that load will succeed. Legacy headers and variable
    schemas are supported. Malformed schema raises FormatError; I/O propagates.
    """
    with h5.File(path, "r") as file:
        try:
            attrs = root_attrs(file)
            variables = known_group(file, "vars")
            documents = known_group(file, "docs")
            groups = known_group(file, "groups")
            return ContainerInfo(
                {
                    name: variable_schema(known_group(variables, name))
                    for name in node_names(variables)
                },
                node_names(documents),
                node_names(groups),
                attrs,
            )
        except ValueError as error:
            raise FormatError(f"{path}: {error}") from error
