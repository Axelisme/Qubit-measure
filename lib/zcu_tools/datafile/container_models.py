"""Value types for the representation-only container API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from pydantic import JsonValue

JsonScalar = str | int | float | bool | None
JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class ContainerAttrs:
    """Root attributes for detached containers and lossless save.

    format_version is a major.minor string in major 3. producer identifies the
    writer. kind is an optional opaque producer-defined label. extra preserves
    unknown finite JSON scalar attributes; reserved header keys are forbidden.
    Legacy readers normalize the version to 3.0 before returning these attrs.
    """

    format_version: str
    producer: str
    kind: str | None
    extra: Mapping[str, JsonScalar]


class FormatError(ValueError):
    """An HDF5 container has an unsupported header or malformed known content."""


class Axis(NamedTuple):
    """One dimension's name, unit (empty for dimensionless), and 1D numeric values.

    Axes follow NumPy dimension order in Variable and VariableSchema.
    Names are unique, nonempty path segments other than '.' and '..'.
    """

    name: str
    unit: str
    values: np.ndarray


@dataclass(frozen=True)
class Variable:
    """A numeric array, its dimension-ordered axes, and unit (empty if unitless).

    Each axis length must equal the corresponding array dimension. Scalars have
    no axes. Integer, floating and complex arrays are supported; bool is not.
    """

    values: np.ndarray
    axes: tuple[Axis, ...]
    unit: str


@dataclass(frozen=True)
class VariableSchema:
    """The variable schema returned by inspect.

    shape lists nonnegative NumPy dimension sizes. axes describes each dimension
    in that order. dtype must be integer, floating or complex. unit is the value
    unit, or empty for unitless values.
    Inspect returns axes with empty values: it does not read coordinate arrays.
    """

    shape: tuple[int, ...]
    axes: tuple[Axis, ...]
    dtype: np.dtype[np.generic]
    unit: str


@dataclass(frozen=True)
class GroupSource:
    """An opaque HDF5 subtree to copy on save, without interpreting its content.

    source is an existing file Path, or detached file-image bytes returned by
    load. node is an absolute HDF5 group path within that source. Loaded sources
    remain usable after the original file is closed, replaced or deleted.
    """

    source: Path | bytes
    node: str


@dataclass(frozen=True)
class Container:
    """Detached contents of a container, with no experiment-specific semantics.

    vars maps names to numeric variables; docs maps names to finite JSON objects;
    groups maps names to opaque subtrees for rewriting. attrs holds all root
    HDF5 attributes, including format, format_version, producer and optional kind.
    """

    vars: Mapping[str, Variable]
    docs: Mapping[str, JsonObject]
    groups: Mapping[str, GroupSource]
    attrs: ContainerAttrs


@dataclass(frozen=True)
class ContainerInfo:
    """Header and schemas without loading variable or coordinate array values.

    vars maps names to declarations (axes have empty coordinate arrays). docs
    lists document names in file order. groups lists opaque subtree names.
    attrs holds all root attributes, as in Container.
    """

    vars: Mapping[str, VariableSchema]
    docs: tuple[str, ...]
    groups: tuple[str, ...]
    attrs: ContainerAttrs
