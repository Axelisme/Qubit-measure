"""Experiment-owned declarations and Native codec for fixed run values (ADR-0063)."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from zcu_tools.datafile import DataVariable, JsonObject, Variable

_DOCUMENT = "/docs/run_values"
_VERSION = 1
_FloatT = TypeVar("_FloatT", bound=np.floating)
type _ResultValue = float | None | NDArray[np.floating]


@dataclass(frozen=True, kw_only=True)
class RunValueAxis:
    """Reference the exact disk axis label of a required Native variable.

    variable is the stable DataVariable identity; label is the stored axis
    name. An invalid identity or empty label raises ValueError. The enclosing
    persistence spec validates that the axis exists and is unique.
    """

    variable: DataVariable
    label: str

    def __post_init__(self) -> None:
        DataVariable(self.variable)
        if not self.label:
            raise ValueError("run value axis requires a nonempty label")


@dataclass(frozen=True, kw_only=True)
class RunValueSpec:
    """Declare a fixed scalar or aligned vector saved in Native, not Labber.

    field_name selects a Result dataclass field; wire_name is its stable
    lowercase snake_case key. unit is the exact disk unit (empty if unitless).
    disk = memory * scale; scale must be finite and nonzero. dtype is the
    reconstructed memory type: float for scalars or the NumPy floating type
    of a vector. axis selects the vector coordinates, or None for a scalar.
    allow_unknown maps scalar None or vector NaN to JSON null and permits a
    missing entry on load. Invalid declarations raise ValueError.
    """

    field_name: str
    wire_name: str
    unit: str
    scale: float = 1.0
    dtype: type = float
    allow_unknown: bool = False
    axis: RunValueAxis | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", self.wire_name) is None:
            raise ValueError(f"invalid run value wire_name {self.wire_name!r}")
        if not math.isfinite(self.scale) or self.scale == 0.0:
            raise ValueError(f"{self.wire_name}: scale must be finite and nonzero")
        if self.axis is None and self.dtype is not float:
            raise ValueError(f"{self.wire_name}: scalar dtype must be float")
        if self.axis is not None and not issubclass(self.dtype, np.floating):
            raise ValueError(
                f"{self.wire_name}: vector dtype must be a NumPy floating type"
            )


@dataclass(frozen=True)
class LoadedRunValues:
    """Validated fixed values in Result memory units, keyed by Result field name.

    Constructed by the Native reader for grouped Result builders. Use the typed
    accessors; undeclared field names and mismatched declarations raise ValueError.
    """

    _fields: Mapping[str, tuple[RunValueSpec, _ResultValue]] = field(
        default_factory=dict
    )

    def scalar(self, name: str) -> float:
        """Read a required scalar by Result field name; nullable fields are rejected."""
        declaration, value = self._field(name)
        if declaration.allow_unknown or not isinstance(value, float):
            raise ValueError(f"run value {name!r} is not a required scalar")
        return value

    def nullable_scalar(self, name: str) -> float | None:
        """Read a nullable scalar by Result field name; required fields are rejected."""
        declaration, value = self._field(name)
        if not declaration.allow_unknown or isinstance(value, np.ndarray):
            raise ValueError(f"run value {name!r} is not a nullable scalar")
        return value

    def vector(self, name: str, *, dtype: type[_FloatT]) -> NDArray[_FloatT]:
        """Read an aligned vector by Result field name and exact declared dtype."""
        declaration, value = self._field(name)
        if declaration.dtype is not dtype or not isinstance(value, np.ndarray):
            raise ValueError(f"run value {name!r} is not a vector with dtype {dtype}")
        return np.asarray(value, dtype=dtype)

    def result_fields(self) -> dict[str, _ResultValue]:
        """Return a fresh mapping of fixed values for a single Result constructor."""
        return {name: value for name, (_, value) in self._fields.items()}

    def _field(self, name: str) -> tuple[RunValueSpec, _ResultValue]:
        try:
            return self._fields[name]
        except KeyError:
            raise ValueError(f"undeclared run value Result field {name!r}") from None


def validate_declarations(
    declarations: tuple[RunValueSpec, ...],
    version: int,
    result_fields: set[str],
    array_fields: set[str],
    required_axes: Mapping[DataVariable, tuple[str, ...]],
) -> None:
    """Reject unsupported versions, duplicate keys and invalid Result/axis mappings.

    result_fields lists dataclass fields; array_fields lists those already
    mapped to vars. required_axes lists disk labels by required variable.
    Errors are ValueError at spec construction.
    """
    if type(version) is not int or version != _VERSION:
        raise ValueError(f"run_values_version must be integer {_VERSION}")
    names = {spec.field_name for spec in declarations}
    if len(names) != len(declarations) or len(
        {s.wire_name for s in declarations}
    ) != len(declarations):
        raise ValueError("run values require unique Result fields and wire names")
    if missing := names - result_fields:
        raise ValueError(
            f"run value Result fields not declared on Result: {sorted(missing)}"
        )
    if overlaps := names & array_fields:
        raise ValueError(
            f"run value Result fields also mapped to vars: {sorted(overlaps)}"
        )
    for spec in declarations:
        if spec.axis is not None:
            axis = spec.axis
            if required_axes.get(axis.variable, ()).count(axis.label) != 1:
                raise ValueError(
                    f"{spec.wire_name}: expected a unique axis {axis.label!r} "
                    f"on required variable {axis.variable!r}"
                )


def encode_run_values(
    variables: Mapping[str, Variable],
    *,
    result: object,
    declarations: tuple[RunValueSpec, ...],
    version: int,
) -> JsonObject:
    """Encode all declared Result fields to the validated Native version.

    variables contains validated disk arrays and coordinates. Invalid values
    raise ValueError with the document field path. No I/O or mutation occurs.
    """
    values: JsonObject = {}
    for spec in declarations:
        path = f"{_DOCUMENT}/values/{spec.wire_name}/value"
        values[spec.wire_name] = {
            "unit": spec.unit,
            "value": _encode_value(
                getattr(result, spec.field_name), spec, variables, path=path
            ),
        }
    return {"version": version, "values": values}


def decode_run_values(
    docs: Mapping[str, JsonObject],
    declarations: tuple[RunValueSpec, ...],
    variables: Mapping[str, Variable],
) -> LoadedRunValues:
    """Restore declared values from Native docs, ignoring undeclared entries.

    variables contains validated disk arrays and coordinates. Missing values
    use the unknown policy; malformed data raises ValueError with the disk path.
    An empty declaration ignores the document.
    """
    if not declarations:
        return LoadedRunValues()
    values = _entries(docs.get("run_values"))
    return LoadedRunValues(
        {s.field_name: (s, _decode_entry(values, s, variables)) for s in declarations}
    )


def _entries(document: JsonObject | None) -> JsonObject:
    if document is None:
        return {}
    version = document.get("version")
    if type(version) is not int or version != _VERSION:
        raise ValueError(
            f"{_DOCUMENT}/version: expected integer {_VERSION}, got {version!r}"
        )
    values = document.get("values")
    if not isinstance(values, dict):
        raise ValueError(f"{_DOCUMENT}/values: expected an object")
    return values


def _decode_entry(
    values: JsonObject, spec: RunValueSpec, variables: Mapping[str, Variable]
) -> _ResultValue:
    path = f"{_DOCUMENT}/values/{spec.wire_name}"
    if spec.wire_name not in values:
        return _missing_value(spec, variables, path=path)
    entry = values[spec.wire_name]
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: expected an object")
    if entry.get("unit") != spec.unit:
        raise ValueError(
            f"{path}/unit: expected {spec.unit!r}, got {entry.get('unit')!r}"
        )
    if "value" not in entry:
        raise ValueError(f"{path}/value: missing value")
    value = entry["value"]
    if spec.axis is None:
        return _scaled_scalar(value, spec, path=f"{path}/value", saving=False)
    length = _axis_length(spec.axis, variables)
    return _decode_vector(value, spec, length=length, path=f"{path}/value")


def _axis_length(axis: RunValueAxis, variables: Mapping[str, Variable]) -> int:
    # The declaration and the Native variable validator guarantee a unique match.
    return next(
        len(coordinate.values)
        for coordinate in variables[axis.variable].axes
        if coordinate.name == axis.label
    )


def _missing_value(
    spec: RunValueSpec, variables: Mapping[str, Variable], *, path: str
) -> _ResultValue:
    if not spec.allow_unknown:
        raise ValueError(
            f"{path}: required run value is missing; use the matching older version "
            "or inspect the original file"
        )
    if spec.axis is None:
        return None
    length = _axis_length(spec.axis, variables)
    return np.full(length, np.nan, dtype=spec.dtype)


def _encode_value(
    value: object, spec: RunValueSpec, variables: Mapping[str, Variable], *, path: str
) -> JsonValue:
    if spec.axis is None:
        return _scaled_scalar(value, spec, path=path, saving=True)
    length = _axis_length(spec.axis, variables)
    if not isinstance(value, np.ndarray) or value.shape != (length,):
        raise ValueError(f"{path}: expected a 1D vector of length {length}")
    if value.dtype != np.dtype(spec.dtype):
        raise ValueError(f"{path}: expected memory dtype {np.dtype(spec.dtype)}")
    encoded: list[JsonValue] = [
        _scaled_scalar(
            None if np.isnan(item) else float(item),
            spec,
            path=f"{path}/{index}",
            saving=True,
        )
        for index, item in enumerate(value)
    ]
    return encoded


def _decode_vector(
    value: JsonValue, spec: RunValueSpec, *, length: int, path: str
) -> NDArray[np.floating]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{path}: expected a 1D array of length {length}")
    numbers = [
        _scaled_scalar(item, spec, path=f"{path}/{index}", saving=False)
        for index, item in enumerate(value)
    ]
    with np.errstate(over="ignore"):
        vector = np.asarray(
            [np.nan if number is None else number for number in numbers],
            dtype=spec.dtype,
        )
    if np.isinf(vector).any():
        raise ValueError(f"{path}: number exceeds finite {np.dtype(spec.dtype)} range")
    return vector


def _scaled_scalar(
    value: object, spec: RunValueSpec, *, path: str, saving: bool
) -> float | None:
    if value is None and spec.allow_unknown:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: expected a finite number")
    try:
        scaled = value * spec.scale if saving else value / spec.scale
    except OverflowError as error:
        raise ValueError(f"{path}: number exceeds finite float range") from error
    if not math.isfinite(scaled):
        raise ValueError(f"{path}: number exceeds finite float range")
    return scaled
