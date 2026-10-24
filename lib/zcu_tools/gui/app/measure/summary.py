"""Project analysis values to JSON without invoking result serializers."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from math import isfinite
from typing import Literal

import numpy as np
from pydantic import BaseModel, JsonValue


@dataclass(frozen=True)
class SummaryIssue:
    """A non-finite number replaced by null at a summary field/index path."""

    path: str
    reason: Literal["non_finite"] = "non_finite"


@dataclass(frozen=True)
class AnalysisSummary:
    """JSON projection and ordered locations of numbers replaced by null.

    summary retains all supported fields. invalid contains one issue per
    non-finite real number, including each component of a complex number.
    """

    summary: JsonValue
    invalid: tuple[SummaryIssue, ...]


def summarize_analysis(result: object) -> AnalysisSummary:
    """Project a result into JSON by field name, starting at path ``summary``.

    Dataclasses, pydantic models and string-key mappings become objects.
    Lists, tuples and numpy arrays become lists. Numpy scalars become JSON
    scalars, and complex numbers become real/imag objects. Non-finite numbers
    become null with a SummaryIssue. Custom serializers are never called.
    Unsupported values or non-string keys raise TypeError naming their path.
    The input is not modified.
    """
    invalid: list[SummaryIssue] = []
    projected = _project(result, "summary", invalid)
    return AnalysisSummary(projected, tuple(invalid))


def _numpy_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    # Extended-precision numpy scalars keep their numpy type after item().
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.complexfloating):
        return complex(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _field_values(value: object, path: str) -> Mapping[str, object]:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, BaseModel):
        return {name: getattr(value, name) for name in type(value).model_fields}
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: mapping key {key!r} is not a string")
            result[key] = item
        return result
    raise TypeError(f"{path}: unsupported value type {type(value).__name__}")


def _project(value: object, path: str, invalid: list[SummaryIssue]) -> JsonValue:
    value = _numpy_value(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if isfinite(value):
            return value
        invalid.append(SummaryIssue(path))
        return None
    if isinstance(value, complex):
        return {
            "real": _project(value.real, f"{path}.real", invalid),
            "imag": _project(value.imag, f"{path}.imag", invalid),
        }
    if isinstance(value, (list, tuple)):
        return [
            _project(item, f"{path}[{index}]", invalid)
            for index, item in enumerate(value)
        ]
    return {
        name: _project(item, f"{path}.{name}", invalid)
        for name, item in _field_values(value, path).items()
    }
