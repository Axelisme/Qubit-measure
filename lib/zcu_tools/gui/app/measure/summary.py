"""Project analysis result objects to JSON without invoking result serializers."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from math import isfinite
from typing import Literal

import numpy as np
from pydantic import BaseModel, JsonValue

from zcu_tools.datafile import JsonObject


@dataclass(frozen=True)
class SummaryIssue:
    """A non-finite number replaced by null at a summary field/index path."""

    path: str
    reason: Literal["non_finite"] = "non_finite"


@dataclass(frozen=True)
class AnalysisSummary:
    """JSON object projection and ordered locations of numbers replaced by null.

    summary retains all fields from a dataclass instance, pydantic model or
    string-key Mapping root. summarize_analysis raises TypeError for other roots.
    invalid contains one issue per non-finite real number, including each
    component of a complex number.
    """

    summary: JsonObject
    invalid: tuple[SummaryIssue, ...]


def summarize_analysis(result: object) -> AnalysisSummary:
    """Project a result object by field name, starting at path ``summary``.

    The root must be a dataclass instance, pydantic model or string-key Mapping.
    Other roots raise TypeError naming the root type and path ``summary``.
    Nested dataclasses, pydantic models and mappings become objects. Nested
    lists, tuples and numpy arrays become lists. Numpy scalars become JSON
    scalars, and complex numbers become real/imag objects. Non-finite numbers
    become null with a SummaryIssue. Custom serializers and to_summary_dict
    are never called. Unsupported nested values or non-string keys raise
    TypeError naming their path. The input is not modified.
    """
    invalid: list[SummaryIssue] = []
    projected: JsonObject = {
        name: _project(item, f"summary.{name}", invalid)
        for name, item in _field_values(result, "summary").items()
    }
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
