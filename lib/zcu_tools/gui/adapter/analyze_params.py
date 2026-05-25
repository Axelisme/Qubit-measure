from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import typing_extensions
from typing_extensions import Annotated, Literal

_LITERAL_ORIGINS = {typing.Literal, typing_extensions.Literal}


@dataclass(frozen=True)
class ParamMeta:
    label: str = ""
    decimals: Optional[int] = None


def _resolve_field_info(
    field: dataclasses.Field,
    hints: dict[str, Any],
) -> tuple[type, Optional[list[Any]], str, Optional[int]]:
    """Return (bare_type, choices, label, decimals)."""
    hint = hints[field.name]
    origin = get_origin(hint)
    if origin is Annotated:
        args = get_args(hint)
        bare = args[0]
        meta = next((arg for arg in args[1:] if isinstance(arg, ParamMeta)), None)
    else:
        bare = hint
        meta = None

    bare_origin = get_origin(bare)
    if bare_origin in _LITERAL_ORIGINS:
        literal_args = get_args(bare)
        if not literal_args:
            raise TypeError(
                f"Analyze parameter Literal must not be empty: {field.name}"
            )
        choices = list(literal_args)
        bare_type = type(literal_args[0])
        if any(type(choice) is not bare_type for choice in literal_args):
            raise TypeError(
                f"Analyze parameter Literal choices must share one type: {field.name}"
            )
    elif bare in {bool, int, float, str}:
        choices = None
        bare_type = bare
    else:
        raise TypeError(f"Unsupported analyze parameter annotation: {bare!r}")

    label = meta.label if meta is not None and meta.label else field.name
    decimals = meta.decimals if meta is not None else None
    return bare_type, choices, label, decimals


T = TypeVar("T")


def reconstruct_params(params_cls: type[T], form_values: dict[str, Any]) -> T:
    if not dataclasses.is_dataclass(params_cls):
        raise TypeError(f"Analyze params must be a dataclass type: {params_cls!r}")

    fields = dataclasses.fields(params_cls)
    hints = get_type_hints(params_cls, include_extras=True)
    field_names = {field.name for field in fields}

    extra = set(form_values) - field_names
    if extra:
        names = ", ".join(sorted(extra))
        raise RuntimeError(f"Unknown analyze params: {names}")

    missing = field_names - set(form_values)
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"Missing analyze params: {names}")

    kwargs: dict[str, Any] = {}
    for field in fields:
        bare_type, choices, _, _ = _resolve_field_info(field, hints)
        raw = form_values[field.name]
        value = _coerce_analyze_value(field.name, bare_type, raw)
        if choices is not None and value not in choices:
            raise RuntimeError(
                f"Analyze param {field.name!r} must be one of {choices}, got {value!r}"
            )
        kwargs[field.name] = value

    return params_cls(**kwargs)


def _coerce_analyze_value(name: str, bare_type: type, raw: Any) -> Any:
    if bare_type is bool:
        if not isinstance(raw, bool):
            raise RuntimeError(
                f"Analyze param {name!r} expects bool, got {type(raw).__name__}"
            )
        return raw
    if bare_type is int:
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise RuntimeError(
                f"Analyze param {name!r} expects int, got {type(raw).__name__}"
            )
        return raw
    if bare_type is float:
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise RuntimeError(
                f"Analyze param {name!r} expects float, got {type(raw).__name__}"
            )
        return float(raw)
    if bare_type is str:
        if not isinstance(raw, str):
            raise RuntimeError(
                f"Analyze param {name!r} expects str, got {type(raw).__name__}"
            )
        return raw
    raise TypeError(f"Unsupported analyze parameter annotation: {bare_type!r}")
