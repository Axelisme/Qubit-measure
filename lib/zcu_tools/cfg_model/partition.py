"""Pure separation of unknown fields from otherwise strict persistent Specs."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from pydantic import BaseModel, JsonValue, TypeAdapter

from .materialization import module_spec_from_raw, waveform_spec_from_raw
from .module import ModuleSpec
from .waveform import WaveformSpec


@dataclass(frozen=True)
class SpecPartition[SpecT]:
    """A strict persistent Spec and detached unknown fields from the input.

    spec contains only supported fields and passes the ordinary strict reader.
    unrecognized retains original names and JSON values, including nested unknown
    fields, without copying supported leaves. List branches retain indices with
    empty objects for elements with no unknown fields. Neither value aliases input.
    """

    spec: SpecT
    unrecognized: dict[str, JsonValue]


def _split_fields(
    raw: dict[str, JsonValue], model: BaseModel
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    known: dict[str, JsonValue] = {}
    unknown: dict[str, JsonValue] = {}
    fields: dict[str, object] = dict(model)
    for key, value in raw.items():
        if key in (model.model_extra or {}):
            unknown[key] = deepcopy(value)
            continue
        parsed = fields[key]
        if isinstance(value, dict) and isinstance(parsed, BaseModel):
            known[key], nested = _split_fields(value, parsed)
            if nested:
                unknown[key] = nested
        elif isinstance(value, list) and isinstance(parsed, list):
            clean, remainder = _split_list(value, parsed)
            known[key] = clean
            if any(remainder):
                unknown[key] = remainder
        else:
            known[key] = deepcopy(value)
    return known, unknown


def _split_list(
    raw: list[JsonValue], models: list[object]
) -> tuple[list[JsonValue], list[JsonValue]]:
    known: list[JsonValue] = []
    unknown: list[JsonValue] = []
    for value, model in zip(raw, models, strict=True):
        if isinstance(value, dict) and isinstance(model, BaseModel):
            clean, remainder = _split_fields(value, model)
            known.append(clean)
            unknown.append(remainder)
        else:
            known.append(deepcopy(value))
            unknown.append({})
    return known, unknown


def partition_module_spec(raw: Mapping[str, JsonValue]) -> SpecPartition[ModuleSpec]:
    """Separate unknown JSON fields and validate a persistent module, without I/O.

    raw must use current discriminators and field names. Only unknown fields are
    separated; missing/invalid supported fields, unknown families and non-JSON
    values raise ValueError. Nested module/waveform fields follow their own schemas.
    No legacy conversion, default inference, reference resolution or evaluation
    occurs. Ordinary runtime readers remain strict about every extra field.
    """
    values = TypeAdapter(dict[str, JsonValue]).validate_python(raw, strict=True)
    parsed = TypeAdapter(ModuleSpec).validate_python(values, extra="allow")
    known, unknown = _split_fields(values, parsed)
    return SpecPartition(module_spec_from_raw(known), unknown)


def partition_waveform_spec(
    raw: Mapping[str, JsonValue],
) -> SpecPartition[WaveformSpec]:
    """Separate unknown JSON fields and validate a persistent waveform, without I/O.

    raw uses current waveform discriminators. Invalid/missing supported fields,
    unknown styles and non-JSON values raise ValueError. Inline ramp fields follow
    their schema; concrete ramp keys remain unresolved. The returned Spec also
    passes waveform_spec_from_raw, whose strict extra-field policy is unchanged.
    """
    values = TypeAdapter(dict[str, JsonValue]).validate_python(raw, strict=True)
    parsed = TypeAdapter(WaveformSpec).validate_python(values, extra="allow")
    known, unknown = _split_fields(values, parsed)
    return SpecPartition(waveform_spec_from_raw(known), unknown)
