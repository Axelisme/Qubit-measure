"""Structural completeness for a new input tree; parsing stays with node writers."""

from collections.abc import Mapping

from .edit_codec import decode_input, encode_input
from .model import (
    CenteredSweepSpec,
    CfgNodeSpec,
    CfgSectionSpec,
    LiteralSpec,
    ReferenceSpec,
    SweepSpec,
)
from .resource import CfgInput, CfgInputError, CfgInputReason


def select_input_shape(
    spec: ReferenceSpec, value: Mapping[str, CfgInput]
) -> CfgSectionSpec:
    key = spec.discriminator
    if key is not None and key in value:
        discriminator = decode_input(encode_input(value[key]))
        for shape in spec.allowed:
            literal = shape.fields[key]
            if isinstance(literal, LiteralSpec) and literal.value == discriminator:
                return shape
        raise CfgInputError(
            CfgInputReason.INVALID_VALUE, "unsupported reference discriminator"
        )
    if len(spec.allowed) == 1:
        return spec.allowed[0]
    raise CfgInputError(
        CfgInputReason.INVALID_VALUE,
        "complete reference input requires a discriminator",
    )


def require_complete_input(spec: CfgNodeSpec, value: CfgInput) -> None:
    if isinstance(spec, ReferenceSpec):
        payload = _mapping(value)
        if "__ref" not in payload:
            require_complete_input(select_input_shape(spec, payload), payload)
    elif isinstance(spec, CfgSectionSpec):
        payload = _mapping(value)
        required = {
            key
            for key, child in spec.fields.items()
            if not isinstance(child, LiteralSpec)
        }
        _require_keys(payload, required)
        for key in required:
            require_complete_input(spec.fields[key], payload[key])
    elif isinstance(spec, (SweepSpec, CenteredSweepSpec)):
        payload = _mapping(value)
        if isinstance(spec, SweepSpec):
            required = {"start", "stop"}
        else:
            required = {"span"}
            if spec.locked_center is None:
                required.add("center")
        _require_keys(payload, required)
        if ("expts" in payload) == ("step" in payload):
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE,
                "complete range input requires exactly one of expts and step",
            )


def _mapping(value: CfgInput) -> Mapping[str, CfgInput]:
    if not isinstance(value, Mapping):
        raise CfgInputError(
            CfgInputReason.INVALID_VALUE, "complete aggregate input must be a mapping"
        )
    return value


def _require_keys(value: Mapping[str, CfgInput], required: set[str]) -> None:
    missing = required - value.keys()
    if missing:
        raise CfgInputError(
            CfgInputReason.INVALID_VALUE,
            f"complete input is missing fields: {', '.join(sorted(missing))}",
        )
