"""Input-only projection when no current resolved result can be trusted."""

from copy import deepcopy
from dataclasses import replace

from .binding.observation import CfgNodeObservation
from .model import (
    CenteredSweepValue,
    CfgNodeValue,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceValue,
    ScalarValue,
    SweepValue,
)
from .reference_key import parse_custom_reference_key


def unavailable_tree(tree: CfgNodeObservation) -> CfgNodeObservation:
    detached = deepcopy(tree)
    children = detached.children
    if isinstance(detached.value, ReferenceValue) and _is_linked(detached.value):
        children = {}
    return replace(
        detached,
        value=_input_value(detached.value),
        valid=False,
        options=None,
        children={key: unavailable_tree(child) for key, child in children.items()},
    )


def _is_linked(value: ReferenceValue) -> bool:
    return (
        not value.is_overridden and parse_custom_reference_key(value.chosen_key) is None
    )


def _scalar_input(value: float | ScalarValue) -> float | ScalarValue:
    if isinstance(value, EvalValue):
        return replace(
            value,
            resolved=None,
            error="Resolution is unavailable",
            validation_error=None,
        )
    return value


def _section_input(value: CfgSectionValue) -> CfgSectionValue:
    return CfgSectionValue(
        {key: _input_value(child) for key, child in value.fields.items()}
    )


def _input_value(value: CfgNodeValue | None) -> CfgNodeValue | None:
    if isinstance(value, EvalValue):
        return replace(
            value,
            resolved=None,
            error="Resolution is unavailable",
            validation_error=None,
        )
    if isinstance(value, CfgSectionValue):
        return _section_input(value)
    if isinstance(value, ReferenceValue):
        return replace(
            value,
            resolved_label=None,
            error="Resolution is unavailable",
            value=CfgSectionValue()
            if _is_linked(value)
            else _section_input(value.value),
        )
    if isinstance(value, SweepValue):
        return replace(
            value,
            start=_scalar_input(value.start),
            stop=_scalar_input(value.stop),
            step=value.step
            if isinstance(value.step, DirectValue)
            else DirectValue(None),
            auto_norm=False,
        )
    if isinstance(value, CenteredSweepValue):
        return replace(
            value,
            center=_scalar_input(value.center),
            step=value.step
            if isinstance(value.step, DirectValue)
            else DirectValue(None),
            auto_norm=False,
        )
    return value
