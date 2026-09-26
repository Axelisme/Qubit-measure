"""Lower already-presented cfg values without expression or catalog access."""

from __future__ import annotations

from copy import deepcopy

from .lowering import RangeFactory, lower_finished_cfg
from .model import (
    CfgNodeValue,
    CfgSchema,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceValue,
    ScalarValue,
    SweepValue,
)
from .reference_key import make_custom_reference_key


def lower_resolved_cfg(
    schema: CfgSchema, *, make_range: RangeFactory
) -> dict[str, object]:
    """Lower an isolated copy using only cached resolution and validation state.

    Every expression and enabled reference must already be resolved. The local
    copy replaces expressions with direct values and library identities with
    cached shape labels, so the existing static validator and lowering algorithm
    need no source ports. Original sources, raw text, and metadata remain intact.
    This does not change the live-context contract of ``lower_finished_cfg``.
    """
    frozen = deepcopy(schema)
    _freeze_value(frozen.value, "")
    return lower_finished_cfg(
        frozen,
        resolve_expression=None,
        resolve_reference=None,
        make_range=make_range,
    )


def _freeze_value(value: CfgNodeValue | None, path: str) -> CfgNodeValue | None:
    if value is None:
        return None
    if isinstance(value, (DirectValue, EvalValue)):
        return _resolved_scalar(value, path)
    if isinstance(value, CfgSectionValue):
        for key, child in value.fields.items():
            child_path = f"{path}.{key}" if path else key
            value.fields[key] = _freeze_value(child, child_path)
        return value
    if isinstance(value, ReferenceValue):
        if value.error is not None:
            raise RuntimeError(f"Config field '{path}': {value.error}")
        if value.resolved_label is None:
            raise RuntimeError(f"Config field '{path}' reference is unresolved")
        # Shape selection is per node, not a cache keyed by library identity.
        value.chosen_key = make_custom_reference_key(value.resolved_label)
        _freeze_value(value.value, path)
        return value
    if isinstance(value, SweepValue):
        value.start = _resolved_edge(value.start, f"{path}.start")
        value.stop = _resolved_edge(value.stop, f"{path}.stop")
    else:
        value.center = _resolved_edge(value.center, f"{path}.center")
        value.span = _resolved_control(value.span, f"{path}.span")
    # Mutate only the detached copy; do not reconstruct range carriers, whose
    # constructors normalize input and may derive a different canonical step.
    value.expts = _resolved_control(value.expts, f"{path}.expts")
    value.step = _resolved_control(value.step, f"{path}.step")
    return value


def _resolved_scalar(value: ScalarValue, path: str) -> DirectValue:
    error = value.error if value.error is not None else value.validation_error
    if error is not None:
        raise RuntimeError(f"Config field '{path}': {error}")
    if isinstance(value, DirectValue):
        return value
    if value.resolved is None:
        raise RuntimeError(
            f"Config field '{path}' expression {value.expr!r} is unresolved"
        )
    return DirectValue(value.resolved)


def _resolved_edge(value: float | ScalarValue, path: str) -> float | DirectValue:
    return (
        _resolved_scalar(value, path)
        if isinstance(value, (DirectValue, EvalValue))
        else value
    )


def _resolved_control[T: (int, float)](
    value: T | DirectValue, path: str
) -> T | DirectValue:
    return _resolved_scalar(value, path) if isinstance(value, DirectValue) else value
