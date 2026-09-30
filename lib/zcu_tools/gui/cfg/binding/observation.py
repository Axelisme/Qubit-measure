from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from ..model import CfgNodeSpec, CfgNodeValue
from .fields import (
    CenteredSweepField,
    CfgField,
    LiteralField,
    ScalarField,
    SectionField,
    SweepField,
)
from .reference import ReferenceField


@dataclass(frozen=True)
class CfgNodeObservation:
    """Detached model data and cached presentation state, including locked fields.

    Named children follow the cfg tree, not the settable-target subset. Reference
    children describe the active cached shape; a disabled optional ref has none.
    Reading this tree never resolves sources. It is not a persistence encoding.
    """

    spec: CfgNodeSpec
    value: CfgNodeValue | None
    valid: bool
    options: tuple[object, ...] | None = None
    children: dict[str, CfgNodeObservation] = field(default_factory=dict)


def observe_cfg(root: SectionField) -> CfgNodeObservation:
    """Capture cached field state without exposing aliases into the live model."""
    return deepcopy(_observe_field(root))


def _observe_field(node: CfgField) -> CfgNodeObservation:
    if isinstance(node, SectionField):
        return CfgNodeObservation(
            node.spec,
            node.get_value(),
            node.is_valid(),
            children={key: _observe_field(child) for key, child in node.fields.items()},
        )
    if isinstance(node, ReferenceField):
        value = node.get_value()
        children = (
            {key: _observe_field(child) for key, child in node.sub_field.fields.items()}
            if value is not None and node.sub_field is not None
            else {}
        )
        return CfgNodeObservation(
            node.spec,
            value,
            node.is_valid(),
            options=node.available_options(),
            children=children,
        )
    if isinstance(node, ScalarField):
        return CfgNodeObservation(
            node.spec,
            node.get_value(),
            node.is_valid(),
            options=node.available_options(),
        )
    if isinstance(node, (LiteralField, SweepField, CenteredSweepField)):
        return CfgNodeObservation(node.spec, node.get_value(), node.is_valid())
    raise TypeError(f"Unsupported cfg field: {type(node).__name__}")
