"""Immutable timeline axes and additive one-dimensional components.

This module owns identity alignment and rational scalar/ndarray arithmetic.
Expression normalization and constrained Abs ranges live in _timeline_expression.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ._timeline_expression import TimeExpression

CycleValues = NDArray[np.int64] | NDArray[np.object_]
CycleLimits = tuple[Fraction | None, Fraction | None]
Scalar = int | Fraction


@dataclass(frozen=True)
class CycleProducer:
    """Runtime recipe, independent of proof components and register bindings.

    axes/points describe a known source. reconstruction and selection retain
    integer recipes for branch lowering. fixed_point stores numerator and shift;
    rounding_remainder identifies a quotient's bounded remainder, not a loop.
    """

    axes: tuple[tuple[str, int], ...] = ()
    points: tuple[int, ...] | None = None
    reconstruction: "TimeExpression | None" = None
    selection: "tuple[TimeExpression, tuple[int, ...]] | None" = None
    fixed_point: "tuple[TimeExpression, int] | None" = None
    rounding_remainder: bool = False


@dataclass(frozen=True, eq=False)
class CycleAxis:
    """An axis ID, one-dimensional values or bounds, and its runtime producer.

    Loop names identify correlated domains. Each unknown write/error gets a
    fresh object ID, even with a range hint. None is an unavailable endpoint.
    values is one-dimensional; no Cartesian proof grids are stored.
    """

    name: str
    identity: str | object
    minimum: int | None
    maximum: int | None
    values: CycleValues | None
    producer: CycleProducer


def cycle_axis(
    name: str,
    minimum: int | None,
    maximum: int | None,
    producer: CycleProducer | None = None,
) -> CycleAxis:
    """Capture a one-axis known source, otherwise an independent range axis."""
    producer = CycleProducer() if producer is None else producer
    identity: str | object = object()
    points = None
    if len(producer.axes) == 1 and not producer.rounding_remainder:
        identity, count = producer.axes[0]
        points = np.asarray(
            producer.points if producer.points is not None else range(count),
            dtype=np.int64,
        )
    if points is not None:
        points.flags.writeable = False
    return CycleAxis(name, identity, minimum, maximum, points, producer)


CONST_AXIS = CycleAxis("constant", "const", 1, 1, np.asarray([1]), CycleProducer())
Component = CycleValues | Scalar
Components = tuple[tuple[CycleAxis, Component], ...]


def scale_component(component: Component, coefficient: Scalar) -> Component:
    """Multiply a scalar or one-dimensional component by a rational constant."""
    if isinstance(component, np.ndarray):
        return np.asarray(component, dtype=object) * np.asarray(
            coefficient, dtype=object
        )
    return component * coefficient


def _component_add(left: Component, right: Component) -> Component:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.asarray(left, dtype=object) + np.asarray(right, dtype=object)
    return left + right


def nonzero_component(component: Component) -> bool:
    """Return whether a scalar or known component has a nonzero value."""
    return bool(np.any(np.asarray(component, dtype=object)))


def component_key(axis: CycleAxis, component: Component) -> tuple[object, ...]:
    """Return an immutable structural key using axis identity and actual values."""
    return (
        axis.identity,
        *(component.flat if isinstance(component, np.ndarray) else (component,)),
    )


@dataclass(frozen=True, eq=False, init=False)
class TimeVariable:
    """A sum of axis components, with a separate integer reconstruction recipe.

    components is the authoritative proof mapping, including the singleton
    const axis. Known components are one-dimensional arrays. Range components
    are coefficients of independent unknowns. terms/constant are runtime recipe
    operands only; proofs never reconstruct information from them after capture.
    """

    constant: Scalar
    terms: tuple[tuple[CycleAxis, Scalar], ...]
    components: Components

    def __init__(
        self,
        constant: Scalar = 0,
        terms: tuple[tuple[CycleAxis, Scalar], ...] = (),
        *,
        components: Components | None = None,
    ) -> None:
        if components is None:
            components = ((CONST_AXIS, np.asarray([constant], dtype=object)),)
            for axis, coefficient in terms:
                component = (
                    coefficient
                    if axis.values is None
                    else scale_component(axis.values, coefficient)
                )
                components = merge_components(components, ((axis, component),))
        frozen = []
        for axis, component in components:
            if isinstance(component, np.ndarray):
                component = component.copy()
                component.flags.writeable = False
            if nonzero_component(component):
                frozen.append((axis, component))
        components = tuple(frozen)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "components", components)

    @cached_property
    def limits(self) -> CycleLimits:
        """Return this immutable variable's cached independent-axis bounds."""
        lower: Fraction | None = Fraction(0)
        upper: Fraction | None = Fraction(0)
        for axis, component in self.components:
            lo, hi = component_limits(axis, component)
            lower = None if lower is None or lo is None else lower + lo
            upper = None if upper is None or hi is None else upper + hi
        return lower, upper

    def __hash__(self) -> int:
        return hash(frozenset(component_key(axis, c) for axis, c in self.components))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TimeVariable) and frozenset(
            component_key(axis, c) for axis, c in self.components
        ) == frozenset(component_key(axis, c) for axis, c in other.components)


def merge_components(
    left: Components,
    right: Components,
) -> Components:
    """Add matched axis IDs, treating missing components as zero and deleting zeros."""
    groups = {axis.identity: (axis, component) for axis, component in left}
    for axis, component in right:
        previous = groups.get(axis.identity)
        if previous is not None:
            component = _component_add(previous[1], component)
        groups[axis.identity] = (axis, component)
    return tuple((axis, c) for axis, c in groups.values() if nonzero_component(c))


def component_limits(axis: CycleAxis, component: Component) -> CycleLimits:
    """Return known extrema or a coefficient-scaled range with unknown endpoints."""
    if isinstance(component, np.ndarray):
        return Fraction(component.min()), Fraction(component.max())
    a, b = (
        (axis.minimum, axis.maximum) if component >= 0 else (axis.maximum, axis.minimum)
    )
    return (
        None if a is None else Fraction(a * component),
        None if b is None else Fraction(b * component),
    )


