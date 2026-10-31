"""Axis-component cycle algebra. Proof envelopes never supply runtime values."""

import logging
import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from ._timeline_bounds import CycleBounds

CycleValues = NDArray[np.int64] | NDArray[np.object_]
Scalar = int | Fraction
MAX_BRANCHES = 8
MAX_CANDIDATES = 81
MAX_DEPTH = 2
MAX_LEAVES = 81


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
    return CycleAxis(name, identity, minimum, maximum, points, producer)


CONST_AXIS = CycleAxis("constant", "const", 1, 1, np.asarray([1]), CycleProducer())
Component = CycleValues | Scalar


def _component_key(axis: CycleAxis, component: Component) -> tuple[object, ...]:
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
    components: tuple[tuple[CycleAxis, Component], ...]

    def __init__(
        self,
        constant: Scalar = 0,
        terms: tuple[tuple[CycleAxis, Scalar], ...] = (),
        *,
        components: tuple[tuple[CycleAxis, Component], ...] | None = None,
    ) -> None:
        if components is None:
            components = ((CONST_AXIS, np.asarray([constant], dtype=object)),)
            for axis, coefficient in terms:
                component = (
                    coefficient if axis.values is None else axis.values * coefficient
                )
                components = _merge_components(components, ((axis, component),))
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "components", components)

    def __hash__(self) -> int:
        return hash(frozenset(_component_key(axis, c) for axis, c in self.components))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TimeVariable) and frozenset(
            _component_key(axis, c) for axis, c in self.components
        ) == frozenset(_component_key(axis, c) for axis, c in other.components)


@dataclass(frozen=True)
class TimeExtremum:
    """Bounded max/min proof branches, optional residual envelope and ledger.

    Each max branch is <= the true value; residual bounds all unrepresented
    parts from above. Min mirrors this. runtime_values retains the unpruned
    operation inputs solely for reconstruction, never for proof evaluation.
    """

    alternatives: tuple["TimeExpression", ...]
    maximum: bool
    residual: TimeVariable | None = None
    ledger: tuple[str, ...] = ()
    runtime_values: tuple["TimeExpression", ...] = field(
        default=(), compare=False, hash=False
    )


TimeExpression = TimeVariable | TimeExtremum


def _merge_components(
    left: tuple[tuple[CycleAxis, Component], ...],
    right: tuple[tuple[CycleAxis, Component], ...],
) -> tuple[tuple[CycleAxis, Component], ...]:
    groups = {axis.identity: (axis, component) for axis, component in left}
    for axis, component in right:
        previous = groups.get(axis.identity)
        if previous is not None:
            component = previous[1] + component
        groups[axis.identity] = (axis, component)
    return tuple((axis, c) for axis, c in groups.values() if np.any(c))


def _component_limits(
    axis: CycleAxis, component: Component
) -> tuple[Fraction | None, Fraction | None]:
    if isinstance(component, np.ndarray):
        return Fraction(component.min()), Fraction(component.max())
    a, b = (
        (axis.minimum, axis.maximum) if component >= 0 else (axis.maximum, axis.minimum)
    )
    return (
        None if a is None else Fraction(a * component),
        None if b is None else Fraction(b * component),
    )


def expression_limits(value: TimeExpression) -> tuple[Fraction | None, Fraction | None]:
    """Add independent component bounds; extrema use retained branches and R."""
    if isinstance(value, TimeExtremum):
        bounds = [expression_limits(v) for v in value.alternatives]
        if value.residual is not None:
            residual = expression_limits(value.residual)
            if value.maximum:
                bounds.append((None, residual[1]))
            else:
                bounds.append((residual[0], None))
        select = max if value.maximum else min
        lows = [lo for lo, _ in bounds if lo is not None]
        highs = [hi for _, hi in bounds if hi is not None]
        return (
            select(lows)
            if lows and (value.maximum or len(lows) == len(bounds))
            else None,
            select(highs)
            if highs and (not value.maximum or len(highs) == len(bounds))
            else None,
        )
    lower: Fraction | None = Fraction(0)
    upper: Fraction | None = Fraction(0)
    for axis, component in value.components:
        lo, hi = _component_limits(axis, component)
        lower = None if lower is None or lo is None else lower + lo
        upper = None if upper is None or hi is None else upper + hi
    return lower, upper


def expression_bounds(value: TimeExpression) -> CycleBounds:
    """Return finite cycle bounds; raise ValueError for an unavailable endpoint."""
    lower, upper = expression_limits(value)
    if lower is None or upper is None:
        raise ValueError("runtime-only time has no compile-time bounds")
    return CycleBounds(math.ceil(lower), math.floor(upper))


def unbounded_time(value: TimeExpression) -> bool:
    """Return whether the proof lacks either endpoint."""
    return any(endpoint is None for endpoint in expression_limits(value))


def time_variables(value: TimeExpression) -> tuple[CycleAxis, ...]:
    """Return surviving runtime recipe identities in traversal order."""
    if isinstance(value, TimeVariable):
        return tuple(axis for axis, _ in value.terms)
    return tuple(
        dict.fromkeys(
            axis
            for v in (
                *value.alternatives,
                *((value.residual,) if value.residual else ()),
            )
            for axis in time_variables(v)
        )
    )


def expression_axes(value: TimeExpression) -> dict[str, int]:
    """Return known loop dimensions without consulting source recipes for bounds."""
    if isinstance(value, TimeExtremum):
        return {
            name: count
            for v in value.alternatives
            for name, count in expression_axes(v).items()
        }
    return {
        axis.identity: len(c)
        for axis, c in value.components
        if isinstance(axis.identity, str)
        and axis.identity != "const"
        and isinstance(c, np.ndarray)
    }


def evaluate_time(value: TimeExpression, axes: dict[str, int]) -> CycleValues | None:
    """Evaluate known components for getters, never for multi-axis range proofs.

    Unknown or compressed extrema return None, not an envelope masquerading as
    a runtime value. Caller dimensions determine output broadcasting order.
    """
    if isinstance(value, TimeExtremum):
        if value.residual is not None:
            return None
        result = None
        select = np.maximum if value.maximum else np.minimum
        for alternative in value.alternatives:
            points = evaluate_time(alternative, axes)
            if points is None:
                return None
            result = points if result is None else select(result, points)
        return result
    result = np.asarray(0, dtype=object)
    for axis, component in value.components:
        if not isinstance(component, np.ndarray):
            if axis.minimum is None or axis.minimum != axis.maximum:
                return None
            result = result + component * axis.minimum
        elif axis.identity == "const":
            result = result + component[0]
        else:
            shape = tuple(
                len(component) if name == axis.identity else 1 for name in axes
            )
            result = result + component.reshape(shape)
    return result


def ceiled_time(value: TimeExpression) -> TimeExpression:
    """Ceil single-axis components exactly; cross-axis branches get fresh errors."""
    if isinstance(value, TimeExtremum):
        return choose_time(
            *(ceiled_time(v) for v in value.alternatives),
            maximum=value.maximum,
            residual=ceiled_time(value.residual) if value.residual else None,
        )
    axes = expression_axes(value)
    if len(axes) <= 1:
        points = evaluate_time(value, axes)
        if points is not None:
            rounded = np.asarray(
                [_ceil_fraction(Fraction(v)) for v in points.flat], dtype=np.int64
            ).reshape(points.shape)
            return point_time("ceil cycles", axes, rounded, reconstruction=value)
    denominator = math.lcm(
        Fraction(value.constant).denominator,
        *(Fraction(c).denominator for _, c in value.terms),
    )
    if denominator == 1:
        return value
    error = cycle_axis(
        "ceil rounding", -math.ceil(Fraction(denominator, 10**10)), denominator - 1
    )
    return add_time(value, TimeVariable(terms=((error, Fraction(1, denominator)),)))


def _ceil_fraction(value: Fraction) -> int:
    nearest = round(value)
    return nearest if abs(value - nearest) <= Fraction(1, 10**10) else math.ceil(value)


def shifted_time(value: TimeExpression, shift: int) -> TimeExpression:
    """Arithmetic right shift: one-axis exact values, otherwise bounded errors."""
    if isinstance(value, TimeExtremum):
        residual = shifted_time(value.residual, shift) if value.residual else None
        return choose_time(
            *(shifted_time(v, shift) for v in value.alternatives),
            maximum=value.maximum,
            residual=residual,
        )
    axes = expression_axes(value)
    if len(axes) <= 1:
        points = evaluate_time(value, axes)
        if points is not None:
            values = np.asarray(
                [int(v) >> shift for v in points.flat], dtype=np.int64
            ).reshape(points.shape)
            variable = cycle_axis(
                "shifted cycles",
                int(values.min()),
                int(values.max()),
                CycleProducer(
                    tuple(axes.items()),
                    tuple(int(v) for v in values.flat),
                    fixed_point=(value, shift),
                ),
            )
            return (
                TimeVariable(terms=((variable, 1),))
                if axes
                else TimeVariable(int(values))
            )
    divisor = 1 << shift
    remainder = cycle_axis(
        "rounding remainder",
        0,
        divisor - 1,
        CycleProducer(
            tuple(axes.items()), fixed_point=(value, shift), rounding_remainder=True
        ),
    )
    return add_time(
        scale_time(value, Fraction(1, divisor)),
        TimeVariable(terms=((remainder, Fraction(-1, divisor)),)),
    )


def exact_bounds(value: TimeExpression) -> CycleBounds | None:
    """Return bounds only when extrema are proved attainable, not just safe."""
    if unbounded_time(value):
        return None
    limits = expression_bounds(value)
    if limits.minimum == limits.maximum:
        return limits
    if isinstance(value, TimeExtremum):
        return None
    if any(not isinstance(c, np.ndarray) for _, c in value.components):
        return None
    return limits


def projected_expression_bounds(value: TimeExpression) -> CycleBounds:
    """Project the common axis-component proof into integer cycles."""
    return expression_bounds(value)


def slack_variables(value: TimeExpression) -> list[str]:
    """Name nonconstant source recipes for diagnostics, without grid evaluation."""
    return sorted({v.name for v in time_variables(value) if v.minimum != v.maximum})


def point_time(
    name: str,
    axes: dict[str, int],
    points: CycleValues,
    *,
    reconstruction: TimeExpression | None = None,
) -> TimeExpression:
    """Capture a single known axis; nonlinear cross-axis results become ranges."""
    lower, upper = int(points.min()), int(points.max())
    if lower == upper:
        return TimeVariable(lower)
    variable = cycle_axis(
        name,
        lower,
        upper,
        CycleProducer(
            tuple(axes.items()),
            tuple(int(v) for v in points.flat),
            reconstruction=reconstruction,
        ),
    )
    return TimeVariable(terms=((variable, 1),))


def reduce_loop_time(
    value: TimeExpression, axis: str, *, last: bool = False
) -> TimeExpression:
    """Reduce a local loop component without constructing a Cartesian grid."""
    if isinstance(value, TimeExtremum):
        # A nonlinear reduction cannot be distributed through max/min.
        lo, hi = expression_limits(value)
        count = expression_axes(value)[axis]
        factor = 1 if last else count
        return TimeVariable(
            terms=(
                (
                    cycle_axis(
                        "loop reduction",
                        None if lo is None else math.floor(lo * factor),
                        None if hi is None else math.ceil(hi * factor),
                    ),
                    1,
                ),
            )
        )
    components = []
    count = expression_axes(value)[axis]
    for source, c in value.components:
        if source.identity == axis and isinstance(c, np.ndarray):
            amount = c[-1] if last else c.sum()
            components.append((CONST_AXIS, np.asarray([amount], dtype=object)))
        else:
            components.append((source, c if last else c * count))
    merged: tuple[tuple[CycleAxis, Component], ...] = ()
    for component in components:
        merged = _merge_components(merged, (component,))
    return TimeVariable(components=merged)


def multiply_time(left: TimeExpression, right: TimeExpression) -> TimeExpression:
    """Scale constants; same-axis multiplication stays known, otherwise bound."""
    if isinstance(right, TimeVariable) and not right.terms:
        return scale_time(left, right.constant)
    if isinstance(left, TimeVariable) and not left.terms:
        return scale_time(right, left.constant)
    axes = expression_axes(left) | expression_axes(right)
    if len(axes) <= 1:
        a, b = evaluate_time(left, axes), evaluate_time(right, axes)
        if a is not None and b is not None:
            return point_time("product", axes, a * b)
    if unbounded_time(left) or unbounded_time(right):
        return TimeVariable(terms=((cycle_axis("product", None, None), 1),))
    a, b = expression_bounds(left), expression_bounds(right)
    products = [x * y for x in (a.minimum, a.maximum) for y in (b.minimum, b.maximum)]
    return TimeVariable(
        terms=((cycle_axis("product", min(products), max(products)), 1),)
    )


def select_time(
    selector: TimeExpression, values: tuple[TimeExpression, ...]
) -> TimeExpression:
    """Retain common components; one-axis selection is exact, others independent."""
    if all(value == values[0] for value in values):
        return values[0]
    if all(isinstance(value, TimeVariable) for value in values):
        first = values[0]
        if isinstance(first, TimeVariable):
            common = tuple(
                (v, c)
                for v, c in first.terms
                if all(
                    isinstance(value, TimeVariable) and (v, c) in value.terms
                    for value in values
                )
            )
            if common:
                origin = TimeVariable(terms=common)
                return add_time(
                    origin,
                    select_time(
                        selector,
                        tuple(
                            add_time(value, scale_time(origin, -1)) for value in values
                        ),
                    ),
                )
    axes = expression_axes(selector)
    for value in values:
        axes.update(expression_axes(value))
    if len(axes) <= 1:
        index = evaluate_time(selector, axes)
        points = [evaluate_time(value, axes) for value in values]
        if index is not None and all(p is not None for p in points):
            result = np.asarray(0, dtype=object)
            for case, point in enumerate(points):
                if point is not None:
                    result = np.where(index == case, point, result)
            return point_time("selected cycles", axes, result)
    if any(unbounded_time(v) for v in values):
        return TimeVariable(terms=((cycle_axis("selected cycles", None, None), 1),))
    limits = [expression_bounds(v) for v in values]
    return TimeVariable(
        terms=(
            (
                cycle_axis(
                    "selected cycles",
                    min(v.minimum for v in limits),
                    max(v.maximum for v in limits),
                ),
                1,
            ),
        )
    )


def _envelope(values: tuple[TimeExpression, ...], *, upper: bool) -> TimeVariable:
    """Bound every input by an additive per-axis envelope (missing axes are 0)."""
    linear = [v for v in values if isinstance(v, TimeVariable)]
    if len(linear) != len(values):
        limits = [expression_limits(v)[1 if upper else 0] for v in values]
        endpoint = (
            None
            if any(v is None for v in limits)
            else (max if upper else min)(v for v in limits if v is not None)
        )
        return (
            TimeVariable(
                terms=(
                    (
                        cycle_axis(
                            "residual envelope",
                            None if endpoint is None else math.floor(endpoint),
                            None if endpoint is None else math.ceil(endpoint),
                        ),
                        1,
                    ),
                )
            )
            if endpoint is None
            else TimeVariable(endpoint)
        )
    sources = {axis.identity: axis for v in linear for axis, _ in v.components}
    components = []
    for identity, axis in sources.items():
        candidates = [
            next((c for a, c in v.components if a.identity == identity), 0)
            for v in linear
        ]
        if axis.values is not None:
            arrays = np.broadcast_arrays(*[np.asarray(c) for c in candidates])
            component = (np.maximum if upper else np.minimum).reduce(arrays)
            components.append((axis, component))
        elif all(c == candidates[0] for c in candidates):
            components.append((axis, candidates[0]))
        else:
            endpoints = [
                _component_limits(axis, c)[1 if upper else 0] for c in candidates
            ]
            if any(v is None for v in endpoints):
                return TimeVariable(
                    terms=((cycle_axis("residual envelope", None, None), 1),)
                )
            endpoint = (max if upper else min)(v for v in endpoints if v is not None)
            components.append((CONST_AXIS, np.asarray([endpoint], dtype=object)))
    merged: tuple[tuple[CycleAxis, Component], ...] = ()
    for component in components:
        merged = _merge_components(merged, (component,))
    return TimeVariable(components=merged)


def _dominates(left: TimeExpression, right: TimeExpression, maximum: bool) -> bool:
    if not isinstance(left, TimeVariable) or not isinstance(right, TimeVariable):
        return False
    lower, upper = expression_limits(add_time(left, scale_time(right, -1)))
    return (
        lower is not None and lower >= 0
        if maximum
        else upper is not None and upper <= 0
    )


def _size(value: TimeExpression) -> tuple[int, int]:
    if isinstance(value, TimeVariable):
        return 0, 1
    sizes = [_size(v) for v in value.alternatives]
    return 1 + max(d for d, _ in sizes), sum(n for _, n in sizes) + int(
        value.residual is not None
    )


def _collapse_deepest(value: TimeExtremum) -> TimeExtremum:
    branches = list(value.alternatives)
    nested = [(i, v) for i, v in enumerate(branches) if isinstance(v, TimeExtremum)]
    if not nested:
        return value
    index, child = max(nested, key=lambda item: _size(item[1])[0])
    if any(isinstance(v, TimeExtremum) for v in child.alternatives):
        branches[index] = _collapse_deepest(child)
        return TimeExtremum(
            tuple(branches),
            value.maximum,
            value.residual,
            (*value.ledger, "deepest subtree"),
            value.runtime_values,
        )
    # max contains min: retain a lower envelope, cover the missing part with
    # the min branch having the lowest upper bound. Min mirrors this rule.
    branches[index] = _envelope(child.alternatives, upper=not value.maximum)
    candidates = (*child.alternatives, *((child.residual,) if child.residual else ()))

    def rank(v: TimeExpression) -> Fraction:
        lo, hi = expression_limits(v)
        endpoint = hi if value.maximum else lo
        return Fraction(0) if endpoint is None else endpoint

    covering = (min if value.maximum else max)(candidates, key=rank)
    residuals = (covering, *((value.residual,) if value.residual else ()))
    residual = _envelope(residuals, upper=value.maximum)
    event = "collapse max/min subtree" if value.maximum else "collapse min/max subtree"
    logging.getLogger(__name__).info("Timeline proof ledger: %s", event)
    return TimeExtremum(
        tuple(branches),
        value.maximum,
        residual,
        (*value.ledger, event),
        value.runtime_values,
    )


def choose_time(
    *values: TimeExpression,
    maximum: bool = True,
    residual: TimeExpression | None = None,
) -> TimeExpression:
    """Simplify a pointwise extremum with m/K/depth/leaf proof budgets."""
    if not values:
        raise ValueError("time extremum requires at least one alternative")
    alternatives: list[TimeExpression] = []
    residues = [] if residual is None else [residual]
    ledger = []
    for value in values:
        if isinstance(value, TimeExtremum) and value.maximum == maximum:
            alternatives.extend(value.alternatives)
            if value.residual is not None:
                residues.append(value.residual)
            ledger.extend(value.ledger)
        else:
            alternatives.append(value)
    unique = list(dict.fromkeys(alternatives))
    # Merge every group involving one known axis, including constants. This
    # happens before cross-axis bounds, so opposing sweeps retain their max.
    groups: dict[str, list[TimeVariable]] = {}
    for value in unique:
        axes = expression_axes(value)
        if (
            isinstance(value, TimeVariable)
            and len(axes) == 1
            and evaluate_time(value, axes) is not None
        ):
            groups.setdefault(next(iter(axes)), []).append(value)
    for axis, group in groups.items():
        if len(group) < 2:
            continue
        axes = expression_axes(group[0])
        points = [evaluate_time(v, axes) for v in group]
        known = [p for p in points if p is not None]
        merged = (np.maximum if maximum else np.minimum).reduce(
            np.broadcast_arrays(*known)
        )
        names = ", ".join(
            sorted({v.name for item in group for v in time_variables(item)})
        )
        replacement = point_time(
            names, axes, merged, reconstruction=TimeExtremum(tuple(group), maximum)
        )
        unique = [v for v in unique if v not in group]
        unique.append(replacement)
    # Cheap bound filter, then aligned-axis subtraction. Keep one on ties.
    limits = [expression_limits(v) for v in unique]
    endpoints = [lo if maximum else hi for lo, hi in limits]
    finite = [v for v in endpoints if v is not None]
    threshold = (max if maximum else min)(finite) if finite else None
    kept: list[TimeExpression] = []
    for index, value in enumerate(unique):
        lo, hi = limits[index]
        bound = hi if maximum else lo
        if (
            threshold is not None
            and bound is not None
            and (bound < threshold if maximum else bound > threshold)
        ):
            continue
        if any(_dominates(other, value, maximum) for other in kept):
            continue
        kept = [other for other in kept if not _dominates(value, other, maximum)]
        kept.append(value)
    if len(kept) > MAX_BRANCHES:

        def rank(v: TimeExpression) -> tuple[float, float]:
            lo, hi = expression_limits(v)
            return (
                float(hi) if hi is not None else math.inf,
                float(lo) if lo is not None else -math.inf,
            )

        kept.sort(key=rank, reverse=maximum)
        residues.extend(kept[MAX_BRANCHES:])
        ledger.append(f"branches {len(kept)} -> {MAX_BRANCHES}")
        kept = kept[:MAX_BRANCHES]
    remainder = _envelope(tuple(residues), upper=maximum) if residues else None
    if remainder is not None and any(_dominates(v, remainder, maximum) for v in kept):
        remainder = None
    if len(kept) == 1 and remainder is None:
        return kept[0]
    result = TimeExtremum(tuple(kept), maximum, remainder, tuple(ledger), values)
    while _size(result)[0] > MAX_DEPTH or _size(result)[1] > MAX_LEAVES:
        collapsed = _collapse_deepest(result)
        if collapsed == result:
            break
        result = collapsed
    return result


def substitute_time(
    value: TimeExpression, replacements: dict[CycleAxis, TimeExpression]
) -> TimeExpression:
    """Substitute immutable runtime identities for Repeat reduction."""
    if isinstance(value, TimeExtremum):
        return choose_time(
            *(substitute_time(v, replacements) for v in value.alternatives),
            maximum=value.maximum,
            residual=substitute_time(value.residual, replacements)
            if value.residual
            else None,
        )
    result: TimeExpression = TimeVariable(value.constant)
    for variable, coefficient in value.terms:
        result = add_time(
            result,
            scale_time(
                replacements.get(variable, TimeVariable(terms=((variable, 1),))),
                coefficient,
            ),
        )
    return result


def scale_time(value: TimeExpression, coefficient: Scalar) -> TimeExpression:
    """Scale components; a negative factor swaps max/min and residual direction."""
    if coefficient == 0:
        return TimeVariable()
    if isinstance(value, TimeExtremum):
        return choose_time(
            *(scale_time(v, coefficient) for v in value.alternatives),
            maximum=value.maximum if coefficient > 0 else not value.maximum,
            residual=scale_time(value.residual, coefficient)
            if value.residual
            else None,
        )
    return TimeVariable(
        value.constant * coefficient,
        tuple((v, c * coefficient) for v, c in value.terms if c * coefficient),
        components=tuple(
            (axis, c * coefficient)
            for axis, c in value.components
            if np.any(c * coefficient)
        ),
    )


def add_time(left: TimeExpression, right: TimeExpression) -> TimeExpression:
    """Align axes and distribute proof branches, with at most K candidates."""
    if isinstance(left, TimeExtremum):
        # Each node has <=m branches plus R; one product therefore fits K.
        candidates = tuple(add_time(v, right) for v in left.alternatives)
        remainder = add_time(left.residual, right) if left.residual else None
        return choose_time(*candidates, maximum=left.maximum, residual=remainder)
    if isinstance(right, TimeExtremum):
        return add_time(right, left)
    coefficients = dict(left.terms)
    for axis, coefficient in right.terms:
        coefficients[axis] = coefficients.get(axis, 0) + coefficient
    return TimeVariable(
        left.constant + right.constant,
        tuple((v, c) for v, c in coefficients.items() if c),
        components=_merge_components(left.components, right.components),
    )
