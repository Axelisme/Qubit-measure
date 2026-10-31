"""Compile-time cycle algebra, independent of register allocation and emission."""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from ._timeline_bounds import CycleBounds


@dataclass(frozen=True)
class CycleIntervalGrid:
    """Coordinate-aware inclusive bounds in integer tProc cycles.

    minimum and maximum are int64 arrays broadcastable to the caller's ordered
    loop-axis grid. At every coordinate minimum <= maximum. These endpoints
    bound unknown inputs, not necessarily attainable values. One result keeps
    the paired grids and their units together for all expression consumers.
    Construction stores arrays without copying or changing them.
    """

    minimum: NDArray[np.int64]
    maximum: NDArray[np.int64]


@dataclass(frozen=True)
class VariableRange:
    """One source's directed cycle endpoints and deviation from their line.

    source is a loop name for a one-loop producer, otherwise a fresh identity.
    start/end follow iteration order, not numerical order. None marks an
    unbounded source. deviation_minimum/maximum enclose the signed distance
    from the endpoint line, in exact fractional cycles; affine sources use zero.
    """

    source: str | object
    start: int | None
    end: int | None
    deviation_minimum: tuple[Fraction, ...] = (Fraction(0),)
    deviation_maximum: tuple[Fraction, ...] = (Fraction(0),)


@dataclass(frozen=True)
class CycleProducer:
    """Immutable runtime recipe, separate from the range used by proofs.

    axes/points retain exact integer values in row-major loop order for Repeat
    reduction and exact-value getters. One axis without points is its loop index.
    interval_points retains unknown-input envelopes for Repeat accumulation.
    reconstruction is integer algebra for an affine lookup; selection is the
    captured selector and declared words. fixed_point holds numerator and shift.
    None means that recipe is unavailable, never that a range is unbounded.
    """

    axes: tuple[tuple[str, int], ...] = ()
    points: tuple[int, ...] | None = None
    interval_points: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    reconstruction: "TimeExpression | None" = None
    selection: "tuple[TimeExpression, tuple[int, ...]] | None" = None
    fixed_point: "tuple[TimeExpression, int] | None" = None


@dataclass(frozen=True, eq=False)
class CycleVariable:
    """One immutable runtime value with diagnostic name, range and recipe.

    Object identity distinguishes writes even when register names are recycled.
    bounds is the sole proof representation; producer never supplies bounds to
    expression consumers. minimum/maximum project that representation in cycles.
    """

    name: str
    bounds: VariableRange
    producer: CycleProducer

    @property
    def minimum(self) -> int | None:
        """Return the conservative integer lower bound, or None when unbounded."""
        r = self.bounds
        if r.start is None or r.end is None:
            return None
        return expression_bounds(LinearTime(terms=((self, 1),))).minimum

    @property
    def maximum(self) -> int | None:
        """Return the conservative integer upper bound, or None when unbounded."""
        r = self.bounds
        if r.start is None or r.end is None:
            return None
        return expression_bounds(LinearTime(terms=((self, 1),))).maximum


def cycle_variable(
    name: str,
    minimum: int | None,
    maximum: int | None,
    axes: tuple[tuple[str, int], ...] = (),
    points: tuple[int, ...] | None = None,
    *,
    interval_points: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    reconstruction: "TimeExpression | None" = None,
    selection: "tuple[TimeExpression, tuple[int, ...]] | None" = None,
    fixed_point: "tuple[TimeExpression, int] | None" = None,
) -> CycleVariable:
    """Capture a producer's range once, without evaluating expression grids.

    A known one-loop sequence shares that loop's ID. Its directed endpoints and
    exact deviations are computed only along that source. Unknown values and
    multi-loop nonlinear results receive independent IDs, even with bounds.
    Recipe metadata survives independently for runtime reconstruction.
    """
    source: str | object = object()
    start, end = minimum, maximum
    low = high = (Fraction(0),)
    if len(axes) == 1:
        source, count = axes[0]
        low = high = (Fraction(0),) * count
        if interval_points is not None or points is not None:
            lower, upper = interval_points if interval_points is not None else (points, points)
            if lower is None or upper is None:
                raise RuntimeError("point producer has no values")
            start, end = lower[0], lower[-1]
            line = tuple(
                Fraction(start) + Fraction((end - start) * i, max(1, count - 1))
                for i in range(count)
            )
            low = tuple(Fraction(value) - origin for value, origin in zip(lower, line, strict=True))
            high = tuple(Fraction(value) - origin for value, origin in zip(upper, line, strict=True))
    return CycleVariable(
        name,
        VariableRange(source, start, end, low, high),
        CycleProducer(axes, points, interval_points, reconstruction, selection, fixed_point),
    )


@dataclass(frozen=True)
class LinearTime:
    """Integer cycles as constant + sum(coefficient * variable).

    constant is a signed cycle offset. terms contains distinct runtime values
    and nonzero integer coefficients. Algebra merges coefficients before
    bounding, so shared origins cancel even when register storage differs.
    """

    constant: int = 0
    terms: tuple[tuple[CycleVariable, int], ...] = ()


@dataclass(frozen=True)
class TimeChoice:
    """A pointwise max or min of nonempty cycle expressions.

    alternatives retains the competing linear paths; maximum selects max when
    true and min otherwise. Arithmetic distributes through choices, including
    reversing max/min under negation. This can grow with the product of path
    counts; flattening and deduplication remove repeated alternatives.
    """

    alternatives: tuple["TimeExpression", ...]
    maximum: bool


TimeExpression = LinearTime | TimeChoice


def expression_bounds(value: TimeExpression) -> CycleBounds:
    """Group directed endpoints by source ID, then add coefficient deviations.

    Different sources are independent. Any surviving unbounded term raises
    ValueError. This rule never enumerates a loop grid, including for max paths.
    """
    if isinstance(value, TimeChoice):
        limits = [expression_bounds(item) for item in value.alternatives]
        select = max if value.maximum else min
        return CycleBounds(
            select(limit.minimum for limit in limits),
            select(limit.maximum for limit in limits),
        )
    groups: dict[str | object, list[tuple[VariableRange, int]]] = {}
    lower = upper = value.constant
    for variable, coefficient in value.terms:
        r = variable.bounds
        if r.start is None or r.end is None:
            raise ValueError("runtime-only time has no compile-time bounds")
        groups.setdefault(r.source, []).append((r, coefficient))
    for source, terms in groups.items():
        if isinstance(source, str):
            count = len(terms[0][0].deviation_minimum)
            lows = [Fraction(0)] * count
            highs = [Fraction(0)] * count
            for r, coefficient in terms:
                if r.start is None or r.end is None:
                    raise RuntimeError("bounded source lost its endpoints")
                for i in range(count):
                    line = Fraction(r.start) + Fraction((r.end - r.start) * i, max(1, count - 1))
                    a = coefficient * (line + r.deviation_minimum[i])
                    b = coefficient * (line + r.deviation_maximum[i])
                    lows[i] += min(a, b)
                    highs[i] += max(a, b)
            lower += math.ceil(min(lows))
            upper += math.floor(max(highs))
        else:
            for r, coefficient in terms:
                if r.start is None or r.end is None:
                    raise RuntimeError("bounded source lost its endpoints")
                endpoints = coefficient * r.start, coefficient * r.end
                lower += min(endpoints)
                upper += max(endpoints)
    return CycleBounds(lower, upper)


def unbounded_time(value: TimeExpression) -> bool:
    """Return whether surviving identities lack author or producer bounds."""
    return any(v.minimum is None or v.maximum is None for v in time_variables(value))


def time_variables(value: TimeExpression) -> tuple[CycleVariable, ...]:
    """Return surviving runtime identities in expression traversal order."""
    if isinstance(value, LinearTime):
        return tuple(variable for variable, _ in value.terms)
    return tuple(
        dict.fromkeys(v for item in value.alternatives for v in time_variables(item))
    )


def expression_axes(value: TimeExpression) -> dict[str, int]:
    """Return only the loop dimensions used by this expression's producers."""
    return dict(axis for variable in time_variables(value) for axis in variable.producer.axes)


def evaluate_time(
    value: TimeExpression, axes: dict[str, int]
) -> NDArray[np.int64] | None:
    """Evaluate legal points by broadcasting shared axes; None means unknown input.

    axes contains the union of dimensions requested by the caller. Missing
    producer dimensions raise KeyError rather than silently losing correlation.
    """
    if isinstance(value, TimeChoice):
        result = None
        select = np.maximum if value.maximum else np.minimum
        for alternative in value.alternatives:
            points = evaluate_time(alternative, axes)
            if points is None:
                return None
            result = points if result is None else select(result, points)
        return result
    result = np.asarray(value.constant, dtype=np.int64)
    for variable, coefficient in value.terms:
        if variable.minimum is None or variable.maximum is None:
            return None
        if variable.minimum == variable.maximum:
            result = result + coefficient * variable.minimum
            continue
        if not variable.producer.axes or variable.producer.interval_points is not None:
            return None
        local = dict(variable.producer.axes)
        points = (
            np.arange(variable.maximum + 1, dtype=np.int64)
            if variable.producer.points is None
            else np.asarray(variable.producer.points, dtype=np.int64).reshape(
                tuple(local.values())
            )
        )
        # Producer dimensions may appear in a different order from the union.
        order = [name for name in axes if name in local]
        points = points.transpose(tuple(list(local).index(name) for name in order))
        shape = tuple(local.get(name, 1) for name in axes)
        result = result + coefficient * points.reshape(shape)
    return result


def exact_bounds(value: TimeExpression) -> CycleBounds | None:
    """Return exact legal-point extrema, or None for a nonconstant unknown input.

    Algebraic constants use no Cartesian allocation. Other expressions evaluate
    only the appearing loop axes, not all of the program's sweep dimensions.
    """
    if unbounded_time(value):
        points = evaluate_time(value, expression_axes(value))
        return (
            None
            if points is None
            else CycleBounds(int(points.min()), int(points.max()))
        )
    limits = expression_bounds(value)
    if limits.minimum == limits.maximum:
        return limits
    points = evaluate_time(value, expression_axes(value))
    return None if points is None else CycleBounds(int(points.min()), int(points.max()))


def bound_unknowns(value: TimeExpression, *, upper: bool) -> TimeExpression:
    """Substitute only unknown inputs by coefficient-aware interval endpoints."""
    if isinstance(value, TimeChoice):
        return choose_time(
            *(bound_unknowns(v, upper=upper) for v in value.alternatives),
            maximum=value.maximum,
        )
    constant = value.constant
    terms = []
    for variable, coefficient in value.terms:
        if variable.producer.interval_points is not None:
            index = int(upper == (coefficient > 0))
            points = variable.producer.interval_points[index]
            endpoint = cycle_variable(
                variable.name, min(points), max(points), variable.producer.axes, points
            )
            terms.append((endpoint, coefficient))
        elif variable.producer.axes:
            terms.append((variable, coefficient))
        else:
            endpoint = (
                variable.maximum if upper == (coefficient > 0) else variable.minimum
            )
            if endpoint is None:
                raise ValueError("runtime-only time has no compile-time bounds")
            constant += coefficient * endpoint
    return LinearTime(constant, tuple(terms))


def projected_expression_bounds(value: TimeExpression) -> CycleBounds:
    """Project cycles through the common directed-source range rule."""
    return expression_bounds(value)

def slack_variables(value: TimeExpression) -> list[str]:
    """Name unknown sources and known producers on axes that change the tail."""
    axes = expression_axes(value)
    changing: set[str] = set()
    grids = interval_arrays(value, axes)
    for points in (grids.minimum, grids.maximum):
        points = np.broadcast_to(points, tuple(axes.values()))
        for dimension, name in enumerate(axes):
            if np.any(np.diff(points, axis=dimension)):
                changing.add(name)
    return sorted(
        {
            v.name
            for v in time_variables(value)
            if v.minimum != v.maximum
            and (
                not v.producer.axes
                or v.producer.interval_points is not None
                or any(name in changing for name, _ in v.producer.axes)
            )
        }
    )


def point_time(
    name: str, axes: dict[str, int], points: NDArray[np.int64],
    *, reconstruction: TimeExpression | None = None,
) -> TimeExpression:
    """Retain a known nonlinear grid as a bounded variable, or a scalar constant."""
    lower, upper = int(points.min()), int(points.max())
    if lower == upper:
        return LinearTime(lower)
    variable = cycle_variable(
        name, lower, upper, tuple(axes.items()), tuple(int(v) for v in points.flat),
        reconstruction=reconstruction,
    )
    return LinearTime(terms=((variable, 1),))


def reduce_loop_time(
    value: TimeExpression, axis: str, *, last: bool = False
) -> TimeExpression:
    """Sum a fixed iteration grid, or retain its last value for a later read.

    Remove only the local loop axis. Unknown interval grids retain exact known
    coordinates while summing bounds rather than claiming actual extrema.
    """
    axes = expression_axes(value)
    dimension = list(axes).index(axis)
    shape = tuple(axes.values())
    remaining = {name: count for name, count in axes.items() if name != axis}
    name = ", ".join(sorted({v.name for v in time_variables(value)}))

    def reduce(points: NDArray[np.int64]) -> NDArray[np.int64]:
        grid = np.broadcast_to(points, shape)
        reduced = (
            grid.take(-1, axis=dimension)
            if last
            else grid.sum(axis=dimension, dtype=np.int64)
        )
        return np.asarray(reduced, dtype=np.int64)

    points = evaluate_time(value, axes)
    if points is not None:
        return point_time(name, remaining, reduce(points))
    if unbounded_time(value):
        return LinearTime(terms=((cycle_variable(name, None, None), 1),))
    grids = interval_arrays(value, axes)
    return interval_time(name, remaining, reduce(grids.minimum), reduce(grids.maximum))


def interval_time(
    name: str, axes: dict[str, int], lower: NDArray[np.int64], upper: NDArray[np.int64]
) -> TimeExpression:
    """Retain known-coordinate bounds of a nonlinear unknown expression."""
    lower, upper = np.broadcast_arrays(lower, upper)
    if np.array_equal(lower, upper):
        return point_time(name, axes, lower)
    variable = cycle_variable(
        name,
        int(lower.min()),
        int(upper.max()),
        tuple(axes.items()),
        interval_points=(
            tuple(int(v) for v in lower.flat),
            tuple(int(v) for v in upper.flat),
        ),
    )
    return LinearTime(terms=((variable, 1),))


def interval_arrays(value: TimeExpression, axes: dict[str, int]) -> CycleIntervalGrid:
    """Bound each coordinate in integer tProc cycles on the ordered axes grid.

    Substitute unknown inputs without discarding known coordinates. Raise
    RuntimeError if an unknown survives substitution; missing axes raise KeyError.
    """
    lower = evaluate_time(bound_unknowns(value, upper=False), axes)
    upper = evaluate_time(bound_unknowns(value, upper=True), axes)
    if lower is None or upper is None:
        raise RuntimeError("unknown input survived interval substitution")
    return CycleIntervalGrid(lower, upper)


def multiply_time(left: TimeExpression, right: TimeExpression) -> TimeExpression:
    """Multiply compile-time values, retaining known nonlinear legal points."""
    if isinstance(right, LinearTime) and not right.terms:
        return scale_time(left, right.constant)
    if isinstance(left, LinearTime) and not left.terms:
        return scale_time(right, left.constant)
    axes = expression_axes(left) | expression_axes(right)
    a, b = evaluate_time(left, axes), evaluate_time(right, axes)
    name = " * ".join(
        sorted({v.name for item in (left, right) for v in time_variables(item)})
    )
    if a is not None and b is not None:
        return point_time(name, axes, a * b)
    if unbounded_time(left) or unbounded_time(right):
        return LinearTime(terms=((cycle_variable(name, None, None), 1),))
    left_grids = interval_arrays(left, axes)
    right_grids = interval_arrays(right, axes)
    products = [
        a * b
        for a in (left_grids.minimum, left_grids.maximum)
        for b in (right_grids.minimum, right_grids.maximum)
    ]
    grids = np.broadcast_arrays(*products)
    return interval_time(name, axes, np.minimum.reduce(grids), np.maximum.reduce(grids))


def select_time(
    selector: TimeExpression, values: tuple[TimeExpression, ...]
) -> TimeExpression:
    """Select a branch or table candidate at the same legal loop coordinates."""
    if all(value == values[0] for value in values):
        return values[0]
    if all(isinstance(value, LinearTime) for value in values):
        first = values[0]
        if isinstance(first, LinearTime):
            common = tuple(
                (variable, coefficient)
                for variable, coefficient in first.terms
                if all(
                    isinstance(value, LinearTime)
                    and (variable, coefficient) in value.terms
                    for value in values
                )
            )
            if common:
                origin = LinearTime(terms=common)
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
    index = evaluate_time(selector, axes)
    name = ", ".join(
        sorted({v.name for value in (selector, *values) for v in time_variables(value)})
    )
    if any(unbounded_time(value) for value in (selector, *values)):
        return LinearTime(terms=((cycle_variable(name, None, None), 1),))
    grids = [interval_arrays(value, axes) for value in values]
    lower_grids = [grid.minimum for grid in grids]
    upper_grids = [grid.maximum for grid in grids]
    if index is None:
        lower = np.minimum.reduce(np.broadcast_arrays(*lower_grids))
        upper = np.maximum.reduce(np.broadcast_arrays(*upper_grids))
    else:
        lower = upper = np.zeros(tuple(axes.values()), dtype=np.int64)
        for case, (lo, hi) in enumerate(zip(lower_grids, upper_grids, strict=True)):
            lower = np.where(index == case, lo, lower)
            upper = np.where(index == case, hi, upper)
    return interval_time(name, axes, lower, upper)


def choose_time(*values: TimeExpression, maximum: bool = True) -> TimeExpression:
    """Keep a pointwise extremum; reject empty input with ValueError."""
    if not values:
        raise ValueError("time extremum requires at least one alternative")
    alternatives: list[TimeExpression] = []
    for value in values:
        if isinstance(value, TimeChoice) and value.maximum == maximum:
            alternatives.extend(value.alternatives)
        else:
            alternatives.append(value)
    unique = tuple(dict.fromkeys(alternatives))
    # Shared unbounded identities cancel in comparisons; only constants remain.
    unique = tuple(
        value
        for value in unique
        if not isinstance(value, LinearTime)
        or not any(
            isinstance(other, LinearTime)
            and other.terms == value.terms
            and (
                other.constant > value.constant
                if maximum
                else other.constant < value.constant
            )
            for other in unique
        )
    )
    if len(unique) == 1:
        return unique[0]
    if any(unbounded_time(value) for value in unique):
        return TimeChoice(unique, maximum)
    limits = [expression_bounds(value) for value in unique]
    for index, limit in enumerate(limits):
        if all(
            index == other
            or (
                limit.minimum >= alternative.maximum
                if maximum
                else limit.maximum <= alternative.minimum
            )
            for other, alternative in enumerate(limits)
        ):
            return unique[index]
    choice = TimeChoice(unique, maximum)
    axes = expression_axes(choice)
    if len(axes) == 1 and all(
        variable.bounds.source in axes for variable in time_variables(choice)
    ):
        points = evaluate_time(choice, axes)
        if points is not None:
            return point_time("loop extremum", axes, points, reconstruction=choice)
        grids = [interval_arrays(value, axes) for value in unique]
        select = np.maximum if maximum else np.minimum
        lower, upper = grids[0].minimum, grids[0].maximum
        for grid in grids[1:]:
            lower, upper = select(lower, grid.minimum), select(upper, grid.maximum)
        return interval_time("loop extremum", axes, lower, upper)
    return choice


def substitute_time(
    value: TimeExpression, replacements: dict[CycleVariable, TimeExpression]
) -> TimeExpression:
    """Replace producer identities in cycle algebra without changing runtime storage."""
    if isinstance(value, TimeChoice):
        return choose_time(
            *(substitute_time(item, replacements) for item in value.alternatives),
            maximum=value.maximum,
        )
    result: TimeExpression = LinearTime(value.constant)
    for variable, coefficient in value.terms:
        result = add_time(
            result,
            scale_time(
                replacements.get(variable, LinearTime(terms=((variable, 1),))),
                coefficient,
            ),
        )
    return result


def scale_time(value: TimeExpression, coefficient: int) -> TimeExpression:
    """Multiply cycles by an integer without rounding runtime coefficients."""
    if isinstance(value, TimeChoice):
        return choose_time(
            *(scale_time(item, coefficient) for item in value.alternatives),
            maximum=value.maximum if coefficient >= 0 else not value.maximum,
        )
    return LinearTime(
        value.constant * coefficient,
        tuple(
            (variable, amount * coefficient)
            for variable, amount in value.terms
            if amount * coefficient
        ),
    )


def add_time(left: TimeExpression, right: TimeExpression) -> TimeExpression:
    """Add cycles, merging shared values and distributing over pointwise extrema."""
    if isinstance(left, TimeChoice):
        return choose_time(
            *(add_time(item, right) for item in left.alternatives),
            maximum=left.maximum,
        )
    if isinstance(right, TimeChoice):
        return add_time(right, left)
    coefficients = dict(left.terms)
    for variable, coefficient in right.terms:
        coefficients[variable] = coefficients.get(variable, 0) + coefficient
    return LinearTime(
        left.constant + right.constant,
        tuple(
            (variable, coefficient)
            for variable, coefficient in coefficients.items()
            if coefficient
        ),
    )
