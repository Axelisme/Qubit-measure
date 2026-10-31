"""Compile-time cycle algebra, independent of register allocation and emission."""

import logging
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from ._timeline_bounds import CycleBounds
from ._timeline_range import VariableRange, source_limits

CycleValues = NDArray[np.int64] | NDArray[np.object_]


@dataclass(frozen=True)
class CycleIntervalGrid:
    """Coordinate-aware inclusive bounds in integer tProc cycles.

    minimum and maximum are int64 arrays broadcastable to the caller's ordered
    loop-axis grid. At every coordinate minimum <= maximum. These endpoints
    bound unknown inputs, not necessarily attainable values. One result keeps
    the paired grids and their units together for all expression consumers.
    Construction stores arrays without copying or changing them.
    """

    minimum: CycleValues
    maximum: CycleValues


@dataclass(frozen=True)
class CycleProducer:
    """Immutable runtime recipe, separate from the range used by proofs.

    axes/points retain exact integer values in row-major loop order for Repeat
    reduction and exact-value getters. One axis without points is its loop index.
    reconstruction is integer algebra for an affine lookup; selection is the
    captured selector and declared words. fixed_point holds numerator and shift.
    rounding_remainder marks a cross-loop shift's integer remainder, whose
    fixed_point recipe reconstructs the whole quotient before rational algebra.
    fallback_branches records a >64-branch proof fallback, otherwise zero.
    None means that recipe is unavailable, never that a range is unbounded.
    """

    axes: tuple[tuple[str, int], ...] = ()
    points: tuple[int, ...] | None = None
    reconstruction: "TimeExpression | None" = None
    selection: "tuple[TimeExpression, tuple[int, ...]] | None" = None
    fixed_point: "tuple[TimeExpression, int] | None" = None
    rounding_remainder: bool = False
    fallback_branches: int = 0


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
        lower, _ = expression_limits(LinearTime(terms=((self, 1),)))
        return None if lower is None else math.ceil(lower)

    @property
    def maximum(self) -> int | None:
        """Return the conservative integer upper bound, or None when unbounded."""
        _, upper = expression_limits(LinearTime(terms=((self, 1),)))
        return None if upper is None else math.floor(upper)


def cycle_variable(
    name: str,
    minimum: int | None,
    maximum: int | None,
    producer: CycleProducer | None = None,
    *,
    interval_points: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> CycleVariable:
    """Capture a producer's range once, without evaluating expression grids.

    A known one-loop sequence shares that loop's ID. Its directed endpoints and
    exact deviations are computed only along that source. Unknown values and
    multi-loop nonlinear results receive independent IDs, even with bounds.
    Recipe metadata survives independently for runtime reconstruction.
    """
    producer = CycleProducer() if producer is None else producer
    axes, points = producer.axes, producer.points
    source: str | object = object()
    start, end = minimum, maximum
    low = high = (Fraction(0),)
    if len(axes) == 1 and not producer.rounding_remainder:
        source, count = axes[0]
        low = high = (Fraction(0),) * count
        if interval_points is not None or points is not None:
            lower, upper = (
                interval_points if interval_points is not None else (points, points)
            )
            if lower is None or upper is None:
                raise RuntimeError("point producer has no values")
            start, end = lower[0], lower[-1]
            line = tuple(
                Fraction(start) + Fraction((end - start) * i, max(1, count - 1))
                for i in range(count)
            )
            low = tuple(
                Fraction(value) - origin
                for value, origin in zip(lower, line, strict=True)
            )
            high = tuple(
                Fraction(value) - origin
                for value, origin in zip(upper, line, strict=True)
            )
    return CycleVariable(
        name,
        VariableRange(source, start, end, low, high),
        producer,
    )


@dataclass(frozen=True)
class LinearTime:
    """Integer cycles as constant + sum(coefficient * variable).

    constant is a signed cycle offset. terms contains distinct runtime values
    and nonzero coefficients. Rational coefficients occur only in cross-loop
    ceil/shift proofs; complete scheduling values remain integer cycles.
    Algebra merges coefficients before bounding, so shared origins cancel even
    when register storage differs.
    """

    constant: int | Fraction = 0
    terms: tuple[tuple[CycleVariable, int | Fraction], ...] = ()


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


def expression_limits(value: TimeExpression) -> tuple[Fraction | None, Fraction | None]:
    """Bound cycles by grouping source IDs, with None for unavailable endpoints.

    Each loop group sums its coefficient-weighted directed line and deviations
    along that axis only. Different source groups add their bounds. Max/min
    paths preserve whatever one-sided bound their alternatives establish.
    """
    if isinstance(value, TimeChoice):
        variables = time_variables(value)
        sources = {v.bounds.source for v in variables}
        if len(sources) == 1 and all(isinstance(source, str) for source in sources):
            axis = next(iter(sources))
            if isinstance(axis, str):
                count = len(variables[0].bounds.deviation_minimum)
                grids = interval_arrays(value, {axis: count})
                return Fraction(grids.minimum.min()), Fraction(grids.maximum.max())
        # Bound unknown inputs before the nonlinear range capture. This keeps
        # shared loop coordinates when an independent feedback value survives.
        # Each resulting linear path still uses source_limits, never a grid.
        variables = time_variables(value)
        if (
            variables
            and all(v.minimum is not None and v.maximum is not None for v in variables)
            and any(
                not isinstance(v.bounds.source, str)
                or v.bounds.deviation_minimum != v.bounds.deviation_maximum
                for v in variables
            )
        ):
            lower_choice = choose_time(
                *(bound_unknowns(item, upper=False) for item in value.alternatives),
                maximum=value.maximum,
            )
            upper_choice = choose_time(
                *(bound_unknowns(item, upper=True) for item in value.alternatives),
                maximum=value.maximum,
            )
            return expression_limits(lower_choice)[0], expression_limits(upper_choice)[
                1
            ]
        limits = [expression_limits(item) for item in value.alternatives]
        select = max if value.maximum else min
        lows = [lo for lo, _ in limits if lo is not None]
        highs = [hi for _, hi in limits if hi is not None]
        return (
            select(lows)
            if lows and (value.maximum or len(lows) == len(limits))
            else None,
            select(highs)
            if highs and (not value.maximum or len(highs) == len(limits))
            else None,
        )
    groups: dict[str | object, list[tuple[VariableRange, int | Fraction]]] = {}
    for variable, coefficient in value.terms:
        groups.setdefault(variable.bounds.source, []).append(
            (variable.bounds, coefficient)
        )
    lower: Fraction | None = Fraction(value.constant)
    upper: Fraction | None = Fraction(value.constant)
    for terms in groups.values():
        lo, hi = source_limits(terms)
        lower = None if lower is None or lo is None else lower + lo
        upper = None if upper is None or hi is None else upper + hi
    return lower, upper


def expression_bounds(value: TimeExpression) -> CycleBounds:
    """Return finite bounds by the common source rule; unbounded raises ValueError."""
    lower, upper = expression_limits(value)
    if lower is None or upper is None:
        raise ValueError("runtime-only time has no compile-time bounds")
    return CycleBounds(math.ceil(lower), math.floor(upper))


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
    return dict(
        axis for variable in time_variables(value) for axis in variable.producer.axes
    )


def evaluate_time(value: TimeExpression, axes: dict[str, int]) -> CycleValues | None:
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
    result = np.asarray(
        value.constant,
        dtype=object if isinstance(value.constant, Fraction) else np.int64,
    )
    for variable, coefficient in value.terms:
        if variable.minimum is None or variable.maximum is None:
            return None
        if variable.minimum == variable.maximum:
            result = np.add(
                result, np.asarray(coefficient * variable.minimum, dtype=object)
            )
            continue
        if (
            not variable.producer.axes
            or variable.bounds.deviation_minimum != variable.bounds.deviation_maximum
            or (
                variable.producer.points is None
                and not variable.producer.rounding_remainder
                and (
                    len(variable.producer.axes) != 1
                    or not isinstance(variable.bounds.source, str)
                )
            )
        ):
            return None
        local = dict(variable.producer.axes)
        if variable.producer.rounding_remainder:
            if variable.producer.fixed_point is None:
                raise RuntimeError("rounding remainder lost its numerator")
            numerator, shift = variable.producer.fixed_point
            raw = evaluate_time(numerator, local)
            if raw is None:
                return None
            points = raw % (1 << shift)
        else:
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
        result = np.add(
            result,
            np.multiply(np.asarray(coefficient, dtype=object), points.reshape(shape)),
        )
    return result


def ceiled_time(value: LinearTime) -> TimeExpression:
    """Bound a cross-loop affine ceil by its line and an independent rounding error.

    Coefficients are exact fractional cycles. Integer lines need no error.
    The lower tolerance preserves ceil_tick's 1e-10-cycle integer snapping.
    The error is compile-time proof metadata, never a runtime time table.
    """
    denominator = math.lcm(
        Fraction(value.constant).denominator,
        *(Fraction(coefficient).denominator for _, coefficient in value.terms),
    )
    if denominator == 1:
        return value
    error = cycle_variable(
        "ceil rounding", -math.ceil(Fraction(denominator, 10**10)), denominator - 1
    )
    return add_time(value, LinearTime(terms=((error, Fraction(1, denominator)),)))


def shifted_time(value: TimeExpression, shift: int) -> TimeExpression:
    """Describe arithmetic right shift, keeping cross-loop linear contributions.

    Single-axis results retain exact deviations. Across axes, an independent
    integer remainder in 0..2**shift-1 accounts for fractional rounding. Its
    recipe retains the original numerator, without constructing a Cartesian grid.
    """
    axes = expression_axes(value)
    if len(axes) <= 1:
        points = evaluate_time(value, axes)
        if points is not None:
            values = points >> shift
            variable = cycle_variable(
                "shifted cycles",
                int(values.min()),
                int(values.max()),
                CycleProducer(
                    tuple(axes.items()),
                    tuple(int(v) for v in values.flat),
                    fixed_point=(value, shift),
                ),
            )
            return LinearTime(terms=((variable, 1),))
    divisor = 1 << shift
    remainder = cycle_variable(
        "rounding remainder",
        0,
        divisor - 1,
        CycleProducer(
            tuple(axes.items()), fixed_point=(value, shift), rounding_remainder=True
        ),
    )
    return add_time(
        scale_time(value, Fraction(1, divisor)),
        LinearTime(terms=((remainder, Fraction(-1, divisor)),)),
    )


def exact_bounds(value: TimeExpression) -> CycleBounds | None:
    """Return the common range only when its extrema are attainable.

    Unknown-input envelopes cannot establish actual slack. A multi-loop
    nonlinear source is independent in this representation; if another term
    reuses its axes, no exact-extrema claim is made. No linear-expression grid
    is evaluated to obtain these bounds.
    """
    if unbounded_time(value):
        return None
    variables = time_variables(value)
    limits = expression_bounds(value)
    if limits.minimum == limits.maximum:
        return limits
    for variable in variables:
        recipe = variable.producer
        if (
            variable.bounds.deviation_minimum != variable.bounds.deviation_maximum
            or not recipe.axes
            or recipe.rounding_remainder
            or (len(recipe.axes) > 1 and recipe.points is None)
        ):
            return None
        if not isinstance(variable.bounds.source, str):
            axes = dict(recipe.axes).keys()
            if any(
                other is not variable and axes & dict(other.producer.axes).keys()
                for other in variables
            ):
                return None
    return limits


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
        r = variable.bounds
        if isinstance(r.source, str) and r.deviation_minimum != r.deviation_maximum:
            deviations = (
                r.deviation_maximum
                if upper == (coefficient > 0)
                else r.deviation_minimum
            )
            if r.start is None or r.end is None:
                raise ValueError("runtime-only time has no compile-time bounds")
            count = len(deviations)
            points = tuple(
                int(
                    Fraction(r.start)
                    + Fraction((r.end - r.start) * i, max(1, count - 1))
                    + deviation
                )
                for i, deviation in enumerate(deviations)
            )
            endpoint = cycle_variable(
                variable.name,
                min(points),
                max(points),
                CycleProducer(variable.producer.axes, points),
            )
            terms.append((endpoint, coefficient))
        elif isinstance(variable.bounds.source, str):
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
    """Name surviving nonconstant cycle sources without constructing a point grid."""
    return sorted(
        {
            variable.name
            for variable in time_variables(value)
            if variable.minimum != variable.maximum
        }
    )


def point_time(
    name: str,
    axes: dict[str, int],
    points: NDArray[np.int64],
    *,
    reconstruction: TimeExpression | None = None,
) -> TimeExpression:
    """Retain a known nonlinear grid as a bounded variable, or a scalar constant."""
    lower, upper = int(points.min()), int(points.max())
    if lower == upper:
        return LinearTime(lower)
    variable = cycle_variable(
        name,
        lower,
        upper,
        CycleProducer(
            tuple(axes.items()),
            tuple(int(v) for v in points.flat),
            reconstruction=reconstruction,
        ),
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

    def reduce(points: CycleValues) -> NDArray[np.int64]:
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
        CycleProducer(tuple(axes.items())),
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
    if len(unique) > 64:
        choice = TimeChoice(unique, maximum)
        limits = [expression_limits(value) for value in unique]
        select = max if maximum else min
        lower = (
            None
            if any(lo is None for lo, _ in limits)
            else select(lo for lo, _ in limits if lo is not None)
        )
        upper = (
            None
            if any(hi is None for _, hi in limits)
            else select(hi for _, hi in limits if hi is not None)
        )
        logging.getLogger(__name__).warning(
            "Timeline range fallback: %s branches exceed 64", len(unique)
        )
        variable = cycle_variable(
            "choice fallback",
            None if lower is None else math.floor(lower),
            None if upper is None else math.ceil(upper),
            CycleProducer(reconstruction=choice, fallback_branches=len(unique)),
        )
        return LinearTime(terms=((variable, 1),))
    if any(unbounded_time(value) for value in unique):
        return TimeChoice(unique, maximum)
    limits = [expression_limits(value) for value in unique]
    for index, (lower, upper) in enumerate(limits):
        if all(
            index == other
            or (
                lower is not None
                and alternative[1] is not None
                and lower >= alternative[1]
                if maximum
                else upper is not None
                and alternative[0] is not None
                and upper <= alternative[0]
            )
            for other, alternative in enumerate(limits)
        ):
            return unique[index]
    return TimeChoice(unique, maximum)


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


def scale_time(value: TimeExpression, coefficient: int | Fraction) -> TimeExpression:
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
