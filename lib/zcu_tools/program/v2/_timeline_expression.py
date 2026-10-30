"""Compile-time cycle algebra, independent of register allocation and emission."""

from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class CycleVariable:
    """One integer-cycle runtime value, identified by object identity.

    name identifies the producer for diagnostics, not equality. minimum and
    maximum are inclusive signed-cycle bounds. Separate writes to a recycled
    register must use separate variables.
    """

    name: str
    minimum: int
    maximum: int


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


def expression_bounds(value: TimeExpression) -> tuple[int, int]:
    """Bound cycles after cancellation, without assuming independent max paths."""
    if isinstance(value, TimeChoice):
        limits = [expression_bounds(item) for item in value.alternatives]
        select = max if value.maximum else min
        return select(lo for lo, _ in limits), select(hi for _, hi in limits)
    lower = upper = value.constant
    for variable, coefficient in value.terms:
        endpoints = coefficient * variable.minimum, coefficient * variable.maximum
        lower += min(endpoints)
        upper += max(endpoints)
    return lower, upper


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
    limits = [expression_bounds(value) for value in unique]
    for index, (lower, upper) in enumerate(limits):
        if all(
            index == other or (lower >= hi if maximum else upper <= lo)
            for other, (lo, hi) in enumerate(limits)
        ):
            return unique[index]
    return TimeChoice(unique, maximum)


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
