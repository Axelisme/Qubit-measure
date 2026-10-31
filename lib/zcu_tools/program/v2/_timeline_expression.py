"""Axis-component cycle algebra. Proof envelopes never supply runtime values."""

import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property, lru_cache

import numpy as np
from numpy.typing import NDArray

from ._timeline_bounds import CycleBounds

CycleValues = NDArray[np.int64] | NDArray[np.object_]
Scalar = int | Fraction
MAX_ABS_TERMS = 20


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
AxisDomains = dict[object, NDArray[np.int64]]


def _component_scale(component: Component, coefficient: Scalar) -> Component:
    if isinstance(component, np.ndarray):
        return np.asarray(component, dtype=object) * np.asarray(
            coefficient, dtype=object
        )
    return component * coefficient


def _component_add(left: Component, right: Component) -> Component:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.asarray(left, dtype=object) + np.asarray(right, dtype=object)
    return left + right


def _nonzero(component: Component) -> bool:
    return bool(np.any(np.asarray(component, dtype=object)))


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
                    coefficient
                    if axis.values is None
                    else _component_scale(axis.values, coefficient)
                )
                components = _merge_components(components, ((axis, component),))
        frozen = []
        for axis, component in components:
            if isinstance(component, np.ndarray):
                component = component.copy()
                component.flags.writeable = False
            frozen.append((axis, component))
        components = tuple(frozen)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "components", components)

    @cached_property
    def limits(self) -> tuple[Fraction | None, Fraction | None]:
        """Return this immutable variable's cached independent-axis bounds."""
        lower: Fraction | None = Fraction(0)
        upper: Fraction | None = Fraction(0)
        for axis, component in self.components:
            lo, hi = _component_limits(axis, component)
            lower = None if lower is None or lo is None else lower + lo
            upper = None if upper is None or hi is None else upper + hi
        return lower, upper

    def __hash__(self) -> int:
        return hash(frozenset(_component_key(axis, c) for axis, c in self.components))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TimeVariable) and frozenset(
            _component_key(axis, c) for axis, c in self.components
        ) == frozenset(_component_key(axis, c) for axis, c in other.components)


@dataclass(frozen=True)
class TimeOperation:
    """Exact integer runtime recipe, never inferred from Abs proof terms.

    kind selects max/min/add/scale. operands retain the original immutable
    values; coefficient is used only by scale.
    """

    kind: str
    operands: tuple["TimeExpression", ...]
    coefficient: Scalar = 1


@dataclass(frozen=True)
class TimeAbs:
    """Linear axis components plus rational multiples of nested absolute values.

    linear is the additive variable. absolute contains normalized arguments and
    nonzero coefficients, including negative coefficients. runtime preserves the
    integer operation independently of proof normalization and range fallback.
    """

    linear: TimeVariable
    absolute: tuple[tuple["TimeExpression", Fraction], ...]
    runtime: TimeOperation = field(compare=False, hash=False)

    @cached_property
    def limits(self) -> tuple[Fraction | None, Fraction | None]:
        """Return this immutable node's cached conservative cycle range."""
        return _abs_limits(self)


TimeExpression = TimeVariable | TimeAbs


def _merge_components(
    left: tuple[tuple[CycleAxis, Component], ...],
    right: tuple[tuple[CycleAxis, Component], ...],
) -> tuple[tuple[CycleAxis, Component], ...]:
    groups = {axis.identity: (axis, component) for axis, component in left}
    for axis, component in right:
        previous = groups.get(axis.identity)
        if previous is not None:
            component = _component_add(previous[1], component)
        groups[axis.identity] = (axis, component)
    return tuple((axis, c) for axis, c in groups.values() if _nonzero(c))


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
    """Return cached bounds without reading mutable runtime storage."""
    return value.limits


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
    """Return surviving source identities for runtime capture and diagnostics."""
    if isinstance(value, TimeVariable):
        return tuple(axis for axis, _ in value.terms)
    return tuple(
        dict.fromkeys(
            axis
            for item in (value.linear, *(x for x, _ in value.absolute))
            for axis in time_variables(item)
        )
    )


def expression_axes(value: TimeExpression) -> dict[str, int]:
    """Return known dimensions from proof components, not producer recipes."""
    if isinstance(value, TimeAbs):
        return {
            name: count
            for item in (value.linear, *(x for x, _ in value.absolute))
            for name, count in expression_axes(item).items()
        }
    return {
        axis.identity: len(c)
        for axis, c in value.components
        if isinstance(axis.identity, str)
        and axis.identity != "const"
        and isinstance(c, np.ndarray)
    }


def evaluate_time(value: TimeExpression, axes: dict[str, int]) -> CycleValues | None:
    """Evaluate actual known values for getters; never evaluate proof envelopes.

    axes controls ndarray broadcasting order. Scalars return a zero-dimensional
    ndarray. Unknown independent sources return None.
    """
    if isinstance(value, TimeAbs):
        result = evaluate_time(value.linear, axes)
        if result is None:
            return None
        for argument, coefficient in value.absolute:
            points = evaluate_time(argument, axes)
            if points is None:
                return None
            result = result + coefficient * np.abs(points)
        return np.asarray(result, dtype=object)
    result = np.asarray(0, dtype=object)
    for axis, component in value.components:
        if not isinstance(component, np.ndarray):
            if axis.minimum is None or axis.minimum != axis.maximum:
                return None
            result = result + np.asarray(component * axis.minimum, dtype=object)
        elif axis.identity == "const":
            result = result + component[0]
        else:
            shape = tuple(
                len(component) if name == axis.identity else 1 for name in axes
            )
            result = result + component.reshape(shape)
    return np.asarray(result, dtype=object)


def ceiled_time(value: TimeExpression) -> TimeExpression:
    """Ceil single-axis components exactly; cross-axis branches get fresh errors."""
    if isinstance(value, TimeAbs):
        # Runtime integer choices are already integral. Rational clock conversion
        # only enters the linear producer paths below.
        return value
    axes = expression_axes(value)
    if len(axes) <= 1:
        points = evaluate_time(value, axes)
        if points is not None:
            rounded = np.asarray(
                [
                    _ceil_fraction(Fraction(v))
                    for v in np.asarray(points, dtype=object).flat
                ],
                dtype=np.int64,
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
    axes = expression_axes(value)
    if len(axes) <= 1:
        points = evaluate_time(value, axes)
        if points is not None:
            values = np.asarray(
                [int(v) >> shift for v in np.asarray(points, dtype=object).flat],
                dtype=np.int64,
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
    if any(axis.values is None for axis in _proof_axes(value).values()):
        return None
    if isinstance(value, TimeAbs) and any(
        len(_abs_arguments(block)) > MAX_ABS_TERMS for block in _connected_blocks(value)
    ):
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
    if isinstance(value, TimeAbs):
        # A nonlinear reduction cannot be distributed through absolute values.
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
    if isinstance(right, TimeVariable) and not _proof_axes(right):
        return scale_time(left, expression_limits(right)[0] or 0)
    if isinstance(left, TimeVariable) and not _proof_axes(left):
        return scale_time(right, expression_limits(left)[0] or 0)
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
            names = ", ".join(sorted({axis.name for axis in time_variables(selector)}))
            return point_time(names or "selected cycles", axes, result)
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


def _proof_axes(value: TimeExpression) -> dict[object, CycleAxis]:
    linear, absolute = _parts(value)
    axes = {
        axis.identity: axis for axis, _ in linear.components if axis.identity != "const"
    }
    for argument, _ in absolute:
        axes.update(_proof_axes(argument))
    return axes


def _connected_blocks(value: TimeAbs) -> list[TimeAbs]:
    groups: list[tuple[set[object], list[tuple[TimeExpression, Fraction]]]] = []
    for argument, coefficient in value.absolute:
        axes = set(_proof_axes(argument))
        terms = [(argument, coefficient)]
        separate = []
        for previous, items in groups:
            if axes & previous:
                axes |= previous
                terms.extend(items)
            else:
                separate.append((previous, items))
        groups = [*separate, (axes, terms)]
    blocks = []
    used: set[object] = set()
    for axes, terms in groups:
        components = tuple(
            (a, c) for a, c in value.linear.components if a.identity in axes
        )
        used |= axes
        blocks.append(
            TimeAbs(TimeVariable(components=components), tuple(terms), value.runtime)
        )
    remaining = tuple(
        (a, c) for a, c in value.linear.components if a.identity not in used
    )
    if remaining:
        blocks.append(TimeAbs(TimeVariable(components=remaining), (), value.runtime))
    return blocks


def _interval_limits(value: TimeExpression) -> tuple[Fraction | None, Fraction | None]:
    linear, absolute = _parts(value)
    lower, upper = linear.limits
    for argument, coefficient in absolute:
        lo, hi = expression_limits(argument)
        abs_lo = (
            Fraction(0)
            if lo is None or hi is None or lo <= 0 <= hi
            else min(abs(lo), abs(hi))
        )
        abs_hi = None if lo is None or hi is None else max(abs(lo), abs(hi))
        a, b = (abs_lo, abs_hi) if coefficient > 0 else (abs_hi, abs_lo)
        lower = None if lower is None or a is None else lower + coefficient * a
        upper = None if upper is None or b is None else upper + coefficient * b
    return lower, upper


def _abs_arguments(value: TimeExpression) -> list[TimeExpression]:
    arguments: list[TimeExpression] = []
    if isinstance(value, TimeAbs):
        for argument, _ in value.absolute:
            arguments.extend(_abs_arguments(argument))
            arguments.append(argument)
    return list(dict.fromkeys(arguments))


def _signed_linear(
    value: TimeExpression, signs: dict[TimeExpression, int]
) -> TimeVariable:
    linear, absolute = _parts(value)
    for argument, coefficient in absolute:
        linear = _linear_add(
            linear,
            _linear_scale(
                _signed_linear(argument, signs), coefficient * signs[argument]
            ),
        )
    return linear


def _domain_limits(
    value: TimeVariable, domains: AxisDomains
) -> tuple[Fraction | None, Fraction | None]:
    components = tuple(
        (
            axis,
            component[domains[axis.identity]]
            if isinstance(component, np.ndarray) and axis.identity in domains
            else component,
        )
        for axis, component in value.components
    )
    return TimeVariable(components=components).limits


def _constrain_domains(
    constraints: tuple[TimeVariable, ...], domains: AxisDomains
) -> AxisDomains | None:
    domains = domains.copy()
    changed = True
    while changed:
        changed = False
        for constraint in constraints:
            _, upper = _domain_limits(constraint, domains)
            if upper is not None and upper < 0:
                return None
            for axis, component in constraint.components:
                if (
                    not isinstance(component, np.ndarray)
                    or axis.identity not in domains
                ):
                    continue
                rest = TimeVariable(
                    components=tuple(
                        (a, c)
                        for a, c in constraint.components
                        if a.identity != axis.identity
                    )
                )
                _, rest_upper = _domain_limits(rest, domains)
                if rest_upper is None:
                    continue
                previous = domains[axis.identity]
                kept = previous[
                    np.asarray(component[previous] + rest_upper >= 0, dtype=bool)
                ]
                if not len(kept):
                    return None
                if len(kept) != len(previous):
                    domains[axis.identity] = kept
                    changed = True
    return domains


def _endpoint_witness(
    value: TimeVariable, domains: AxisDomains, *, upper: bool
) -> dict[object, int]:
    witness = {axis: int(indices[0]) for axis, indices in domains.items()}
    for axis, component in value.components:
        if isinstance(component, np.ndarray) and axis.identity in domains:
            indices = domains[axis.identity]
            select = np.argmax if upper else np.argmin
            witness[axis.identity] = int(indices[int(select(component[indices]))])
    return witness


def _witness_satisfies(constraint: TimeVariable, witness: dict[object, int]) -> bool:
    result = Fraction(0)
    for axis, component in constraint.components:
        if isinstance(component, np.ndarray):
            result += Fraction(component[witness.get(axis.identity, 0)])
        else:
            return False
    return result >= 0


def _linear_extreme(
    value: TimeVariable,
    constraints: tuple[TimeVariable, ...],
    domains: AxisDomains,
    *,
    upper: bool,
) -> Fraction | None:
    best: Fraction | None = None
    pending = [domains]
    while pending:
        current = _constrain_domains(constraints, pending.pop())
        if current is None:
            continue
        lo, hi = _domain_limits(value, current)
        estimate = hi if upper else lo
        if estimate is None:
            return None
        if best is not None and (estimate <= best if upper else estimate >= best):
            continue
        witness = _endpoint_witness(value, current, upper=upper)
        violated = next(
            (c for c in constraints if not _witness_satisfies(c, witness)), None
        )
        if violated is None:
            best = estimate
            continue
        split = next(
            (
                a.identity
                for a, _ in violated.components
                if a.identity in current and len(current[a.identity]) > 1
            ),
            None,
        )
        if split is None:
            continue
        indices = current[split]
        middle = len(indices) // 2
        pending.extend(
            ({**current, split: indices[:middle]}, {**current, split: indices[middle:]})
        )
    return best


def _constrained_limits(
    value: TimeVariable, constraints: tuple[TimeVariable, ...], domains: AxisDomains
) -> tuple[Fraction | None, Fraction | None]:
    lower, upper = _domain_limits(value, domains)
    components = {axis.identity: c for axis, c in value.components}
    for constraint in constraints:
        for axis, component in constraint.components:
            if (
                axis.identity == "const"
                or isinstance(component, np.ndarray)
                or not component
            ):
                continue
            amount = components.get(axis.identity, 0)
            if isinstance(amount, np.ndarray):
                continue
            ratio = Fraction(amount) / component
            rest = _linear_add(value, _linear_scale(constraint, -ratio))
            lo, hi = _domain_limits(rest, domains)
            if ratio > 0 and lo is not None:
                lower = lo if lower is None else max(lower, lo)
            if ratio < 0 and hi is not None:
                upper = hi if upper is None else min(upper, hi)
    return lower, upper


def _block_limits(value: TimeAbs) -> tuple[Fraction | None, Fraction | None]:
    arguments = _abs_arguments(value)
    if len(arguments) > MAX_ABS_TERMS:
        return _interval_limits(value)
    axes = _proof_axes(value)
    known = all(axis.values is not None for axis in axes.values())
    domains: AxisDomains = {
        identity: np.arange(len(axis.values), dtype=np.int64)
        for identity, axis in axes.items()
        if axis.values is not None
    }
    leaves: list[tuple[Fraction | None, Fraction | None]] = []
    pending: list[
        tuple[dict[TimeExpression, int], tuple[TimeVariable, ...], AxisDomains]
    ] = [({}, (), domains)]
    while pending:
        signs, constraints, current = pending.pop()
        current = _constrain_domains(constraints, current)
        if current is None:
            continue
        if len(signs) < len(arguments):
            argument = arguments[len(signs)]
            linear = _signed_linear(argument, signs)
            lo, hi = _domain_limits(linear, current)
            choices = (
                (1,)
                if lo is not None and lo >= 0
                else (-1,)
                if hi is not None and hi <= 0
                else (1, -1)
            )
            for sign in choices:
                pending.append(
                    (
                        {**signs, argument: sign},
                        (*constraints, _linear_scale(linear, sign)),
                        current,
                    )
                )
            continue
        linear = _signed_linear(value, signs)
        if known:
            bounds = (
                _linear_extreme(linear, constraints, current, upper=False),
                _linear_extreme(linear, constraints, current, upper=True),
            )
            if bounds == (None, None):
                continue
        else:
            bounds = _constrained_limits(linear, constraints, current)
        leaves.append(bounds)
    if not leaves:
        return _interval_limits(value)
    return (
        None
        if any(lo is None for lo, _ in leaves)
        else min(lo for lo, _ in leaves if lo is not None),
        None
        if any(hi is None for _, hi in leaves)
        else max(hi for _, hi in leaves if hi is not None),
    )


def _abs_limits(value: TimeAbs) -> tuple[Fraction | None, Fraction | None]:
    axes = expression_axes(value)
    if len(axes) <= 1:
        points = evaluate_time(value, axes)
        if points is not None:
            return Fraction(points.min()), Fraction(points.max())
    limits = [_block_limits(block) for block in _connected_blocks(value)]
    return (
        None
        if any(lo is None for lo, _ in limits)
        else sum((lo for lo, _ in limits if lo is not None), Fraction(0)),
        None
        if any(hi is None for _, hi in limits)
        else sum((hi for _, hi in limits if hi is not None), Fraction(0)),
    )


def _linear_add(left: TimeVariable, right: TimeVariable) -> TimeVariable:
    coefficients = dict(left.terms)
    for axis, coefficient in right.terms:
        coefficients[axis] = coefficients.get(axis, 0) + coefficient
    return TimeVariable(
        left.constant + right.constant,
        tuple((v, c) for v, c in coefficients.items() if c),
        components=_merge_components(left.components, right.components),
    )


def _linear_scale(value: TimeVariable, coefficient: Scalar) -> TimeVariable:
    return TimeVariable(
        value.constant * coefficient,
        tuple((v, c * coefficient) for v, c in value.terms if c * coefficient),
        components=tuple(
            (axis, _component_scale(c, coefficient))
            for axis, c in value.components
            if _nonzero(_component_scale(c, coefficient))
        ),
    )


def _parts(
    value: TimeExpression,
) -> tuple[TimeVariable, tuple[tuple[TimeExpression, Fraction], ...]]:
    return (
        (value, ())
        if isinstance(value, TimeVariable)
        else (value.linear, value.absolute)
    )


def _orientation(value: TimeExpression) -> int:
    linear, absolute = _parts(value)
    for _, component in sorted(
        linear.components, key=lambda item: repr(item[0].identity)
    ):
        items = np.asarray(component, dtype=object).flat
        for amount in items:
            if amount:
                return 1 if amount > 0 else -1
    for _, coefficient in absolute:
        if coefficient:
            return 1 if coefficient > 0 else -1
    return 1


def _normalized(
    linear: TimeVariable,
    absolute: tuple[tuple[TimeExpression, Fraction], ...],
    runtime: TimeOperation,
) -> TimeExpression:
    combined: dict[TimeExpression, Fraction] = {}
    for argument, coefficient in absolute:
        if not coefficient:
            continue
        lo, hi = expression_limits(argument)
        if lo is not None and lo >= 0:
            replacement = argument
        elif hi is not None and hi <= 0:
            replacement = scale_time(argument, -1)
        else:
            if _orientation(argument) < 0:
                argument = scale_time(argument, -1)
            combined[argument] = combined.get(argument, Fraction(0)) + coefficient
            continue
        variable, terms = _parts(replacement)
        linear = _linear_add(linear, _linear_scale(variable, coefficient))
        for term, amount in terms:
            combined[term] = combined.get(term, Fraction(0)) + coefficient * amount
    terms = tuple(
        sorted(
            ((x, c) for x, c in combined.items() if c), key=lambda item: repr(item[0])
        )
    )
    return TimeAbs(linear, terms, runtime) if terms else linear


def scale_time(value: TimeExpression, coefficient: Scalar) -> TimeExpression:
    """Multiply every linear and Abs coefficient, retaining negative rationals."""
    if coefficient == 0:
        return TimeVariable()
    if coefficient == 1:
        return value
    linear, absolute = _parts(value)
    return _normalized(
        _linear_scale(linear, coefficient),
        tuple((x, c * coefficient) for x, c in absolute),
        TimeOperation("scale", (value,), coefficient),
    )


def add_time(left: TimeExpression, right: TimeExpression) -> TimeExpression:
    """Align linear axes and concatenate Abs terms without branch products."""
    a, x = _parts(left)
    b, y = _parts(right)
    return _normalized(_linear_add(a, b), (*x, *y), TimeOperation("add", (left, right)))


@lru_cache(maxsize=4096)
def difference_limits(
    left: TimeExpression, right: TimeExpression
) -> tuple[Fraction | None, Fraction | None]:
    """Cache normalized structural differences for dominance, not runtime values."""
    return expression_limits(add_time(left, scale_time(right, -1)))


def _binary_choice(
    left: TimeExpression, right: TimeExpression, *, maximum: bool
) -> TimeExpression:
    lo, hi = difference_limits(left, right)
    if lo is not None and lo >= 0:
        return left if maximum else right
    if hi is not None and hi <= 0:
        return right if maximum else left
    total = add_time(
        scale_time(left, Fraction(1, 2)), scale_time(right, Fraction(1, 2))
    )
    linear, terms = _parts(total)
    difference = add_time(left, scale_time(right, -1))
    recipe = TimeOperation("max" if maximum else "min", (left, right))
    return _normalized(
        linear, (*terms, (difference, Fraction(1 if maximum else -1, 2))), recipe
    )


def choose_time(*values: TimeExpression, maximum: bool = True) -> TimeExpression:
    """Return exact max/min proof algebra; runtime keeps original integer inputs."""
    if not values:
        raise ValueError("time choice requires at least one value")
    unique = list(dict.fromkeys(values))
    # Reduce same-axis subsets before introducing independent dimensions.
    groups: dict[str, list[TimeExpression]] = {}
    for value in unique:
        axes = expression_axes(value)
        if len(axes) == 1 and evaluate_time(value, axes) is not None:
            groups.setdefault(next(iter(axes)), []).append(value)
    for group in groups.values():
        if len(group) < 2:
            continue
        axes = expression_axes(group[0])
        points = [evaluate_time(value, axes) for value in group]
        known = [point for point in points if point is not None]
        merged = (np.maximum if maximum else np.minimum).reduce(
            np.broadcast_arrays(*known)
        )
        proof = group[0]
        for value in group[1:]:
            proof = _binary_choice(proof, value, maximum=maximum)
        names = ", ".join(
            sorted({axis.name for value in group for axis in time_variables(value)})
        )
        replacement = point_time(names, axes, merged, reconstruction=proof)
        unique = [value for value in unique if value not in group]
        unique.append(replacement)
    result = unique[0]
    for value in unique[1:]:
        result = _binary_choice(result, value, maximum=maximum)
    return result


def substitute_time(
    value: TimeExpression, replacements: dict[CycleAxis, TimeExpression]
) -> TimeExpression:
    """Substitute captured immutable sources, including nested Abs arguments."""
    if isinstance(value, TimeAbs):
        linear = substitute_time(value.linear, replacements)
        result = linear
        for argument, coefficient in value.absolute:
            substituted = substitute_time(argument, replacements)
            absolute = choose_time(substituted, scale_time(substituted, -1))
            result = add_time(result, scale_time(absolute, coefficient))
        return result
    unchanged = tuple(
        (axis, component)
        for axis, component in value.components
        if axis not in replacements
    )
    result: TimeExpression = TimeVariable(
        value.constant,
        tuple(
            (axis, coefficient)
            for axis, coefficient in value.terms
            if axis not in replacements
        ),
        components=unchanged,
    )
    for axis, component in value.components:
        if axis in replacements:
            if isinstance(component, np.ndarray):
                raise ValueError("cannot substitute an arbitrary known-axis component")
            result = add_time(result, scale_time(replacements[axis], component))
    return result
