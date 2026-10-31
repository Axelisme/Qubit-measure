"""Runtime cycle emission plans with correlated compile-time time expressions."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np
from qick.asm_v2 import AsmInst, Macro, QickParam, TimedMacro, WriteReg

from ._timeline_axis import CycleAxis, CycleProducer, TimeVariable
from ._timeline_bounds import CycleBounds
from ._timeline_expression import (
    TimeAbs,
    TimeExpression,
    add_time,
    ceiled_time,
    choose_time,
    cycle_axis,
    difference_limits,
    evaluate_time,
    expression_axes,
    expression_limits,
    known_time_point,
    point_time,
    plan_time_choice,
    projected_expression_bounds,
    scale_time,
    select_time,
    shifted_time,
    unbounded_time,
)
from ._timeline_guard import bounds
from .macro.write_reg import WriteRegOp
from .modules.util import get_fclk


def ceil_tick(ticks: float) -> int:
    """Ceil cycle values, ignoring floating noise within 1e-10 of an integer."""
    nearest = round(ticks)
    return (
        nearest
        if math.isclose(ticks, nearest, rel_tol=0, abs_tol=1e-10)
        else math.ceil(ticks)
    )


@dataclass(frozen=True)
class NativeLength:
    """Resolved generator length in us and its fabric clock in MHz.

    value retains QICK's native rounded start and truncated sweep steps. clock
    identifies the integer length lattice before any tProc conversion.
    """

    value: float | QickParam
    clock: float


def align_end(
    prog, value: float | QickParam, native: NativeLength | None = None
) -> "Time":
    """Ceil a resolved aggregate in us at every native sweep point.

    Call only at an aggregate or placement boundary during preprocessing.
    Scalars and exact integer-affine sweeps remain static. Non-affine native
    lengths use k=8 upper fixed-point conversion, retaining the aggregate tail.
    native identifies the original integer lattice, not an already rounded end.
    """

    cycles = value * get_fclk(prog)
    if not isinstance(cycles, QickParam) or not cycles.is_sweep():
        scalar = cycles.start if isinstance(cycles, QickParam) else cycles
        return ceil_tick(scalar) / get_fclk(prog)
    missing = cycles.spans.keys() - prog.loop_dict.keys()
    if missing:
        raise ValueError(
            f"{sorted(missing)}: missing native sweep axis for timeline end"
        )
    axes = {
        name: count for name, count in prog.loop_dict.items() if name in cycles.spans
    }
    if native is not None and isinstance(native.value, QickParam):
        ratio = Fraction(str(get_fclk(prog))) / Fraction(str(native.clock))
        steps = (
            Fraction(round(span * native.clock), prog.loop_dict[axis] - 1) * ratio
            for axis, span in native.value.spans.items()
            if prog.loop_dict[axis] > 1
        )
        if any(step.denominator != 1 for step in steps):
            # A few sampled ceils can look affine by coincidence. The runtime
            # consumer still owes the same upper conversion at every length.
            return prog._timeline_book.math.native_end(value, native)
    points = np.asarray(cycles.to_array(axes), dtype=float)
    rounded = np.array([ceil_tick(float(point)) for point in points.flat]).reshape(
        points.shape
    )
    start = int(rounded.flat[0])
    spans: dict[str, int] = {}
    for dimension, (axis, count) in enumerate(axes.items()):
        origin = [0] * rounded.ndim
        if count > 1:
            origin[dimension] = 1
            spans[axis] = (int(rounded[tuple(origin)]) - start) * (count - 1)
    affine = QickParam(start, spans)
    if np.array_equal(rounded, affine.to_array(axes, all_loops=True)):
        return step_time(prog, affine / get_fclk(prog))
    if native is None:
        raise ValueError("runtime cross-grid time requires generator/ADC conversion")
    return prog._timeline_book.math.native_end(value, native)


def placement_time(prog, value: float | QickParam) -> float | QickParam:
    """Ceil scalar placement; author sweeps retain native start/step quantization."""
    if isinstance(value, QickParam) and value.is_sweep():
        return step_time(prog, value)
    scalar = value.start if isinstance(value, QickParam) else value
    return ceil_tick(scalar * get_fclk(prog)) / get_fclk(prog)


def step_time(prog, value: float | QickParam) -> float | QickParam:
    if not isinstance(value, QickParam):
        return value
    missing = value.spans.keys() - prog.loop_dict.keys()
    if missing:
        raise ValueError(f"{sorted(missing)}: missing native sweep axis")
    # Use QICK public conversion/rounding APIs, never inspect raw sweep internals.
    param = QickParam(value.start, value.spans.copy())
    prog.us2cycles(param)
    return param.get_rounded(loop_counts=prog.loop_dict)


@dataclass(frozen=True)
class DeferredCoverage:
    """Proof that adding cycles to an aggregate covers one physical boundary.

    boundary identifies the same emission's physical end; it is never read as a
    timing operand. cycles is a nonnegative conservative debt in tProc ticks.
    Captured selection and raw paired posts establish it before any propagation.
    """

    boundary: "Time"
    cycles: int


@dataclass(frozen=True, init=False)
class RuntimeEnd:
    """A captured cycle register and its immutable proof expression.

    register names runtime storage; expression describes the integer value
    independently of later writes. minimum/maximum always derive from expression
    using the common source-range rule, including overflow and max decisions.
    None marks an unavailable endpoint. covers lists ends proven no later than
    this value. deferred records physical boundaries covered after an additional
    nonnegative advance; negative operations and reference debits drop it.

    Constructor endpoints only seed a fresh independent source when expression
    is omitted. Equal endpoints seed a constant. An explicit expression is
    authoritative; copying a value must retain it rather than a register name.
    Final ends must be nonnegative; arithmetic intermediates may be signed32.
    """

    register: str
    covers: tuple["Time", ...]
    deferred: tuple[DeferredCoverage, ...]
    expression: TimeExpression

    def __init__(
        self,
        register: str,
        minimum: int | None = None,
        maximum: int | None = None,
        covers: tuple["Time", ...] = (),
        deferred: tuple[DeferredCoverage, ...] = (),
        expression: TimeExpression | None = None,
    ) -> None:
        # Endpoints seed a fresh producer only. Existing algebra is authoritative.
        if expression is None:
            expression = (
                TimeVariable(minimum)
                if minimum is not None and minimum == maximum
                else TimeVariable(terms=((cycle_axis(register, minimum, maximum), 1),))
            )
        object.__setattr__(self, "register", register)
        object.__setattr__(self, "covers", covers)
        object.__setattr__(self, "deferred", deferred)
        object.__setattr__(self, "expression", expression)

    @property
    def minimum(self) -> int | None:
        """Return the expression's lower cycle bound, or None for unknown time."""
        lower, _ = expression_limits(self.expression)
        return None if lower is None else math.ceil(lower)

    @property
    def maximum(self) -> int | None:
        """Return the expression's upper cycle bound, or None for unknown time."""
        _, upper = expression_limits(self.expression)
        return None if upper is None else math.floor(upper)


def runtime_expression(value: RuntimeEnd) -> TimeExpression:
    """Return the immutable value identity, independent of mutable register storage."""
    return value.expression


@dataclass(frozen=True)
class MicrosecondBounds:
    """Inclusive native timestamp bounds in microseconds, possibly signed.

    minimum and maximum bound every legal point after cycle projection. They
    are conservative for unknown inputs and exact otherwise. The projection
    owner uses this pair to keep us separate from integer-cycle bounds.
    Construction stores endpoints unchanged; callers supply ordered values.
    """

    minimum: float
    maximum: float


def projected_bounds(prog, value: "Time") -> MicrosecondBounds:
    """Project correlated bounds to us at a native timestamp boundary.

    Unknown inputs remain conservative. Expression substitution failures
    propagate as RuntimeError; static values use their native parameter bounds.
    """
    if isinstance(value, RuntimeEnd):
        expression = runtime_expression(value)
        limits = projected_expression_bounds(expression)
        return MicrosecondBounds(
            limits.minimum / get_fclk(prog), limits.maximum / get_fclk(prog)
        )
    lower, upper = bounds(value)
    return MicrosecondBounds(lower, upper)


def parameter_expression(
    prog, value: float | QickParam, *, cycles: bool = False
) -> TimeExpression:
    """Describe an already-rounded native parameter without emitting instructions.

    Inputs normally use us; cycles=True describes raw register initialization.
    Shared loop names use the same compiler-owned variable identity.
    """
    factor = 1 if cycles else get_fclk(prog)
    if not isinstance(value, QickParam):
        return TimeVariable(round(value * factor))
    axes = prog._timeline_book.cycle_axes
    terms: list[tuple[CycleAxis, int]] = []
    for name, span in value.spans.items():
        count = prog.loop_dict[name]
        if count > 1:
            if name not in axes:
                axes[name] = cycle_axis(
                    name, 0, count - 1, CycleProducer(((name, count),))
                )
            terms.append((axes[name], round(span * factor / (count - 1))))
    return TimeVariable(round(value.start * factor), tuple(terms))


def time_expression(prog, value: "Time") -> TimeExpression:
    """Describe a relative end in integer cycles without register allocation."""
    return (
        runtime_expression(value)
        if isinstance(value, RuntimeEnd)
        else parameter_expression(prog, value)
    )


def native_event_expression(prog, value: "Time") -> TimeExpression:
    """Capture a physical event end with a safe integer-cycle projection.

    Raw us ends may include fractional waveform/ADC cycles. Ceil each legal
    point here, before integer reference debits; do not round a physical end
    like a register initialization. RuntimeEnd already stores integer cycles
    and retains its expression unchanged. This emits no instructions and does
    not participate in the module owner's separate event-plus-post rounding.
    Missing native sweep dimensions raise KeyError.
    """
    if isinstance(value, RuntimeEnd):
        return runtime_expression(value)
    ticks = value * get_fclk(prog)
    if not isinstance(ticks, QickParam):
        return TimeVariable(ceil_tick(ticks))
    axes = {name: prog.loop_dict[name] for name in ticks.spans}
    if len(axes) > 1:
        line = TimeVariable(Fraction(str(ticks.start)))
        for name, span in ticks.spans.items():
            count = axes[name]
            if count > 1:
                contribution = scale_time(
                    source_expression(prog, name), Fraction(str(span)) / (count - 1)
                )
                combined = add_time(line, contribution)
                if not isinstance(combined, TimeVariable):
                    raise RuntimeError("native affine time lost its loop algebra")
                line = combined
        return ceiled_time(line)
    raw = np.asarray(ticks.to_array(axes, all_loops=True), dtype=float)
    points = np.asarray(
        [ceil_tick(float(point)) for point in raw.flat], dtype=np.int64
    ).reshape(raw.shape)
    return point_time(", ".join(axes), axes, points)


def source_expression(prog, name: str) -> TimeExpression:
    """Return a register's current point provenance, or its certified unknown range."""
    book = prog._timeline_book
    if name in book.source_expressions:
        return book.source_expressions[name]
    if name in prog.loop_dict:
        count = prog.loop_dict[name]
        if name not in book.cycle_axes:
            book.cycle_axes[name] = cycle_axis(
                name, 0, count - 1, CycleProducer(((name, count),))
            )
        return TimeVariable(terms=((book.cycle_axes[name], 1),))
    register = prog.reg_dict.get(name)
    if register is not None and register.init is not None:
        raw = register.init
        return parameter_expression(prog, QickParam(raw.start, raw.spans), cycles=True)
    limits = prog._time_bounds.describe(prog, name)
    if limits.minimum == limits.maximum:
        return TimeVariable(limits.minimum)
    return TimeVariable(terms=((cycle_axis(name, limits.minimum, limits.maximum), 1),))


def bounded_source_expression(
    prog, name: str, minimum: int, maximum: int
) -> TimeExpression:
    """Describe a Repeat count using known points or its author-guaranteed range."""
    register = prog.reg_dict.get(name)
    if (
        name in prog._timeline_book.source_expressions
        or name in prog.loop_dict
        or (register is not None and register.init is not None)
    ):
        expression = source_expression(prog, name)
        if not unbounded_time(expression):
            return expression
    return (
        TimeVariable(minimum)
        if minimum == maximum
        else TimeVariable(
            terms=(
                (
                    cycle_axis(
                        name,
                        minimum,
                        maximum,
                        CycleProducer(
                            reconstruction=prog._timeline_book.source_expressions.get(
                                name
                            )
                        ),
                    ),
                    1,
                ),
            )
        )
    )


def lookup_expression(
    prog, name: str, values: tuple[int, ...], selector: str
) -> TimeExpression:
    """Preserve one lookup's legal points using the selector at this occurrence."""
    index = source_expression(prog, selector)
    reconstruction = None
    if len(values) > 1:
        step = values[1] - values[0]
        if all(value == values[0] + step * i for i, value in enumerate(values)):
            reconstruction = add_time(TimeVariable(values[0]), scale_time(index, step))
    axes = expression_axes(index)
    if len(axes) > 1:
        if reconstruction is not None:
            return reconstruction
        return TimeVariable(
            terms=(
                (
                    cycle_axis(
                        name,
                        min(values),
                        max(values),
                        CycleProducer(selection=(index, values)),
                    ),
                    1,
                ),
            )
        )
    points = evaluate_time(index, axes)
    if points is None:
        return TimeVariable(terms=((cycle_axis(name, min(values), max(values)), 1),))
    selected = np.asarray(values, dtype=np.int64)[points.astype(np.int64)]
    lower, upper = int(selected.min()), int(selected.max())
    if lower == upper:
        return TimeVariable(lower)
    variable = cycle_axis(
        name,
        lower,
        upper,
        CycleProducer(
            tuple(axes.items()),
            tuple(int(v) for v in selected.flat),
            reconstruction=reconstruction,
            selection=(index, values),
        ),
    )
    return TimeVariable(terms=((variable, 1),))


Time = float | QickParam | RuntimeEnd
LIMIT = 2**31 - 1


@dataclass(frozen=True)
class EventEnd:
    """Unrounded output end as an origin plus a fractional offset in us.

    origin is the timeline placement point, possibly an integer-cycle register.
    offset includes resolved event start offset and waveform/ADC duration;
    a runtime offset is already an integer-cycle value.
    Keeping it separate defers ceil until the owner adds its post-delay.
    native retains the integer length lattice for runtime conversion, or None
    for operations without a swept generator length.
    rounded contains ceiled event boundaries used only for same-channel
    no-overlap proof after a nonnegative post; it never supplies aggregate time.
    """

    origin: Time
    offset: Time
    # The ceiled event boundary, retained only for same-channel no-overlap proof.
    rounded: tuple["Time", ...] = ()
    # Exact static offset in tProc cycles, retained only inside this operation.
    static_cycles: Fraction | None = None
    native: NativeLength | None = None


@dataclass(frozen=True)
class SelectedEventEnd:
    """Raw selected output offsets and paired posts, relative to origin (us).

    selector is a certified zero-based register. offsets and posts have equal
    nonzero lengths in selector order. rounded carries physical channel ends
    solely for coverage proof; it is never the aggregate timing source.
    native pairs each offset with its generator-length lattice when available.
    """

    origin: Time
    selector: str
    offsets: tuple[float | QickParam, ...]
    posts: tuple[float | QickParam, ...]
    rounded: tuple[Time, ...] = ()
    native: tuple[NativeLength, ...] = ()


def optional_time_bounds(prog, value: Time) -> tuple[float, float] | None:
    """Return finite us bounds, or None for an unbounded runtime value."""
    if isinstance(value, RuntimeEnd):
        if value.minimum is None or value.maximum is None:
            return None
        return value.minimum / get_fclk(prog), value.maximum / get_fclk(prog)
    return bounds(value)


def time_bounds(prog, value: Time) -> tuple[float, float]:
    """Require finite us bounds; runtime-only inputs raise ValueError."""
    result = optional_time_bounds(prog, value)
    if result is None:
        raise ValueError("runtime-only time has no compile-time bounds")
    return result


def _csd(value: int) -> list[tuple[int, int]]:
    digits: list[tuple[int, int]] = []
    exponent = 0
    while value:
        if value & 1:
            sign = 2 - value % 4
            digits.append((exponent, sign))
            value -= sign
        value //= 2
        exponent += 1
    return digits


def _conversion_weight(ratio: Fraction, clock: float, tproc: float) -> int:
    # For each residue modulo 256 the ceil error grows with n. Its final
    # member therefore bounds every length <=500us, without a point table.
    maximum = math.floor(500 * clock)
    lower = math.ceil(256 * ratio)
    upper = math.floor(256 * (ratio + Fraction(str(tproc)) / maximum))
    candidates = []
    for weight in range(lower, upper + 1):
        if all(
            Fraction((n * weight + 255) // 256) - n * ratio <= Fraction(str(tproc))
            for n in range(max(0, maximum - 255), maximum + 1)
        ):
            candidates.append(weight)
    if not candidates:
        raise ValueError(
            "k=8 runtime conversion cannot meet the 500us/1us error budget"
        )
    return min(candidates, key=lambda weight: (len(_csd(weight)), weight))


class RuntimeMath(TimedMacro):
    """Prepare integer cycle arithmetic and conditional candidate selection."""

    def __init__(self, prog):
        super().__init__()
        self.prog = prog
        self.operations: list[Macro] = []
        self.floors: dict[tuple[int, int], tuple[Time, Time, Time]] = {}
        self.claimed = set(prog._timeline_live_registers())
        previous = prog._timeline_book.math if prog._timeline_book.preparing else None
        if previous is not None:
            self.claimed.update(previous.claimed)

    def native_end(
        self, aggregate: float | QickParam, native: NativeLength
    ) -> RuntimeEnd:
        """Ceil one aggregate using a checked CSD upper fabric conversion.

        Only runtime cross-grid values reach here. Static and integer-affine
        conversions stay exact. Every multiply/shift intermediate is signed32.
        """
        clock = native.clock
        raw = native.value * clock
        counts = (
            QickParam(round(raw.start), {k: round(v) for k, v in raw.spans.items()})
            if isinstance(raw, QickParam)
            else round(raw)
        )
        ratio = Fraction(str(get_fclk(self.prog))) / Fraction(str(clock))
        weight = _conversion_weight(ratio, clock, get_fclk(self.prog))
        sources = self.prog._timeline_book.native_length_sources
        key = (
            (
                round(counts.start),
                tuple(sorted((k, round(v)) for k, v in counts.spans.items())),
            )
            if isinstance(counts, QickParam)
            else (counts, ())
        )
        if key not in sources:
            sources[key] = self.cycles(counts / get_fclk(self.prog))
        source = sources[key]
        product = self.multiply(source, weight)
        tail = (aggregate - native.value) * get_fclk(self.prog)
        if isinstance(tail, QickParam):
            constant = tail.start
            spans = {k: round(v * 256) for k, v in tail.spans.items()}
        else:
            constant, spans = tail, {}
        # The product is integer. Ceil the fractional bias once, keeping the
        # public 1e-10-cycle tolerance at the complete aggregate boundary.
        nearest = round(constant * 256)
        bias = (
            nearest
            if abs(constant - nearest / 256) <= 1e-10
            else math.ceil(constant * 256)
        )
        if spans:
            tail_value = self.cycles(QickParam(bias + 255, spans) / get_fclk(self.prog))
            numerator = self.op(
                product,
                "+",
                tail_value,
                limits=self._sum_limits(product, tail_value),
                target=product.register,
            )
        else:
            numerator = self.op(
                product,
                "+",
                bias + 255,
                limits=self._sum_limits(product, bias + 255),
                target=product.register,
            )
        if numerator.minimum is None or numerator.maximum is None:
            raise RuntimeError("native conversion lost its known bounds")
        result = self.op(
            numerator,
            "ASR",
            8,
            limits=(numerator.minimum // 256, numerator.maximum // 256),
            target=product.register,
        )
        return RuntimeEnd(
            result.register,
            expression=shifted_time(runtime_expression(numerator), 8),
        )

    def multiply(self, value: RuntimeEnd, factor: int) -> RuntimeEnd:
        """Multiply by a nonnegative integer with checked CSD shifts/adds."""
        if factor < 0:
            raise ValueError("constant multiplication requires a nonnegative factor")
        if factor == 1:
            return value
        digits = _csd(factor)
        if not digits:
            return self.cycles(0.0)
        coefficient = 1
        result = self.snapshot(value)
        previous = digits[-1][0]
        for exponent, sign in reversed(digits[:-1]):
            shift = previous - exponent
            while shift:
                amount = min(15, shift)
                result = self.op(
                    result,
                    "SL",
                    amount,
                    limits=(
                        None if result.minimum is None else result.minimum << amount,
                        None if result.maximum is None else result.maximum << amount,
                    ),
                    target=result.register,
                )
                coefficient <<= amount
                shift -= amount
            coefficient += sign
            result = self.op(
                result,
                "+" if sign > 0 else "-",
                value,
                # Both operands are multiples of this same native source.
                limits=(
                    None if value.minimum is None else coefficient * value.minimum,
                    None if value.maximum is None else coefficient * value.maximum,
                ),
                target=result.register,
            )
            previous = exponent
        while previous:
            amount = min(15, previous)
            result = self.op(
                result,
                "SL",
                amount,
                limits=(
                    None if result.minimum is None else result.minimum << amount,
                    None if result.maximum is None else result.maximum << amount,
                ),
                target=result.register,
            )
            previous -= amount
        return result

    def reconstruct(
        self,
        expression: TimeExpression,
        inputs: dict[CycleAxis, RuntimeEnd],
        captured_ends: Sequence[RuntimeEnd] = (),
    ) -> RuntimeEnd:
        """Materialize exact cycle algebra using captured branch-entry inputs.

        Integer producer algebra uses native axis registers and captured entry
        snapshots, never an automatically allocated point table or stale storage.
        captured_ends are immutable incoming timeline values. Reuse their cycle
        algebra before reading source variables, which may have been overwritten
        since those ends were computed. Other variables must be known on the
        point grid or supplied in inputs.
        """
        for captured in captured_ends:
            delta = add_time(expression, scale_time(runtime_expression(captured), -1))
            if delta is not None:
                lower, upper = expression_limits(delta)
                if lower is not None and lower >= 0:
                    if upper == 0:
                        return captured
                    return self.add(captured, self.reconstruct(delta, inputs))
        if isinstance(expression, TimeAbs):
            operation = expression.runtime
            values = [
                self.reconstruct(item, inputs, captured_ends)
                for item in operation.operands
            ]
            if operation.kind == "add":
                return self.add(values[0], values[1])
            if operation.kind == "scale":
                coefficient = operation.coefficient
                if coefficient != int(coefficient):
                    raise RuntimeError("runtime recipe requires an integer multiplier")
                scaled = self.multiply(values[0], abs(int(coefficient)))
                return scaled if coefficient >= 0 else self.subtract(0.0, scaled)
            return self.cycles(
                self.join(values, maximum=operation.kind == "max")
            )
        return self._reconstruct_linear(expression, inputs)

    def _reconstruct_linear(
        self, expression: TimeVariable, inputs: dict[CycleAxis, RuntimeEnd]
    ) -> RuntimeEnd:
        quotient = self._reconstruct_quotients(expression, inputs)
        if quotient is not None:
            return quotient
        if all(
            variable not in inputs
            and len(variable.producer.axes) == 1
            and variable.producer.points is None
            for variable, _ in expression.terms
        ):
            spans: dict[str, int] = {}
            for variable, coefficient in expression.terms:
                axis, count = variable.producer.axes[0]
                amount = coefficient * (count - 1)
                if amount != int(amount):
                    raise RuntimeError(
                        "fractional cycle span lost its conversion recipe"
                    )
                spans[axis] = spans.get(axis, 0) + int(amount)
            return self.cycles(
                QickParam(float(expression.constant), spans) / get_fclk(self.prog)
            )
        result = self.cycles(float(expression.constant) / get_fclk(self.prog))
        for variable, coefficient in expression.terms:
            if coefficient != int(coefficient):
                raise RuntimeError(
                    "fractional cycle coefficient lost its conversion recipe"
                )
            coefficient = int(coefficient)
            if variable in inputs:
                term = inputs[variable]
            elif variable.producer.reconstruction is not None:
                term = self.reconstruct(variable.producer.reconstruction, inputs)
                term = replace(
                    term,
                    expression=TimeVariable(terms=((variable, 1),)),
                )
            elif variable.producer.fixed_point is not None:
                term = self._reconstruct_fixed_point(variable, inputs)
            elif variable.producer.selection is not None:
                index, words = variable.producer.selection
                selected = self.reconstruct(index, inputs)
                term = self.selected_end(
                    selected.register,
                    tuple(word / get_fclk(self.prog) for word in words),
                    selector_expression=index,
                )
            elif len(variable.producer.axes) == 1 and variable.producer.points is None:
                axis, count = variable.producer.axes[0]
                term = self.cycles(
                    QickParam(0, {axis: count - 1}) / get_fclk(self.prog)
                )
            else:
                raise RuntimeError(
                    f"{variable.name}: integer time expression lost its runtime producer"
                )
            scaled = self.multiply(term, abs(coefficient))
            result = (
                self.add(
                    result,
                    scaled,
                    target=result.register if result.register != "s0" else None,
                )
                if coefficient > 0
                else self.subtract(result, scaled)
            )
        return result

    def _reconstruct_quotients(
        self, expression: TimeVariable, inputs: dict[CycleAxis, RuntimeEnd]
    ) -> RuntimeEnd | None:
        """Recover whole integer shifts before reconstructing rational proof terms."""
        for variable, coefficient in expression.terms:
            recipe = variable.producer
            if recipe.rounding_remainder and recipe.fixed_point is not None:
                numerator, shift = recipe.fixed_point
                divisor = 1 << shift
                copies = -coefficient * divisor
                if copies != int(copies):
                    raise RuntimeError(
                        "fractional conversion lost its complete quotient"
                    )
                quotient = add_time(
                    scale_time(numerator, Fraction(1, divisor)),
                    TimeVariable(terms=((variable, Fraction(-1, divisor)),)),
                )
                rest = add_time(expression, scale_time(quotient, -copies))
                converted = self._reconstruct_fixed_point(variable, inputs)
                scaled = self.multiply(converted, abs(int(copies)))
                remainder = self.reconstruct(rest, inputs)
                return (
                    self.add(remainder, scaled)
                    if copies > 0
                    else self.subtract(remainder, scaled)
                )
        return None

    def _reconstruct_fixed_point(
        self, variable: CycleAxis, inputs: dict[CycleAxis, RuntimeEnd]
    ) -> RuntimeEnd:
        if variable.producer.fixed_point is None:
            raise RuntimeError("fixed-point producer has no numerator")
        numerator, shift = variable.producer.fixed_point
        raw = self.reconstruct(numerator, inputs)
        if raw.minimum is None or raw.maximum is None:
            raise RuntimeError("fixed-point reconstruction lost bounds")
        result = self.op(
            raw, "ASR", shift, limits=(raw.minimum >> shift, raw.maximum >> shift)
        )
        expression = TimeVariable(terms=((variable, 1),))
        if variable.producer.rounding_remainder:
            divisor = 1 << shift
            expression = add_time(
                scale_time(numerator, Fraction(1, divisor)),
                TimeVariable(terms=((variable, Fraction(-1, divisor)),)),
            )
        return RuntimeEnd(result.register, expression=expression)

    def join(
        self,
        values: Sequence[Time],
        inputs: dict[CycleAxis, RuntimeEnd] | None = None,
        captured_ends: Sequence[RuntimeEnd] = (),
        *,
        maximum: bool = True,
        target: str | None = None,
    ) -> Time:
        """Materialize one N-way max/min from original recipes and live bindings.

        values are reference-relative microseconds or captured cycle registers.
        With inputs supplied, reconstruct branch results from entry registers;
        otherwise RuntimeEnd values are opaque live operands. Captured entry
        max values never expand into historical source-register reads.
        target may alias the first operand only when that operand is dead.
        """
        if not values:
            raise ValueError("time choice requires at least one value")
        expressions = tuple(time_expression(self.prog, value) for value in values)
        captured = (
            tuple(value for value in values if isinstance(value, RuntimeEnd))
            if inputs is None
            else tuple(captured_ends)
        )
        bindings = {runtime_expression(value): value for value in captured}
        common, residuals, atoms = plan_time_choice(
            expressions, maximum=maximum, captured=tuple(bindings)
        )
        sources = {} if inputs is None else inputs.copy()
        for axis, expression in atoms.items():
            if expression in bindings:
                sources[axis] = bindings[expression]
        shared = self.reconstruct(common, sources)
        operands = [self.reconstruct(value, sources, (shared,)) for value in residuals]
        result = operands[0]
        destination = target
        if len(operands) > 1:
            destination = self.allocate() if target is None else target
        for operand in operands[1:]:
            result = self._select(result, operand, maximum=maximum, target=destination)
        result = self.add(shared, result, target=destination)
        proof = choose_time(*expressions, maximum=maximum)
        covers = tuple(
            item
            for value in values
            for item in (value, *(value.covers if isinstance(value, RuntimeEnd) else ()))
        ) if maximum else ()
        return replace(result, expression=proof, covers=covers)

    def source(
        self,
        name: str,
        range_hint: tuple[int, int] | None = None,
        *,
        author: bool = False,
    ) -> RuntimeEnd:
        """Read cycle bounds, permitting unbounded author inputs when requested.

        Hints are author guarantees, not hardware checks. Certified framework
        sources keep their existing producer audit.
        """
        if not author:
            limits = self.prog._time_bounds.source(self.prog, name)
        else:
            try:
                self.prog._get_reg(name)
            except KeyError as exc:
                raise ValueError(f"{name}: unknown time register") from exc
            if range_hint is not None:
                if (
                    name in self.prog._time_bounds.framework_sources
                    or not self.prog._time_bounds.author_source(name)
                    or name in self.prog.loop_dict
                ):
                    # A hint supplies bounds, not permission to bypass owner audits.
                    self.prog._time_bounds.source(self.prog, name)
                limits = CycleBounds(*range_hint)
            elif (
                name not in self.prog.loop_dict
                and self.prog._time_bounds.author_source(name)
            ):
                limits = None
            else:
                limits = self.prog._time_bounds.source(self.prog, name)
        if limits is None:
            expression = self.prog._timeline_book.source_expressions.get(name)
            return RuntimeEnd(name, None, None, expression=expression)
        expression = (
            bounded_source_expression(self.prog, name, limits.minimum, limits.maximum)
            if range_hint is not None
            else source_expression(self.prog, name)
        )
        return RuntimeEnd(name, limits.minimum, limits.maximum, expression=expression)

    def _selected_axes(self, values: tuple[float | QickParam, ...]) -> dict[str, int]:
        spans = set().union(
            *(v.spans.keys() for v in values if isinstance(v, QickParam))
        )
        missing = spans - self.prog.loop_dict.keys()
        if missing:
            raise ValueError(
                f"{sorted(missing)}: missing native sweep axis for selected end"
            )
        return {
            name: count for name, count in self.prog.loop_dict.items() if name in spans
        }

    def _selected_points(
        self, values: tuple[float | QickParam, ...], axes: dict[str, int]
    ) -> list[int]:
        points: list[int] = []
        for value in values:
            param = value if isinstance(value, QickParam) else QickParam(value)
            points.extend(
                ceil_tick(float(p) * get_fclk(self.prog))
                for p in np.broadcast_to(
                    np.asarray(param.to_array(axes, all_loops=True)),
                    tuple(axes.values()),
                ).flat
            )
        return points

    def coverage_debt(
        self,
        raw: tuple[float | QickParam, ...],
        completed: tuple[float | QickParam, ...],
    ) -> int:
        """Prove a conservative tick debt from paired raw and aggregate ends.

        Both tuples use the same captured selection and origin. Evaluate the
        shared exact-ceil rule on their common native grid, without emitting
        instructions or using rounded physical metadata as a timing source.
        """
        axes = self._selected_axes(raw + completed)
        physical = self._selected_points(raw, axes)
        aggregate = self._selected_points(completed, axes)
        return max(0, *(a - b for a, b in zip(physical, aggregate, strict=True)))

    def selected_end(
        self,
        selector: str,
        values: tuple[float | QickParam, ...],
        *,
        selector_expression: TimeExpression | None = None,
        native: tuple[NativeLength, ...] = (),
    ) -> RuntimeEnd:
        """Select statically ceiled candidate offsets without allocating time data.

        Static candidates ceil exactly. Swept generator candidates use native's
        paired integer lattice for runtime fixed-point conversion. Native loop
        axes retain their register initialization. The already certified selector
        chooses the candidate at runtime, without longest-candidate padding.
        """
        candidates = tuple(
            align_end(self.prog, value, native[index] if native else None)
            for index, value in enumerate(values)
        )
        limits = [time_bounds(self.prog, value) for value in candidates]
        lower = min(round(lo * get_fclk(self.prog)) for lo, _ in limits)
        upper = max(round(hi * get_fclk(self.prog)) for _, hi in limits)
        self.check(lower, upper)
        operands: list[int | str] = []
        for value in candidates:
            if isinstance(value, RuntimeEnd):
                operands.append(value.register)
            elif isinstance(value, QickParam) and value.is_sweep():
                operands.append(self.cycles(value).register)
            else:
                scalar = value.start if isinstance(value, QickParam) else value
                if isinstance(scalar, RuntimeEnd):
                    operands.append(scalar.register)
                else:
                    operands.append(round(scalar * get_fclk(self.prog)))
        register = (
            candidates[0].register
            if isinstance(candidates[0], RuntimeEnd)
            else self.allocate()
        )
        for index, operand in enumerate(operands):
            inst = {
                "CMD": "REG_WR",
                "DST": self.prog._get_reg(register),
                "SRC": "imm" if isinstance(operand, int) else "op",
            }
            inst["LIT" if isinstance(operand, int) else "OP"] = (
                f"#{operand}"
                if isinstance(operand, int)
                else self.prog._get_reg(operand)
            )
            if index:
                self.operations.append(
                    AsmInst(
                        inst={
                            "CMD": "TEST",
                            "OP": f"{self.prog._get_reg(selector)} - #{index}",
                            "UF": "1",
                        },
                        addr_inc=1,
                    )
                )
                inst["IF"] = "Z"
            self.operations.append(AsmInst(inst=inst, addr_inc=1))
        # Candidate conversions have been consumed by the selection writes.
        # Their immutable expressions survive; their storage is no longer live.
        for candidate in candidates:
            if isinstance(candidate, RuntimeEnd) and candidate.register != register:
                self.claimed.discard(candidate.register)
        return RuntimeEnd(
            register,
            lower,
            upper,
            expression=select_time(
                source_expression(self.prog, selector)
                if selector_expression is None
                else selector_expression,
                tuple(time_expression(self.prog, value) for value in candidates),
            ),
        )

    def cycles(self, value: Time) -> RuntimeEnd:
        if isinstance(value, RuntimeEnd):
            return value
        key = str(len(self.t_regs))
        rounded = self.convert_time(self.prog, value, key)
        self.prog._time_bounds.time_parameters(self)
        if rounded is None:
            raise ValueError("timeline time cannot be None")
        lo, hi = bounds(rounded)
        lower, upper = round(lo * get_fclk(self.prog)), round(hi * get_fclk(self.prog))
        self.check(lower, upper)
        operand = self.t_regs[key]
        if isinstance(operand, str):
            expression = parameter_expression(self.prog, rounded)
            return RuntimeEnd(operand, lower, upper, expression=expression)
        if operand == 0:
            return RuntimeEnd("s0", 0, 0)
        register = self.allocate()
        self.operations.append(WriteReg(dst=register, src=operand))
        return RuntimeEnd(register, lower, upper)

    @staticmethod
    def check(lower: int, upper: int) -> None:
        if lower < -(2**31) or upper > LIMIT:
            raise ValueError("timeline arithmetic may overflow signed 32-bit cycles")

    def check_end(self, end: Time) -> None:
        """Check an end without allocating registers for a static expression."""
        limits = optional_time_bounds(self.prog, end)
        if limits is None:
            return
        lo, hi = limits
        lower, upper = round(lo * get_fclk(self.prog)), round(hi * get_fclk(self.prog))
        self.check(lower, upper)
        if lower < 0:
            raise ValueError("timeline end must be nonnegative cycles")

    def preprocess_time(self, value: RuntimeEnd) -> float:
        """Supply QICK allocation metadata, not a runtime scheduling bound.

        QICK requires a scalar while allocating outputs. Unbounded starts use
        an inert placeholder; callers replace t_regs before translate and reject
        runtime-only time getters.
        """
        limits = optional_time_bounds(self.prog, value)
        return 0.0 if limits is None else limits[1]

    def time_parameter(self, value: RuntimeEnd) -> QickParam:
        """Return an exactly known affine time, never runtime allocation metadata.

        A runtime-only or non-affine value cannot be returned as QickParam and
        raises RuntimeError. This read does not alter the scheduled end.
        """
        expression = runtime_expression(value)
        axes = expression_axes(expression)
        origin = known_time_point(expression, {})
        if origin is None:
            raise RuntimeError(
                "time is runtime-only; cannot evaluate it at compile time"
            )
        start = int(origin)
        spans = {}
        terms: list[tuple[CycleAxis, Fraction]] = []
        for axis, count in axes.items():
            if count > 1:
                endpoint = known_time_point(expression, {axis: count - 1})
                if endpoint is None:
                    raise RuntimeError(
                        "time is runtime-only; cannot evaluate it at compile time"
                    )
                span = int(endpoint) - start
                spans[axis] = span
                variable = cycle_axis(
                    axis, 0, count - 1, CycleProducer(((axis, count),))
                )
                terms.append((variable, Fraction(span, count - 1)))
        candidate = TimeVariable(start, tuple(terms))
        if difference_limits(expression, candidate) != (0, 0):
            raise RuntimeError(
                "non-affine runtime time cannot be represented by QickParam"
            )
        return QickParam(start, spans) / get_fclk(self.prog)

    def allocate(self) -> str:
        live = self.prog._timeline_live_registers()
        for register in self.prog._timeline_register_pool:
            if register not in self.claimed and register not in live:
                self.claimed.add(register)
                return register
        register = self.prog.add_reg(
            f"_timeline_end_{len(self.prog._timeline_register_pool)}"
        )
        self.prog._timeline_register_pool.append(register)
        self.claimed.add(register)
        return register

    def scratch(self, index: int) -> str:
        return self.prog.add_reg(f"_timeline_tmp_{index}", allow_reuse=True)

    def op(
        self,
        left: RuntimeEnd,
        operator: str,
        right: RuntimeEnd | int,
        *,
        limits: tuple[int | None, int | None],
        target: str | None = None,
    ) -> RuntimeEnd:
        register = self.allocate() if target is None else target
        self.operations.append(
            WriteRegOp(
                dst=register,
                lhs=left.register,
                op=operator,
                rhs=right.register if isinstance(right, RuntimeEnd) else right,
            )
        )
        covers: tuple[Time, ...] = ()
        deferred: list[DeferredCoverage] = []
        expression = None
        if operator in ("+", "-"):
            rhs = (
                runtime_expression(right)
                if isinstance(right, RuntimeEnd)
                else TimeVariable(right)
            )
            expression = add_time(
                runtime_expression(left),
                rhs if operator == "+" else scale_time(rhs, -1),
            )
        if operator == "SL" and isinstance(right, int):
            expression = scale_time(runtime_expression(left), 1 << right)
        if operator == "ASR" and isinstance(right, int):
            numerator = runtime_expression(left)
            if expression_axes(numerator):
                expression = shifted_time(numerator, right)
        if operator == "+":
            right_minimum = right.minimum if isinstance(right, RuntimeEnd) else right
            if right_minimum is not None and right_minimum >= 0:
                covers = (left, *left.covers)
                for proof in left.deferred:
                    remaining = max(0, proof.cycles - right_minimum)
                    if remaining:
                        deferred.append(DeferredCoverage(proof.boundary, remaining))
                    else:
                        covers += (proof.boundary,)
        result = RuntimeEnd(register, *limits, covers, tuple(deferred), expression)
        if result.minimum is not None and result.maximum is not None:
            self.check(result.minimum, result.maximum)
        return result

    def covers(self, end: Time, previous: Time) -> bool:
        """Prove coverage with exact identities or the common source-range rule."""
        if end is previous or (isinstance(end, RuntimeEnd) and any(item is previous for item in end.covers)):
            return True
        lower, _ = difference_limits(
            time_expression(self.prog, end), time_expression(self.prog, previous)
        )
        return lower is not None and lower >= 0

    def snapshot(self, value: RuntimeEnd) -> RuntimeEnd:
        register = self.allocate()
        self.operations.append(WriteReg(dst=register, src=value.register))
        return RuntimeEnd(
            register,
            value.minimum,
            value.maximum,
            value.covers,
            value.deferred,
            runtime_expression(value),
        )

    def add(self, a: Time, b: Time, *, target: str | None = None) -> RuntimeEnd:
        if isinstance(b, QickParam) and not b.is_sweep():
            b = b.start
        left = self.cycles(a)
        if left.minimum == left.maximum == 0:
            return self.cycles(b)
        if isinstance(b, (float, int)):
            amount = round(b * get_fclk(self.prog))
            if amount == 0:
                return left
            return self.op(
                left,
                "+",
                amount,
                limits=self._sum_limits(left, amount),
                target=target,
            )
        right = self.cycles(b)
        if right.minimum == right.maximum == 0:
            return left
        return self.op(
            left,
            "+",
            right,
            limits=self._sum_limits(left, right),
            target=target,
        )

    def subtract(self, a: Time, b: Time) -> RuntimeEnd:
        if isinstance(b, QickParam) and not b.is_sweep():
            b = b.start
        left = self.cycles(a)
        if isinstance(b, (float, int)):
            amount = round(b * get_fclk(self.prog))
            return self.op(left, "-", amount, limits=self._sum_limits(left, -amount))
        right = self.cycles(b)
        return self.op(
            left,
            "-",
            right,
            limits=self._sum_limits(left, right, subtract=True),
        )

    def maximum(self, a: Time, b: Time, *, target: str | None = None) -> Time:
        """Join two ends through the common N-way recipe planner."""
        return self.join((a, b), target=target)

    def _select(
        self, left: RuntimeEnd, right: RuntimeEnd, *, maximum: bool, target: str | None
    ) -> RuntimeEnd:
        lower, upper = difference_limits(runtime_expression(left), runtime_expression(right))
        safe = lower is not None and upper is not None and -(2**31) <= lower <= upper <= LIMIT
        if not safe:
            return self._masked_select(left, right, maximum=maximum, target=target)
        destination = self.allocate() if target is None else target
        self.operations.append(
            AsmInst(
                inst={
                    "CMD": "TEST",
                    "OP": f"{self.prog._get_reg(left.register)} - {self.prog._get_reg(right.register)}",
                    "UF": "1",
                },
                addr_inc=1,
            )
        )
        if destination != left.register:
            self.operations.append(WriteReg(dst=destination, src=left.register))
        self.operations.append(
            AsmInst(
                inst={
                    "CMD": "REG_WR",
                    "DST": self.prog._get_reg(destination),
                    "SRC": "op",
                    "OP": self.prog._get_reg(right.register),
                    "IF": "S" if maximum else "NS",
                },
                addr_inc=1,
            )
        )
        return RuntimeEnd(destination, expression=choose_time(
            runtime_expression(left), runtime_expression(right), maximum=maximum
        ))

    @staticmethod
    def _sum_limits(
        left: RuntimeEnd, right: RuntimeEnd | int, *, subtract: bool = False
    ) -> tuple[int | None, int | None]:
        rhs = (
            runtime_expression(right)
            if isinstance(right, RuntimeEnd)
            else TimeVariable(right)
        )
        expression = add_time(
            runtime_expression(left), scale_time(rhs, -1) if subtract else rhs
        )
        lower, upper = expression_limits(expression)
        return None if lower is None else math.ceil(
            lower
        ), None if upper is None else math.floor(upper)

    def _masked_select(
        self, left: RuntimeEnd, right: RuntimeEnd, *, maximum: bool, target: str | None
    ) -> RuntimeEnd:
        """Select signed words without trusting an overflowing subtraction sign."""
        destination = self.allocate() if target is None else target
        delta, signs, mask = (self.scratch(index) for index in range(3))
        x, y = left.register, right.register

        def emit(dst: str, a: str, operator: str, b: str | int) -> None:
            rhs = f"#{b}" if isinstance(b, int) else self.prog._get_reg(b)
            self.operations.append(AsmInst(inst={
                "CMD": "REG_WR", "DST": self.prog._get_reg(dst), "SRC": "op",
                "OP": f"{self.prog._get_reg(a)} {operator} {rhs}",
            }, addr_inc=1))

        emit(delta, x, "-", y)
        emit(signs, x, "^", y)
        emit(mask, x, "^", delta)
        emit(mask, mask, "AND", signs)
        emit(mask, mask, "^", delta)
        # ASR counts are four-bit hardware operands.
        for shift in (15, 15, 1):
            emit(mask, mask, "ASR", shift)
        if not maximum:
            emit(mask, mask, "^", -1)
        emit(delta, x, "^", y)
        emit(delta, delta, "AND", mask)
        emit(destination, x, "^", delta)
        return RuntimeEnd(destination, expression=choose_time(
            runtime_expression(left), runtime_expression(right), maximum=maximum
        ))

    def floor(self, end: Time, advance: Time) -> Time:
        key = (id(end), id(advance))
        if key not in self.floors:
            self.floors[key] = (end, advance, self._floor(end, advance))
        return self.floors[key][2]

    def _floor(self, end: Time, advance: Time) -> Time:
        if (
            isinstance(end, RuntimeEnd)
            and isinstance(advance, RuntimeEnd)
            and end.register == advance.register
        ):
            return 0.0
        if end is advance or (
            isinstance(advance, RuntimeEnd)
            and any(item is end for item in advance.covers)
        ):
            return 0.0
        if not isinstance(end, RuntimeEnd) and not isinstance(advance, RuntimeEnd):
            return self.maximum(0.0, end - advance)
        limits = optional_time_bounds(self.prog, end)
        advance_limits = optional_time_bounds(self.prog, advance)
        if (
            limits is not None
            and advance_limits is not None
            and limits[1] <= advance_limits[0] + 1e-12
        ):
            return 0.0
        delta = self.subtract(end, advance)
        return self.maximum(0.0, delta, target=delta.register)

    def translate(self, prog) -> None:
        previous = prog._timeline_math_active
        prog._timeline_math_active = True
        try:
            for operation in self.operations:
                # Control lowering owns s15/flags, not compiler time storage.
                self.prog._timeline_math_active = not (
                    isinstance(operation, AsmInst) and operation.inst["CMD"] == "TEST"
                )
                operation.translate(prog)
        finally:
            prog._timeline_math_active = previous
