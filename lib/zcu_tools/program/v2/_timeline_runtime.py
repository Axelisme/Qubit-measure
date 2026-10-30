"""Runtime cycle emission plans with correlated compile-time time expressions."""

import math
from dataclasses import dataclass

import numpy as np
from qick.asm_v2 import AsmInst, Macro, QickParam, TimedMacro, WriteReg

from ._timeline_bounds import CycleBounds
from ._timeline_expression import (
    CycleVariable,
    LinearTime,
    TimeExpression,
    add_time,
    choose_time,
    evaluate_time,
    expression_axes,
    projected_expression_bounds,
    scale_time,
)
from ._timeline_guard import bounds
from .macro.dmem import DmemLookup, sweep_table_index
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


def align_end(prog, value: float | QickParam) -> "Time":
    """Ceil a resolved aggregate in us at every native sweep point.

    Call only at an aggregate or placement boundary during preprocessing.
    Scalars and exact integer-affine sweeps remain static. Other point grids
    become a certified dmem lookup at this operation, without advancing reference.
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
    return prog._timeline_book.math.lookup(rounded.flatten().tolist(), axes)


def placement_time(prog, value: float | QickParam) -> float | QickParam:
    """Resolve QICK nearest-cycle placement, including scalar and sweep steps."""
    return step_time(prog, value if isinstance(value, QickParam) else QickParam(value))


def step_time(prog, value: float | QickParam) -> float | QickParam:
    if not isinstance(value, QickParam):
        return value
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


@dataclass(frozen=True)
class RuntimeEnd:
    """Relative cycle end stored in register, with inclusive minimum/maximum.

    register is a compiler-owned data-register name. minimum/maximum bound the
    value on every executable path. Arithmetic intermediates may be signed32;
    final timeline ends must be nonnegative. covers lists ends proven no later than this
    value, allowing a following sync to consume them without new arithmetic.
    deferred records physical boundaries covered only after an additional proven
    nonnegative cycle advance; negative operations and reference debits drop it.
    expression carries correlated integer-cycle algebra for native timestamp
    projection only. None creates a fresh bounded value (or an exact constant);
    copying a value must preserve its expression, not its recycled register name.
    The original intervals still control register arithmetic and overflow checks.
    """

    register: str
    minimum: int
    maximum: int
    covers: tuple["Time", ...] = ()
    deferred: tuple[DeferredCoverage, ...] = ()
    expression: TimeExpression | None = None

    def __post_init__(self) -> None:
        if self.expression is None:
            value = (
                LinearTime(self.minimum)
                if self.minimum == self.maximum
                else LinearTime(
                    terms=(
                        (CycleVariable(self.register, self.minimum, self.maximum), 1),
                    )
                )
            )
            object.__setattr__(self, "expression", value)


def runtime_expression(value: RuntimeEnd) -> TimeExpression:
    """Return the immutable value identity, independent of mutable register storage."""
    if value.expression is None:
        raise RuntimeError("runtime end has no compile-time cycle expression")
    return value.expression


def projected_bounds(prog, value: "Time") -> tuple[float, float]:
    """Project correlated cycle bounds only at a native timestamp boundary."""
    if isinstance(value, RuntimeEnd):
        expression = runtime_expression(value)
        lower, upper = projected_expression_bounds(expression)
        return lower / get_fclk(prog), upper / get_fclk(prog)
    return bounds(value)


def parameter_expression(
    prog, value: float | QickParam, *, cycles: bool = False
) -> TimeExpression:
    """Describe an already-rounded native parameter without emitting instructions.

    Inputs normally use us; cycles=True describes raw register initialization.
    Shared loop names use the same compiler-owned variable identity.
    """
    factor = 1 if cycles else get_fclk(prog)
    if not isinstance(value, QickParam):
        return LinearTime(round(value * factor))
    axes = prog._timeline_book.cycle_axes
    terms: list[tuple[CycleVariable, int]] = []
    for name, span in value.spans.items():
        count = prog.loop_dict[name]
        if count > 1:
            if name not in axes:
                axes[name] = CycleVariable(name, 0, count - 1, ((name, count),))
            terms.append((axes[name], round(span * factor / (count - 1))))
    return LinearTime(round(value.start * factor), tuple(terms))


def time_expression(prog, value: "Time") -> TimeExpression:
    """Describe a relative end in integer cycles without register allocation."""
    return (
        runtime_expression(value)
        if isinstance(value, RuntimeEnd)
        else parameter_expression(prog, value)
    )


def source_expression(prog, name: str) -> TimeExpression:
    """Return a register's current point provenance, or its certified unknown range."""
    book = prog._timeline_book
    if name in book.source_expressions:
        return book.source_expressions[name]
    if name in prog.loop_dict:
        count = prog.loop_dict[name]
        if name not in book.cycle_axes:
            book.cycle_axes[name] = CycleVariable(name, 0, count - 1, ((name, count),))
        return LinearTime(terms=((book.cycle_axes[name], 1),))
    register = prog.reg_dict.get(name)
    if register is not None and register.init is not None:
        raw = register.init
        return parameter_expression(prog, QickParam(raw.start, raw.spans), cycles=True)
    limits = prog._time_bounds.describe(prog, name)
    if limits.minimum == limits.maximum:
        return LinearTime(limits.minimum)
    return LinearTime(terms=((CycleVariable(name, limits.minimum, limits.maximum), 1),))


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
        return source_expression(prog, name)
    return (
        LinearTime(minimum)
        if minimum == maximum
        else LinearTime(terms=((CycleVariable(name, minimum, maximum), 1),))
    )


def lookup_expression(
    prog, name: str, values: tuple[int, ...], selector: str
) -> TimeExpression:
    """Preserve one lookup's legal points using the selector at this occurrence."""
    index = source_expression(prog, selector)
    axes = expression_axes(index)
    points = evaluate_time(index, axes)
    if points is None:
        return LinearTime(terms=((CycleVariable(name, min(values), max(values)), 1),))
    selected = np.asarray(values, dtype=np.int64)[points]
    lower, upper = int(selected.min()), int(selected.max())
    if lower == upper:
        return LinearTime(lower)
    variable = CycleVariable(
        name, lower, upper, tuple(axes.items()), tuple(int(v) for v in selected.flat)
    )
    return LinearTime(terms=((variable, 1),))


Time = float | QickParam | RuntimeEnd
LIMIT = 2**31 - 1


@dataclass(frozen=True)
class EventEnd:
    """Unrounded output end as an origin plus a fractional offset in us.

    origin is the timeline placement point, possibly an integer-cycle register.
    offset includes resolved event start offset and waveform/ADC duration;
    a runtime offset is already an integer-cycle value.
    Keeping it separate defers ceil until the owner adds its post-delay.
    rounded contains ceiled event boundaries used only for same-channel
    no-overlap proof after a nonnegative post; it never supplies aggregate time.
    """

    origin: Time
    offset: Time
    # The ceiled event boundary, retained only for same-channel no-overlap proof.
    rounded: tuple["Time", ...] = ()


@dataclass(frozen=True)
class SelectedEventEnd:
    """Raw selected output offsets and paired posts, relative to origin (us).

    selector is a certified zero-based register. offsets and posts have equal
    nonzero lengths in selector order. rounded carries physical channel ends
    solely for coverage proof; it is never the aggregate timing source.
    """

    origin: Time
    selector: str
    offsets: tuple[float | QickParam, ...]
    posts: tuple[float | QickParam, ...]
    rounded: tuple[Time, ...] = ()


def time_bounds(prog, value: Time) -> tuple[float, float]:
    if isinstance(value, RuntimeEnd):
        return value.minimum / get_fclk(prog), value.maximum / get_fclk(prog)
    return bounds(value)


def _known_maximum(a: Time, b: Time) -> Time | None:
    if a is b or (isinstance(a, RuntimeEnd) and any(item is b for item in a.covers)):
        return a
    if isinstance(b, RuntimeEnd) and any(item is a for item in b.covers):
        return b
    if not isinstance(a, RuntimeEnd) and not isinstance(b, RuntimeEnd):
        lower, upper = bounds(a - b)
        if lower >= -1e-12:
            return a
        if upper <= 1e-12:
            return b
    return None


class RuntimeMath(TimedMacro):
    """Prepare flag-free cycle arithmetic; translate at its owning macro position."""

    def __init__(self, prog):
        super().__init__()
        self.prog = prog
        self.operations: list[Macro] = []
        self.floors: dict[tuple[int, int], tuple[Time, Time, Time]] = {}
        self.claimed = set(prog._timeline_live_registers())
        previous = prog._timeline_book.math if prog._timeline_book.preparing else None
        if previous is not None:
            self.claimed.update(previous.claimed)

    def source(self, name: str) -> RuntimeEnd:
        """Describe a certified register at this consumer, retaining known points."""
        limits = self.prog._time_bounds.source(self.prog, name)
        return RuntimeEnd(
            name,
            limits.minimum,
            limits.maximum,
            expression=source_expression(self.prog, name),
        )

    def lookup(self, values: list[int], axes: dict[str, int]) -> RuntimeEnd:
        """Materialize exact nonnegative cycle points over native sweep axes.

        values is row-major in axes order, after owner aggregate rounding.
        The table/index owner supplies native sweep initialization and lookup;
        its dependencies participate in the same whole-stream timing audit.
        """
        lower, upper = min(values), max(values)
        self.check(lower, upper)
        if lower < 0:
            raise ValueError("timeline end must be nonnegative cycles")
        owner = self.prog._timing_table_owner
        offset = self.prog.add_timing_dmem(values, owner=owner)
        register = self.allocate()
        index = sweep_table_index(
            self.prog, axes, name=f"_timeline_index_{len(self.prog.reg_dict)}"
        )
        operation = DmemLookup(
            dst=register, index=index, offset=offset, address=self.scratch(3)
        )
        proof = self.prog._time_bounds
        # Native loop counters control how often the swept index advances;
        # their whole-stream write provenance is part of the bounds proof.
        proof.register(index, CycleBounds(0, len(values) - 1), (), tuple(axes))
        limits = {index: CycleBounds(0, len(values) - 1)}
        proof.register(register, CycleBounds(lower, upper), (operation,), (), limits)
        proof.capture_inputs(self.prog, register, (), limits)
        proof.defined.add(register)
        proof.watch_signed(self.prog, register)
        self.operations.append(operation)
        expression = LinearTime(
            terms=(
                (
                    CycleVariable(
                        register, lower, upper, tuple(axes.items()), tuple(values)
                    ),
                    1,
                ),
            )
        )
        return RuntimeEnd(register, lower, upper, expression=expression)

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
        self, selector: str, values: tuple[float | QickParam, ...]
    ) -> RuntimeEnd:
        """Ceil raw us offsets once, selected by a certified arbitrary register.

        Native axes inside each candidate remain native axes; selector is a
        separate table dimension. Bounds and initialization are checked at this
        execution position and every selector/dependency write is audited.
        Offsets may be signed32; the aggregate owner checks the final end.
        """
        axes = self._selected_axes(values)
        count = math.prod(axes.values())
        points = self._selected_points(values, axes)
        lower, upper = min(points), max(points)
        self.check(lower, upper)
        proof = self.prog._time_bounds
        limits = {selector: CycleBounds(0, len(values) - 1)}
        index = self.scratch(0)
        # Index arithmetic is compiler-owned. It does not certify the author
        # selector as a native sweep or relax the selector's input proof.
        native = None
        if axes:
            native = sweep_table_index(
                self.prog, axes, name=f"_selected_grid_{len(self.prog.reg_dict)}"
            )
            proof.register(native, CycleBounds(0, count - 1), (), tuple(axes))
            limits[native] = CycleBounds(0, count - 1)
        self.check(0, len(values) * count - 1)
        self.operations.append(WriteReg(dst=index, src=selector))
        for bit in bin(count)[3:]:
            self.operations.append(WriteRegOp(dst=index, lhs=index, op="+", rhs=index))
            if bit == "1":
                self.operations.append(
                    WriteRegOp(dst=index, lhs=index, op="+", rhs=selector)
                )
        if native is not None:
            self.operations.append(WriteRegOp(dst=index, lhs=index, op="+", rhs=native))
        offset = self.prog.add_timing_dmem(points, owner=self.prog._timing_table_owner)
        register = self.allocate()
        operation = DmemLookup(
            dst=register, index=index, offset=offset, address=self.scratch(1)
        )
        proof.register(register, CycleBounds(lower, upper), (operation,), (), limits)
        proof.capture_inputs(self.prog, register, (), limits)
        proof.defined.add(register)
        proof.watch_signed(self.prog, register)
        self.operations.append(operation)
        return RuntimeEnd(register, lower, upper)

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
        lo, hi = time_bounds(self.prog, end)
        lower, upper = round(lo * get_fclk(self.prog)), round(hi * get_fclk(self.prog))
        self.check(lower, upper)
        if lower < 0:
            raise ValueError("timeline end must be nonnegative cycles")

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
        limits: tuple[int, int],
        target: str | None = None,
    ) -> RuntimeEnd:
        self.check(*limits)
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
                else LinearTime(right)
            )
            expression = add_time(
                runtime_expression(left),
                rhs if operator == "+" else scale_time(rhs, -1),
            )
        if operator == "+":
            right_minimum = right.minimum if isinstance(right, RuntimeEnd) else right
            if right_minimum >= 0:
                covers = (left, *left.covers)
                for proof in left.deferred:
                    remaining = max(0, proof.cycles - right_minimum)
                    if remaining:
                        deferred.append(DeferredCoverage(proof.boundary, remaining))
                    else:
                        covers += (proof.boundary,)
        return RuntimeEnd(register, *limits, covers, tuple(deferred), expression)

    @staticmethod
    def covers(end: Time, previous: Time) -> bool:
        """Return whether an existing exact identity/coverage proves end >= previous."""
        return _known_maximum(end, previous) is end

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
                limits=(left.minimum + amount, left.maximum + amount),
                target=target,
            )
        right = self.cycles(b)
        if right.minimum == right.maximum == 0:
            return left
        return self.op(
            left,
            "+",
            right,
            limits=(left.minimum + right.minimum, left.maximum + right.maximum),
            target=target,
        )

    def subtract(self, a: Time, b: Time) -> RuntimeEnd:
        if isinstance(b, QickParam) and not b.is_sweep():
            b = b.start
        left = self.cycles(a)
        if isinstance(b, (float, int)):
            amount = round(b * get_fclk(self.prog))
            return self.op(
                left, "-", amount, limits=(left.minimum - amount, left.maximum - amount)
            )
        right = self.cycles(b)
        return self.op(
            left,
            "-",
            right,
            limits=(left.minimum - right.maximum, left.maximum - right.minimum),
        )

    def maximum(self, a: Time, b: Time, *, target: str | None = None) -> Time:
        known = _known_maximum(a, b)
        if known is not None:
            return known
        left, right = self.cycles(a), self.cycles(b)
        if left.minimum >= right.maximum:
            return a
        if right.minimum >= left.maximum:
            return b
        destination = self.allocate() if target is None else target
        # A fresh result can hold intermediates. An explicitly aliased target
        # must preserve both input operands until the final addition.
        work = (
            destination
            if destination not in (left.register, right.register)
            else self.scratch(2)
        )
        delta = self.op(
            left,
            "-",
            right,
            limits=(left.minimum - right.maximum, left.maximum - right.minimum),
            target=self.scratch(0),
        )
        # Split before adding: (x + ABS(x))/2 can overflow even when max(0,x) fits.
        if delta.minimum == -(2**31):
            raise ValueError("timeline ABS may overflow signed 32-bit cycles")
        absolute = self.scratch(1)
        self.operations.append(
            AsmInst(
                inst={
                    "CMD": "REG_WR",
                    "DST": self.prog._get_reg(absolute),
                    "SRC": "op",
                    "OP": f"ABS {self.prog._get_reg(delta.register)}",
                },
                addr_inc=1,
            )
        )
        half = self.op(
            delta,
            "ASR",
            1,
            limits=(delta.minimum // 2, delta.maximum // 2),
            target=work,
        )
        abs_half = self.op(
            RuntimeEnd(absolute, 0, max(abs(delta.minimum), abs(delta.maximum))),
            "ASR",
            1,
            limits=(0, max(abs(delta.minimum), abs(delta.maximum)) // 2),
            target=self.scratch(1),
        )
        remainder = self.op(
            delta,
            "-",
            half,
            limits=((delta.minimum + 1) // 2, (delta.maximum + 1) // 2),
            target=self.scratch(0),
        )
        parity = self.op(remainder, "-", half, limits=(0, 1), target=self.scratch(0))
        total = self.op(
            half,
            "+",
            abs_half,
            limits=(min(0, delta.minimum), max(0, delta.maximum)),
            target=work,
        )
        positive = self.op(
            total,
            "+",
            parity,
            limits=(max(0, delta.minimum), max(0, delta.maximum)),
            target=work,
        )
        result = self.op(
            right,
            "+",
            positive,
            limits=(max(left.minimum, right.minimum), max(left.maximum, right.maximum)),
            target=destination,
        )
        covers = (
            a,
            b,
            *(a.covers if isinstance(a, RuntimeEnd) else ()),
            *(b.covers if isinstance(b, RuntimeEnd) else ()),
        )
        # max(a,b) is no earlier than either input, so either input's deferred
        # physical boundary remains covered after the same additional debt.
        deferred = (
            *(a.deferred if isinstance(a, RuntimeEnd) else ()),
            *(b.deferred if isinstance(b, RuntimeEnd) else ()),
        )
        return RuntimeEnd(
            result.register,
            result.minimum,
            result.maximum,
            covers,
            deferred,
            choose_time(runtime_expression(left), runtime_expression(right)),
        )

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
        _, hi = time_bounds(self.prog, end)
        a_lo, _ = time_bounds(self.prog, advance)
        if hi <= a_lo + 1e-12:
            return 0.0
        delta = self.subtract(end, advance)
        return self.maximum(0.0, delta, target=delta.register)

    def translate(self, prog) -> None:
        previous = prog._timeline_math_active
        prog._timeline_math_active = True
        try:
            for operation in self.operations:
                operation.translate(prog)
        finally:
            prog._timeline_math_active = previous
