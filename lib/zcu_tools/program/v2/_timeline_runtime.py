"""Cycle-valued expressions for timeline ends that are not affine sweeps."""

import math
from dataclasses import dataclass

import numpy as np
from qick.asm_v2 import AsmInst, Macro, QickParam, TimedMacro, WriteReg

from ._timeline_bounds import CycleBounds
from ._timeline_guard import bounds
from .macro.dmem import DmemLookup, sweep_table_index
from .macro.write_reg import WriteRegOp
from .modules.util import get_fclk


def align_end(prog, value: float | QickParam) -> "Time":
    """Ceil a resolved aggregate in us at every native sweep point.

    Call only at an aggregate or placement boundary during preprocessing.
    Scalars and exact integer-affine sweeps remain static. Other point grids
    become a certified dmem lookup at this operation, without advancing reference.
    """

    # Arithmetic on already integral tick values can land just above an integer.
    # Do not turn that floating-point noise into another scheduled tick.
    def ceil_tick(ticks: float) -> int:
        nearest = round(ticks)
        return (
            nearest
            if math.isclose(ticks, nearest, rel_tol=0, abs_tol=1e-10)
            else math.ceil(ticks)
        )

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
class RuntimeEnd:
    """Relative cycle end stored in register, with inclusive minimum/maximum.

    register is a compiler-owned data-register name. minimum/maximum bound the
    value on every executable path. covers lists ends proven no later than this
    value, allowing a following sync to consume them without new arithmetic.
    """

    register: str
    minimum: int
    maximum: int
    covers: tuple["Time", ...] = ()


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
        proof.source(self.prog, register)
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
            return RuntimeEnd(operand, lower, upper)
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
        if operator == "+":
            right_minimum = right.minimum if isinstance(right, RuntimeEnd) else right
            if right_minimum >= 0:
                covers = (left, *left.covers)
        return RuntimeEnd(register, *limits, covers)

    @staticmethod
    def covers(end: Time, previous: Time) -> bool:
        """Return whether an existing exact identity/coverage proves end >= previous."""
        return _known_maximum(end, previous) is end

    def snapshot(self, value: RuntimeEnd) -> RuntimeEnd:
        register = self.allocate()
        self.operations.append(WriteReg(dst=register, src=value.register))
        return RuntimeEnd(register, value.minimum, value.maximum, value.covers)

    def add(self, a: Time, b: Time) -> RuntimeEnd:
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
                left, "+", amount, limits=(left.minimum + amount, left.maximum + amount)
            )
        right = self.cycles(b)
        if right.minimum == right.maximum == 0:
            return left
        return self.op(
            left,
            "+",
            right,
            limits=(left.minimum + right.minimum, left.maximum + right.maximum),
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
            target=self.scratch(2),
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
            target=self.scratch(2),
        )
        positive = self.op(
            total,
            "+",
            parity,
            limits=(max(0, delta.minimum), max(0, delta.maximum)),
            target=self.scratch(2),
        )
        result = self.op(
            right,
            "+",
            positive,
            limits=(max(left.minimum, right.minimum), max(left.maximum, right.maximum)),
            target=target,
        )
        covers = (
            a,
            b,
            *(a.covers if isinstance(a, RuntimeEnd) else ()),
            *(b.covers if isinstance(b, RuntimeEnd) else ()),
        )
        return RuntimeEnd(result.register, result.minimum, result.maximum, covers)

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
