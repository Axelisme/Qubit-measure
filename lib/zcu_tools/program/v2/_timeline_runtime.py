"""Cycle-valued expressions for timeline ends that are not affine sweeps."""

from dataclasses import dataclass

from qick.asm_v2 import AsmInst, Macro, QickParam, TimedMacro, WriteReg

from ._timeline_guard import bounds
from .macro.write_reg import WriteRegOp
from .modules.util import get_fclk


@dataclass(frozen=True)
class _Cycles:
    register: str
    minimum: int
    maximum: int


Time = float | QickParam | _Cycles
LIMIT = 2**31 - 1


def time_bounds(prog, value: Time) -> tuple[float, float]:
    if isinstance(value, _Cycles):
        return value.minimum / get_fclk(prog), value.maximum / get_fclk(prog)
    return bounds(value)


class _RuntimeMath(TimedMacro):
    """Prepare flag-free cycle arithmetic; translate at its owning macro position."""

    def __init__(self, prog):
        super().__init__()
        self.prog = prog
        self.operations: list[Macro] = []
        self.floors: dict[tuple[int, int], Time] = {}

    def cycles(self, value: Time) -> _Cycles:
        if isinstance(value, _Cycles):
            return value
        key = str(len(self.t_regs))
        rounded = self.convert_time(self.prog, value, key)
        lo, hi = bounds(rounded)
        lower, upper = round(lo * get_fclk(self.prog)), round(hi * get_fclk(self.prog))
        self.check(lower, upper)
        operand = self.t_regs[key]
        if isinstance(operand, str):
            return _Cycles(operand, lower, upper)
        if operand == 0:
            return _Cycles("s0", 0, 0)
        register = self.prog.add_reg()
        self.operations.append(WriteReg(dst=register, src=operand))
        return _Cycles(register, lower, upper)

    @staticmethod
    def check(lower: int, upper: int) -> None:
        if lower < -LIMIT or upper > LIMIT:
            raise ValueError("timeline arithmetic may overflow signed 32-bit cycles")

    def scratch(self, index: int) -> str:
        return self.prog.add_reg(f"_timeline_tmp_{index}", allow_reuse=True)

    def op(self, left: _Cycles, operator: str, right: _Cycles | int, *, limits: tuple[int, int], target: str | None = None) -> _Cycles:
        self.check(*limits)
        register = self.prog.add_reg() if target is None else target
        self.operations.append(WriteRegOp(dst=register, lhs=left.register, op=operator,
                                         rhs=right.register if isinstance(right, _Cycles) else right))
        return _Cycles(register, *limits)

    def add(self, a: Time, b: Time) -> _Cycles:
        left = self.cycles(a)
        if isinstance(b, (float, int)):
            amount = round(b * get_fclk(self.prog))
            if amount == 0:
                return left
            return self.op(left, "+", amount,
                           limits=(left.minimum + amount, left.maximum + amount))
        right = self.cycles(b)
        if right.minimum == right.maximum == 0:
            return left
        return self.op(left, "+", right, limits=(left.minimum + right.minimum, left.maximum + right.maximum))

    def subtract(self, a: Time, b: Time) -> _Cycles:
        left, right = self.cycles(a), self.cycles(b)
        return self.op(left, "-", right, limits=(left.minimum - right.maximum, left.maximum - right.minimum))

    def maximum(self, a: Time, b: Time) -> Time:
        if a is b:
            return a
        if not isinstance(a, _Cycles) and not isinstance(b, _Cycles):
            lower, upper = bounds(a - b)
            if lower >= -1e-12:
                return a
            if upper <= 1e-12:
                return b
        left, right = self.cycles(a), self.cycles(b)
        if left.minimum >= right.maximum:
            return a
        if right.minimum >= left.maximum:
            return b
        delta = self.op(left, "-", right,
                        limits=(left.minimum-right.maximum, left.maximum-right.minimum),
                        target=self.scratch(0))
        # Split before adding: (x + ABS(x))/2 can overflow even when max(0,x) fits.
        absolute = self.scratch(1)
        self.operations.append(AsmInst(inst={
            "CMD": "REG_WR", "DST": self.prog._get_reg(absolute), "SRC": "op",
            "OP": f"ABS {self.prog._get_reg(delta.register)}",
        }, addr_inc=1))
        half = self.op(delta, "ASR", 1, limits=(delta.minimum // 2, delta.maximum // 2), target=self.scratch(2))
        abs_half = self.op(_Cycles(absolute, 0, max(abs(delta.minimum), abs(delta.maximum))),
                           "ASR", 1, limits=(0, max(abs(delta.minimum), abs(delta.maximum)) // 2), target=self.scratch(1))
        twice = self.op(half, "+", half, limits=(2 * half.minimum, 2 * half.maximum), target=self.scratch(3))
        parity = self.op(delta, "-", twice, limits=(0, 1), target=self.scratch(0))
        total = self.op(half, "+", abs_half, limits=(min(0, delta.minimum), max(0, delta.maximum)), target=self.scratch(2))
        positive = self.op(total, "+", parity, limits=(max(0, delta.minimum), max(0, delta.maximum)), target=self.scratch(2))
        return self.op(right, "+", positive, limits=(max(left.minimum, right.minimum), max(left.maximum, right.maximum)))

    def floor(self, end: Time, advance: Time) -> Time:
        key = (id(end), id(advance))
        if key not in self.floors:
            self.floors[key] = self._floor(end, advance)
        return self.floors[key]

    def _floor(self, end: Time, advance: Time) -> Time:
        if end is advance:
            return 0.0
        if not isinstance(end, _Cycles) and not isinstance(advance, _Cycles):
            return self.maximum(0.0, end - advance)
        lo, hi = time_bounds(self.prog, end)
        a_lo, a_hi = time_bounds(self.prog, advance)
        if hi <= a_lo + 1e-12:
            return 0.0
        return self.maximum(0.0, self.subtract(end, advance))

    def translate(self, prog) -> None:
        for operation in self.operations:
            operation.translate(prog)
