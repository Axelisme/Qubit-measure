"""Producer-owned cycle bounds and emission-time write provenance."""

from dataclasses import dataclass
from numbers import Integral

from qick.asm_v2 import CloseLoop, Macro, QickParam, WriteReg

from ._timeline_guard import bounds
from .ir.instructions import BaseInst


@dataclass(frozen=True)
class CycleBounds:
    """Inclusive range of a producer's integer tProc cycle value.

    minimum and maximum bound every executable write, across cases, loop
    back-edges and sweep resets. Values used as time must be nonnegative and
    no larger than 2147483647. Non-time words need not satisfy that restriction.
    """

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("cycle bounds must be ordered")


class _ProducerBounds:
    def __init__(self):
        self.constants: dict[str, list[int]] = {}
        self.allowed: dict[int, set[str]] = {}
        self.sources: set[str] = set()
        self.origins: dict[int, Macro | None] = {}

    def observe(self, macro: Macro) -> None:
        if isinstance(macro, WriteReg) and isinstance(macro.src, Integral):
            self.constants.setdefault(macro.dst, []).append(int(macro.src))
            self.allowed.setdefault(id(macro), set()).add(macro.dst)

    def source(self, prog, name: str) -> CycleBounds:
        # Resolve now; an undeclared register must never silently mean zero.
        prog._get_reg(name)
        self.sources.add(name)
        constants = self.constants.get(name)
        if constants:
            result = CycleBounds(min(constants), max(constants))
        else:
            register = prog.reg_dict.get(name)
            if register is None or register.init is None:
                raise ValueError(f"{name}: unknown time producer bounds")
            raw = register.init
            lo, hi = bounds(QickParam(raw.start, raw.spans))
            result = CycleBounds(round(lo), round(hi))
        if result.minimum < 0 or result.maximum > 2**31 - 1:
            raise ValueError(f"{name}: time producer exceeds nonnegative signed 32-bit cycles")
        return result

    def audit(self, prog) -> None:
        watched = {prog._get_reg(name): name for name in self.sources}
        for raw in prog.prog_list:
            inst = BaseInst.from_dict(raw)
            address = raw["P_ADDR"]
            origin = self.origins.get(address)
            for physical in inst.reg_write & watched.keys():
                name = watched[physical]
                if name in self.allowed.get(id(origin), set()):
                    continue
                register = prog.reg_dict.get(name)
                # QICK initializes, steps and resets its own swept registers.
                if register is not None and register.init is not None and (
                    origin is None or isinstance(origin, CloseLoop)
                ):
                    continue
                raise ValueError(f"{name}: unregistered time producer write at instruction {address}")
