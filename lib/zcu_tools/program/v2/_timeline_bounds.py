"""Producer-owned cycle bounds and emission-time write provenance."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

from qick.asm_v2 import CloseLoop, Macro, OpenLoop, QickParam, TimedMacro, WriteReg

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


class ProducerBounds:
    """Track known producers and reject unregistered writes to timing dependencies.

    The program observes native constant writes and owner-certified operations
    before preprocessing. Translation supplies the originating macro for each
    instruction. This is provenance checking, not interval interpretation of ALU.
    """

    def __init__(self):
        self.ranges: dict[str, list[CycleBounds]] = {}
        self.dependencies: dict[str, set[str]] = {}
        self.requirements: dict[str, list[tuple[str, CycleBounds]]] = {}
        self.allowed: dict[int, set[str]] = {}
        self.sources: set[str] = set()
        self.defined: set[str] = set()
        self.origins: dict[int, Macro | None] = {}
        self.generated: dict[str, set[int]] = {}
        self.sweep_times: set[str] = set()

    def observe(self, macro: Macro) -> None:
        """Register the bounds and write identity of a native constant producer."""
        # Subclasses can replace translate/expand, so their src is not a proof.
        if type(macro) is WriteReg and isinstance(macro.src, Integral):
            value = int(macro.src)
            self.register(macro.dst, CycleBounds(value, value), (macro,), ())

    def register(
        self,
        name: str,
        limits: CycleBounds,
        writers: tuple[Macro, ...],
        dependencies: tuple[str, ...],
        dependency_ranges: Mapping[str, CycleBounds] | None = None,
    ) -> None:
        """Register a framework owner's value bounds, exact writers and inputs.

        Writers may write other registers, but certify only name. Dependencies
        must themselves have known producers. The owner guarantees these bounds
        across its entire operation, including lookup/word conversion steps.
        """
        self.ranges.setdefault(name, []).append(limits)
        self.dependencies.setdefault(name, set()).update(dependencies)
        if dependency_ranges is not None:
            self.dependencies[name].update(dependency_ranges)
            self.requirements.setdefault(name, []).extend(dependency_ranges.items())
        for writer in writers:
            self.allowed.setdefault(id(writer), set()).add(name)

    def activate(self, macro: Macro) -> None:
        """Mark values written by an observed producer as initialized on this path."""
        self.defined.update(self.allowed.get(id(macro), ()))

    def source(self, prog, name: str, *, check_defined: bool = True) -> CycleBounds:
        """Resolve a nonnegative cycle source and all of its producer dependencies.

        Unknown registers, uninitialized paths and unproved/out-of-range values
        raise ValueError. Loop counters and QICK sweep registers are initialized
        by their native loop machinery, including back-edges and resets.
        """
        result = self._resolve(prog, name, check_defined=check_defined)
        pending = [name]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            self._resolve(prog, current, check_defined=check_defined)
            self.sources.add(current)
            pending.extend(self.dependencies.get(current, ()))
            for dependency, allowed in self.requirements.get(current, ()):
                actual = self._resolve(prog, dependency, check_defined=check_defined)
                if actual.minimum < allowed.minimum or actual.maximum > allowed.maximum:
                    raise ValueError(
                        f"{current}: index dependency {dependency} allowed "
                        f"[{allowed.minimum}, {allowed.maximum}], actual "
                        f"[{actual.minimum}, {actual.maximum}]"
                    )
        return result

    def _resolve(self, prog, name: str, *, check_defined: bool) -> CycleBounds:
        try:
            prog._get_reg(name)
        except KeyError as exc:
            raise ValueError(f"{name}: unknown time register") from exc
        register = prog.reg_dict.get(name)
        if name in prog.loop_dict:
            result = CycleBounds(0, prog.loop_dict[name] - 1)
        elif register is not None and register.init is not None:
            raw = register.init
            lo, hi = bounds(QickParam(raw.start, raw.spans))
            result = CycleBounds(round(lo), round(hi))
        else:
            ranges = self.ranges.get(name)
            if not ranges or (check_defined and name not in self.defined):
                raise ValueError(
                    f"{name}: unknown or uninitialized time producer bounds"
                )
            result = CycleBounds(
                min(r.minimum for r in ranges), max(r.maximum for r in ranges)
            )
        if result.minimum < 0 or result.maximum > 2**31 - 1:
            raise ValueError(
                f"{name}: time producer exceeds nonnegative signed 32-bit cycles"
            )
        return result

    def emission(
        self, raw, address: int, origin: Macro | None, *, arithmetic: bool
    ) -> None:
        """Record compiler arithmetic writers independently of author producers."""
        self.origins[address] = origin
        if arithmetic:
            for register in BaseInst.from_dict(raw).reg_write:
                self.generated.setdefault(register, set()).add(address)

    def time_parameters(self, macro: TimedMacro) -> None:
        """Watch registers allocated by native time conversion, not arbitrary data."""
        self.sweep_times.update(
            name for name in macro.t_regs.values() if isinstance(name, str)
        )

    def _audit_sweeps(self, prog) -> None:
        for name, register in prog.reg_dict.items():
            raw = register.init
            if raw is None or name not in self.sweep_times | self.sources:
                continue
            lo = raw.start + sum(min(0, span) for span in raw.spans.values())
            hi = raw.start + sum(max(0, span) for span in raw.spans.values())
            increments = [step["step"] for step in raw.steps.values()]
            if (
                lo + min([0, *increments]) < -(2**31)
                or hi + max([0, *increments]) > 2**31 - 1
            ):
                raise ValueError(
                    f"{name}: signed 32-bit sweep backedge/reset intermediate overflow"
                )

    def audit(self, prog) -> None:
        """Audit every translated write, including unreachable alternate paths.

        Report the logical register and original instruction position when an
        unregistered writer invalidates a time source or its dependencies.
        """
        for name in tuple(self.sources):
            self.source(prog, name, check_defined=False)
        self._audit_sweeps(prog)
        watched = {
            register.full_addr(): name
            for name, register in prog.reg_dict.items()
            if register.init is not None and name in self.sweep_times | self.sources
        }
        watched.update({prog._get_reg(name): name for name in self.sources})
        watched.update({name: name for name in self.generated})
        for raw in prog.prog_list:
            inst = BaseInst.from_dict(raw)
            address = raw["P_ADDR"]
            origin = self.origins.get(address)
            for physical in inst.reg_write & watched.keys():
                name = watched[physical]
                if physical in self.generated:
                    if address in self.generated[physical]:
                        continue
                    raise ValueError(
                        f"{name}: unregistered compiler time write at instruction {address}"
                    )
                if name in self.allowed.get(id(origin), set()):
                    continue
                register = prog.reg_dict.get(name)
                native_sweep = register is not None and register.init is not None
                native_loop = name in prog.loop_dict
                if (native_sweep or native_loop) and (
                    origin is None or type(origin) in (OpenLoop, CloseLoop)
                ):
                    continue
                raise ValueError(
                    f"{name}: unregistered time producer write at instruction {address}"
                )
