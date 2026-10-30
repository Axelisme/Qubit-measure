"""Admit parallel lowered sequences without changing native reference semantics."""

from collections.abc import Sequence
from dataclasses import dataclass

from qick.asm_v2 import (
    CloseLoop,
    ConfigReadout,
    Delay,
    IncReg,
    Macro,
    Pulse,
    QickParam,
    QickProgramV2,
    ReadDmem,
    TimedMacro,
    Trigger,
    WriteReg,
)

from ._timeline_macros import (
    FixedRepeatLoop,
    ModuleEmission,
    NativeTimelineDelay,
    PreparedTimelineMacro,
    TimelineAdvance,
    TimelineBranch,
    TimelineFork,
    TimelineJoin,
    TimelineRepeatEnter,
    TimelineRepeatExit,
    TimelineRuntimeAdvance,
    TimelineSync,
    TimeProducer,
)
from ._timeline_outputs import (
    TimelineObservedPulse,
    TimelinePlacedConfig,
    TimelinePlacedPulse,
    TimelinePlacedTrigger,
)
from ._timeline_selected import TimelineSelectedPulse
from ._timeline_state import TimelineBook
from .macro.delay import DelayRegAuto
from .macro.dmem import DmemLookup
from .macro.pluse_reg import PulseByReg
from .macro.readout import ReadoutEmission
from .macro.wmem import (
    ConfigReadoutFromRegs,
    PatchWmemFromDmem,
    PatchWmemFromRegs,
    PulseFromLengthReg,
    PulseFromRegs,
)
from .macro.write_reg import WriteRegOp

# Exact types matter: an unknown subclass may override reference or word effects.
_REFERENCE_ADVANCE = (
    Delay,
    DelayRegAuto,
    NativeTimelineDelay,
    TimelineSync,
    TimelineRepeatEnter,
    TimelineRepeatExit,
    TimelineRuntimeAdvance,
)
_REFERENCE_STABLE = (
    TimelineAdvance,
    TimelineFork,
    TimelineJoin,
    TimelinePlacedPulse,
    TimelineObservedPulse,
    TimelinePlacedConfig,
    TimelinePlacedTrigger,
    TimelineSelectedPulse,
    Pulse,
    ConfigReadout,
    Trigger,
    PulseFromRegs,
    PulseFromLengthReg,
    PulseByReg,
    ConfigReadoutFromRegs,
    PatchWmemFromRegs,
    PatchWmemFromDmem,
    WriteReg,
    WriteRegOp,
    IncReg,
    ReadDmem,
    DmemLookup,
    FixedRepeatLoop,
    CloseLoop,
    ReadoutEmission,
)


def _advances(macro: Macro) -> bool:
    if type(macro) in (
        PreparedTimelineMacro,
        TimelineObservedPulse,
        TimelinePlacedPulse,
        TimelinePlacedConfig,
        NativeTimelineDelay,
        TimelineRuntimeAdvance,
        TimelineSelectedPulse,
    ):
        return _advances(macro.source)
    if type(macro) in (ModuleEmission, TimeProducer):
        return any([_advances(child) for child in macro.macros])
    if type(macro) is TimelineParallel:
        return macro.advancing is not None
    if type(macro) is TimelineBranch:
        # Validate every case even though case closure already implies advance.
        for case in macro.cases:
            for child in case:
                _advances(child)
        return True
    if type(macro) in _REFERENCE_ADVANCE:
        return True
    if type(macro) in _REFERENCE_STABLE and not _unknown_control_write(macro):
        return False
    raise ValueError(
        f"parallel admission: unknown reference effect of {type(macro).__name__}"
    )


def _unknown_control_write(macro: Macro) -> bool:
    if type(macro) is DmemLookup:
        names = (macro.dst, macro.address)
    elif type(macro) is PatchWmemFromDmem:
        names = (macro.addr_reg, macro.val_reg)
    elif type(macro) in (WriteReg, WriteRegOp, IncReg, ReadDmem):
        names = (macro.dst,)
    else:
        names = ()
    # Processor control/peripheral destinations are not ordinary data storage.
    # Only zero and native output-time/jump scratch have bounded semantics here.
    unknown = {f"s{i}" for i in range(1, 14)}
    return any(QickProgramV2.REG_ALIASES.get(name, name) in unknown for name in names)


class TimelineParallel(TimedMacro):
    """Emit whole lowered sequences from one origin, then merge actual ends.

    book owns timeline state. timeline is an existing placement target. branches
    contains fresh lowered operations in author order. names contains one fresh
    timeline name per branch. This macro owns their fork/join bookkeeping.
    One advancing branch moves last; two or more raise ValueError.
    Unknown reference effects always fail admission. Exchanged sequences require
    known, disjoint physical register/word effects. No scheduling is done within
    a sequence; existing physical-end, case and backedge guards remain active.
    Time parameters come from the first child in author order, not emit order.
    """

    def __init__(
        self,
        book: TimelineBook,
        timeline: str,
        branches: Sequence[list[Macro]],
        names: Sequence[str],
    ):
        super().__init__()
        self.book = book
        self.timeline = timeline
        self.branches = list(branches)
        self.timing_source = next(
            (
                macro
                for branch in branches
                for macro in branch
                if isinstance(macro, TimedMacro)
            ),
            None,
        )
        advancing = [
            index
            for index, branch in enumerate(branches)
            if any([_advances(macro) for macro in branch])
        ]
        if len(advancing) > 1:
            raise ValueError(
                f"parallel admission: multiple reference-advancing branches {advancing}"
            )
        self.advancing = advancing[0] if advancing else None
        order = [i for i in range(len(branches)) if i != self.advancing]
        if self.advancing is not None:
            order.append(self.advancing)
        self.macros = [
            PreparedTimelineMacro(book, TimelineFork(book, timeline, tuple(names))),
            *(macro for index in order for macro in branches[index]),
            PreparedTimelineMacro(book, TimelineJoin(book, tuple(names), timeline)),
        ]

    def preprocess(self, prog) -> None:
        self.book.end(self.timeline, placement=True)
        for macro in self.macros:
            macro.preprocess(prog)
        if self.advancing is not None and self.advancing < len(self.branches) - 1:
            left = _resources(prog, self.branches[self.advancing])
            for branch in self.branches[self.advancing + 1 :]:
                right = _resources(prog, branch)
                if left is None or right is None:
                    raise ValueError("parallel exchange: unknown register/word effects")
                conflict = left.conflicts(right)
                if conflict:
                    raise ValueError(
                        f"parallel exchange: register/word conflict {sorted(conflict)}"
                    )

    def translate(self, prog) -> None:
        for macro in self.macros:
            macro.translate(prog)

    def list_time_params(self) -> list[str]:
        """List the first author-order child's parameters, or [] if untimed."""
        return self.timing_source.list_time_params() if self.timing_source else []

    def get_time_param(self, name: str) -> QickParam:
        """Return first-child timing; absent events/parameters raise RuntimeError."""
        if self.timing_source is None:
            raise RuntimeError("this parallel emission has no timed operations")
        return self.timing_source.get_time_param(name)


@dataclass
class _Effects:
    # Physical register / wmem identities; locals are native atomic-operation
    # scratch, overwritten before use and consumed before that operation exits.
    reads: set[str]
    writes: set[str]
    locals: set[str]

    def conflicts(self, other: "_Effects") -> set[str]:
        # Local/local reuse is safe, but an explicit author access (even an
        # alias) prevents the owner from proving occurrence-local lifetime.
        return (
            (self.writes & (other.reads | other.writes | other.locals))
            | (self.reads & (other.writes | other.locals))
            | (self.locals & (other.reads | other.writes))
        )


def _source(macro: Macro) -> Macro:
    return macro.source if isinstance(macro, PreparedTimelineMacro) else macro


def _resources(prog, macros: Sequence[Macro]) -> _Effects | None:
    result = _Effects(set(), set(), set())
    loops = [
        source for macro in macros if type(source := _source(macro)) is FixedRepeatLoop
    ]
    closings = [loop.closing for loop in loops]
    for macro in macros:
        if any(_source(macro) is closing for closing in closings):
            continue  # The paired FixedRepeatLoop owns these counter/sweep effects.
        effects = _resource(prog, macro)
        if effects is None:
            return None
        result.reads.update(effects.reads)
        result.writes.update(effects.writes)
        result.locals.update(effects.locals)
    # Volatile/peripheral reads cannot be commuted merely because neither
    # branch explicitly writes their registers (for example random or time).
    if result.reads & {f"s{i}" for i in range(1, 14)}:
        return None
    return result


def _resource(prog, macro: Macro) -> _Effects | None:
    if isinstance(macro, PreparedTimelineMacro):
        # Generated cycle arithmetic is not assumed to be scratch: its live
        # values can escape into timeline ends. Retain its actual accesses.
        return _resources(prog, [macro.source, *macro.math.operations])
    if type(macro) in (ModuleEmission, TimeProducer, TimelineParallel):
        children = macro.macros
        if isinstance(macro, ModuleEmission):
            children = [*children, *macro.finishers]
        return _resources(prog, children)
    if type(macro) is TimelineBranch:
        result = _resources(
            prog,
            [child for case in macro.prepared for child in case]
            + [op for tail in macro.tails for op in tail.operations],
        )
        if result is not None:
            result.reads.add(prog._get_reg(macro.by))
            result.locals.add("s15")  # Native dispatch prepares its own jump address.
        return result
    if type(macro) is FixedRepeatLoop:
        # QICK CloseLoop also patches native swept words/registers. A nested
        # fixed index with such effects needs its native owner's summary.
        if any(
            macro.name in sweep.spans for wave in prog.waves for sweep in wave.sweeps()
        ) or any(
            reg.init is not None and macro.name in reg.init.spans
            for reg in prog.reg_dict.values()
        ):
            return None
        return _Effects(set(), {prog._get_reg(macro.name)}, {"s15"})
    return _leaf_resource(prog, macro)


def _patch_resource(prog, macro) -> _Effects:
    words = {
        f"wmem:{prog.wave2idx[name]}"
        for name in prog.pulses[macro.name].get_wavenames()
    }
    result = _Effects(words.copy(), words.copy(), {f"w{i}" for i in range(6)})
    if isinstance(macro, PatchWmemFromDmem):
        result.reads.add(prog._get_reg(macro.idx_reg))
        result.writes.update(
            prog._get_reg(name) for name in (macro.addr_reg, macro.val_reg)
        )
    else:
        result.reads.update(
            prog._get_reg(name)
            for name in (macro.freq_reg, macro.gain_reg)
            if name is not None
        )
    return result


def _leaf_resource(prog, macro: Macro) -> _Effects | None:
    if type(macro) in (WriteReg, WriteRegOp, IncReg, ReadDmem, DmemLookup):
        return _register_resource(prog, macro)
    if type(macro) in (
        TimelinePlacedPulse,
        TimelineObservedPulse,
        TimelinePlacedConfig,
    ):
        return _resource(prog, macro.source)
    if type(macro) in (
        TimelineFork,
        TimelineJoin,
        TimelineAdvance,
        TimelineSync,
        TimelineRepeatEnter,
        TimelineRepeatExit,
        Delay,
    ):
        return _timeline_resource(prog, macro)
    if type(macro) is NativeTimelineDelay:
        return _timed_resource(prog, macro.source)
    if type(macro) in (
        Pulse,
        ConfigReadout,
        PulseFromRegs,
        ConfigReadoutFromRegs,
        PatchWmemFromRegs,
        PatchWmemFromDmem,
    ):
        return _wave_resource(prog, macro)
    return _Effects(set(), set(), set()) if type(macro) is ReadoutEmission else None


def _timeline_resource(prog, macro) -> _Effects:
    if type(macro) in (TimelineFork, TimelineJoin):
        return _Effects(set(), set(), set())
    if type(macro) is TimelineAdvance:
        return _Effects(
            {prog._get_reg(macro.duration)}
            if isinstance(macro.duration, str)
            else set(),
            set(),
            set(),
        )
    return _timed_resource(prog, macro)


def _register_resource(prog, macro) -> _Effects | None:
    if type(macro) is DmemLookup:
        return _Effects(
            {prog._get_reg(macro.index)},
            {prog._get_reg(macro.dst), prog._get_reg(macro.address)},
            set(),
        )
    reads: set[str] = set()
    if type(macro) is WriteRegOp:
        reads.add(prog._get_reg(macro.lhs))
        operand = macro.rhs
    elif type(macro) is ReadDmem:
        operand = macro.addr
    else:
        operand = macro.src
        if type(macro) is IncReg:
            reads.add(prog._get_reg(macro.dst))
    if isinstance(operand, str):
        reads.add(prog._get_reg(operand))
    # Native IncReg may prepare the reusable 'scratch' register for a large
    # literal. Keep it conservative rather than assuming that name is local.
    writes = {prog._get_reg(macro.dst)}
    if (
        type(macro) is IncReg
        and isinstance(operand, int)
        and not -(2**23) <= operand < 2**23
    ):
        # QICK allocates this reusable scratch only during translation. Its
        # physical identity/lifetime is not yet available to this owner.
        return None
    return _Effects(reads, writes, set())


def _timed_resource(prog, macro) -> _Effects:
    # Native time parameter registers are read, never treated as local merely
    # because the compiler allocated them.
    registers = {
        prog._get_reg(value)
        for value in macro.t_regs.values()
        if isinstance(value, str)
    }
    return _Effects(registers, set(), {"s14"} if registers else set())


def _wave_resource(prog, macro) -> _Effects:
    if type(macro) in (PatchWmemFromRegs, PatchWmemFromDmem):
        return _patch_resource(prog, macro)
    result = _timed_resource(prog, macro)
    result.reads.update(
        f"wmem:{prog.wave2idx[name]}"
        for name in prog.pulses[macro.name].get_wavenames()
    )
    if isinstance(macro, (PulseFromRegs, ConfigReadoutFromRegs)):
        result.reads.update(
            prog._get_reg(name)
            for name in (macro.freq_reg, macro.gain_reg)
            if name is not None
        )
    # Native playback loads a complete wave before consuming it at WPORT.
    # Directly patched runtime fields do not persist into wmem.
    result.locals.update(f"w{index}" for index in range(6))
    return result
