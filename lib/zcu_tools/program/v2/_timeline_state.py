"""Relative timeline state, native channel tails and repeat scope bookkeeping."""

from dataclasses import dataclass, field

from qick.asm_v2 import QickParam, WriteReg

from ._timeline_expression import CycleVariable
from ._timeline_guard import PulseGuard as _PulseGuard
from ._timeline_guard import bounds
from ._timeline_runtime import (
    RuntimeEnd,
    RuntimeMath,
    Time,
    projected_bounds,
    time_bounds,
)


@dataclass(frozen=True)
class RepeatFrameSnapshot:
    """Repeat path effects saved while Branch prepares each case.

    advance is reference movement in us; variable marks path-dependent movement.
    written_gens and written_ros contain channel IDs touched on the path.
    pulse_guards holds independent per-generator playback guards for that path.
    """

    advance: float | QickParam
    variable: bool
    written_gens: set[int]
    written_ros: set[int]
    pulse_guards: dict[int, _PulseGuard]


@dataclass
class RepeatFrame:
    parent: str
    external: dict[str, Time]
    waiting: set[str]
    native_gens: dict[int, float | QickParam]
    native_ros: dict[int, float | QickParam]
    advance: float | QickParam = 0.0
    variable: bool = False
    written_gens: set[int] = field(default_factory=set)
    written_ros: set[int] = field(default_factory=set)
    pulse_guards: dict[int, _PulseGuard] = field(default_factory=dict)
    defined: set[str] = field(default_factory=set)
    initial_external: dict[str, Time] = field(default_factory=dict)
    entry_math: RuntimeMath | None = None
    debits: list[tuple[RuntimeMath, Time]] = field(default_factory=list)
    tail_gens: dict[int, RuntimeEnd] = field(default_factory=dict)
    tail_ros: dict[int, RuntimeEnd] = field(default_factory=dict)
    initial_runtime_gens: dict[int, RuntimeEnd] = field(default_factory=dict)
    initial_runtime_ros: dict[int, RuntimeEnd] = field(default_factory=dict)

    def promote(
        self, prog, generated_gens: dict[int, Time], generated_ros: dict[int, Time]
    ) -> None:
        """Materialize unknown total advances at entry and debit their exact paths.

        Preprocessing records reference transfers at their owning macro plans.
        Only an unknown count/path requests these entry stores and transfers.
        Native tails share a store with a matching named end when possible.
        """
        if self.entry_math is None:
            raise RuntimeError("repeat entry has no arithmetic plan")
        for name, end in self.external.items():
            if time_bounds(prog, end)[1] > 1e-12:
                self.external[name] = self._store(prog, end)
        self.tail_gens = self._native_stores(
            prog, self.native_gens, self.initial_runtime_gens, self.written_gens
        )
        self.tail_ros = self._native_stores(
            prog, self.native_ros, self.initial_runtime_ros, self.written_ros
        )
        stores = {
            end.register: end
            for end in [
                *self.external.values(),
                *self.tail_gens.values(),
                *self.tail_ros.values(),
            ]
            if isinstance(end, RuntimeEnd)
        }
        for math, advance in self.debits:
            for end in stores.values():
                remaining = math.cycles(math.floor(end, advance))
                math.operations.append(
                    WriteReg(dst=end.register, src=remaining.register)
                )
        for tails, generated, written in (
            (self.tail_gens, generated_gens, self.written_gens),
            (self.tail_ros, generated_ros, self.written_ros),
        ):
            for ch in written:
                operand = prog._timeline_book.math.cycles(generated[ch])
                prog._timeline_book.math.operations.append(
                    WriteReg(dst=tails[ch].register, src=operand.register)
                )
                tail = tails[ch]
                tails[ch] = RuntimeEnd(
                    tail.register, 0, max(tail.maximum, operand.maximum)
                )

    def _store(self, prog, end: Time) -> RuntimeEnd:
        if self.entry_math is None:
            raise RuntimeError("repeat entry has no arithmetic plan")
        original = self.entry_math.cycles(end)
        register = prog.add_reg()
        self.entry_math.operations.append(WriteReg(dst=register, src=original.register))
        return RuntimeEnd(register, 0, original.maximum)

    def _native_stores(
        self,
        prog,
        initial: dict[int, float | QickParam],
        runtime: dict[int, RuntimeEnd],
        written: set[int],
    ) -> dict[int, RuntimeEnd]:
        result: dict[int, RuntimeEnd] = {}
        for ch, value in initial.items():
            end = runtime.get(ch, value)
            if time_bounds(prog, end)[1] <= 1e-12 and ch not in written:
                continue
            match = next(
                (
                    name
                    for name, candidate in self.initial_external.items()
                    if candidate is end
                    or (
                        not isinstance(candidate, RuntimeEnd)
                        and not isinstance(end, RuntimeEnd)
                        and all(abs(v) < 1e-12 for v in bounds(candidate - end))
                    )
                ),
                None,
            )
            candidate = (
                self.external[match]
                if match is not None and ch not in written
                else None
            )
            result[ch] = (
                candidate
                if isinstance(candidate, RuntimeEnd)
                else self._store(prog, end)
            )
        return result

    def rebase_native(self, prog, count: int) -> None:
        """Debit external timestamps, preserving tails generated in the last iteration."""
        generated_gens = {ch: prog.get_timestamp(gen_ch=ch) for ch in self.written_gens}
        generated_ros = {ch: prog.get_timestamp(ro_ch=ch) for ch in self.written_ros}
        for ch, end in self.native_gens.items():
            prog.set_timestamp(end, gen_ch=ch)
        for ch, end in self.native_ros.items():
            prog.set_timestamp(end, ro_ch=ch)
        prog.decrement_timestamps(self.advance * count)
        for ch, end in generated_gens.items():
            # With a zero minimum the body may not execute. Retain the later
            # native timestamp; QICK owns native timestamp comparison policy.
            value = end if count else max(end, prog.get_timestamp(gen_ch=ch))
            prog.set_timestamp(value, gen_ch=ch)
        for ch, end in generated_ros.items():
            value = end if count else max(end, prog.get_timestamp(ro_ch=ch))
            prog.set_timestamp(value, ro_ch=ch)


class TimelineBook:
    """Store only relative ends, including the ordinary default timeline."""

    def __init__(self):
        self.ends: dict[str, Time] = {"default": 0.0}
        self.cycle_axes: dict[str, CycleVariable] = {}
        self.math: RuntimeMath
        self.preparing = 0
        self.pinned: set[str] = set()
        self.runtime_gens: dict[int, RuntimeEnd] = {}
        self.runtime_ros: dict[int, RuntimeEnd] = {}
        self.waiting: set[str] = set()
        self.repeats: list[RepeatFrame] = []
        self.native_only = False
        self.used: set[str] | None = None

    def set_end(self, name: str, end: Time) -> None:
        self.math.check_end(end)
        self.ends[name] = end

    def require(self, name: str) -> None:
        """Reject unknown names without marking a timeline used or changing its end."""
        if name not in self.ends:
            raise ValueError(f"{name}: unknown timeline; declare it with fork in _body")

    def end(self, name: str, *, placement: bool = False) -> Time:
        self.require(name)
        if self.used is not None:
            self.used.add(name)
        if placement and name in self.waiting:
            raise ValueError(
                f"{name}: timeline can only wait after register repeat; "
                "sync it or use join/fork to establish a conservative scheduling point"
            )
        return self.ends[name]

    def decrement(self, t: Time, *, observe_pulses: bool = True) -> None:
        if self.native_only:
            return
        consumed = self._debit_native(t) if observe_pulses else set()
        self._debit_repeats(t, observe_pulses=observe_pulses, consumed=consumed)
        self.ends = {name: self.math.floor(end, t) for name, end in self.ends.items()}
        self.waiting = {
            name
            for name in self.waiting
            if time_bounds(self.math.prog, self.ends[name])[1] > 1e-12
        }

    def _debit_native(self, t: Time) -> set[int]:
        consumed: set[int] = set()
        for channels, generator in (
            (self.runtime_gens, True),
            (self.runtime_ros, False),
        ):
            for ch, end in list(channels.items()):
                remaining = self.math.floor(end, t)
                if isinstance(remaining, RuntimeEnd):
                    channels[ch] = remaining
                else:
                    del channels[ch]
                    if generator and time_bounds(self.math.prog, remaining)[1] <= 1e-12:
                        consumed.add(ch)
                self.native_timestamp(
                    self.math.prog,
                    projected_bounds(self.math.prog, remaining)[1],
                    gen_ch=ch if generator else None,
                    ro_ch=None if generator else ch,
                )

        return consumed

    def _debit_repeats(
        self, t: Time, *, observe_pulses: bool, consumed: set[int]
    ) -> None:
        if self.repeats:
            if observe_pulses:
                for active in self.repeats:
                    active.debits.append((self.math, t))
            frame = self.repeats[-1]
            proof_advance = (
                time_bounds(self.math.prog, t)[0] if isinstance(t, RuntimeEnd) else t
            )
            frame.variable = frame.variable or isinstance(t, RuntimeEnd)
            frame.advance += proof_advance
            if observe_pulses:
                for ch, guard in frame.pulse_guards.items():
                    # A consumed runtime pulse end proves a constant-zero
                    # transfer, preserving correlation between start and end.
                    shifts = () if ch in consumed else (-proof_advance,)
                    guard.then(_PulseGuard(ch, shifts=shifts, floors=(0.0,)))

    def native_timestamp(
        self,
        prog,
        value: float | QickParam,
        *,
        gen_ch: int | None = None,
        ro_ch: int | None = None,
    ) -> None:
        """Project a compiler bound without recording a new hardware event."""
        previous = self.native_only
        self.native_only = True
        try:
            prog.set_timestamp(value, gen_ch=gen_ch, ro_ch=ro_ch)
        finally:
            self.native_only = previous

    def channel_end(self, prog, ch: int, *, generator: bool) -> Time:
        """Return a declared native channel's exact runtime end or static timestamp."""
        runtime = self.runtime_gens if generator else self.runtime_ros
        if ch in runtime:
            return runtime[ch]
        return prog.get_timestamp(
            gen_ch=ch if generator else None, ro_ch=None if generator else ch
        )

    def channel_maximum(self, prog, *, gens: bool, ros: bool) -> Time | None:
        """Return selected native channel ends, retaining any runtime value.

        None means neither selection contains a declared channel, matching QICK
        auto-delay behavior. Static channels keep their native timestamps.
        """
        result: Time | None = None
        for enabled, channels, generator in (
            (gens, prog.gen_chs, True),
            (ros, prog.ro_chs, False),
        ):
            if enabled:
                for ch in channels:
                    value = self.channel_end(prog, ch, generator=generator)
                    result = (
                        value if result is None else self.math.maximum(result, value)
                    )
        return result

    def maximum(self) -> Time:
        end: Time = 0.0
        for candidate in self.ends.values():
            end = self.math.maximum(end, candidate)
        return end
