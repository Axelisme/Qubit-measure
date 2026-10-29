"""Preprocess timeline operations and translate their flag-free cycle plans."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from copy import deepcopy

from qick.asm_v2 import (
    CondJump,
    Delay,
    Jump,
    Label,
    Macro,
    QickParam,
    TimedMacro,
    WriteReg,
)

from ._timeline_bounds import CycleBounds
from ._timeline_guard import PulseGuard as _PulseGuard
from ._timeline_guard import bounds as _bounds
from ._timeline_outputs import (
    TimelineObservedPulse,
    TimelinePlacedConfig,
    TimelinePlacedPulse,
    TimelinePlacedTrigger,
)
from ._timeline_runtime import (
    RuntimeEnd,
    RuntimeMath,
    Time,
    align_end,
    step_time,
    time_bounds,
)
from ._timeline_state import RepeatFrame, TimelineBook
from .macro.delay import DelayRegAuto
from .modules.base import Module
from .modules.util import get_fclk


class TimelineFork(Macro):
    """Copy one declared end to fresh names at this preprocessing position."""

    def __init__(self, book: TimelineBook, source: str, targets: tuple[str, ...]):
        self.book = book
        self.source = source
        self.targets = targets

    def preprocess(self, prog) -> None:
        super().preprocess(prog)
        end = self.book.end(self.source)
        if len(set(self.targets)) != len(self.targets) or any(
            name in self.book.ends
            or any(name in frame.external for frame in self.book.repeats)
            for name in self.targets
        ):
            raise ValueError("fork requires new, distinct timeline names")
        self.book.ends.update(dict.fromkeys(self.targets, end))

    def translate(self, prog) -> None:
        pass


class TimelineJoin(Macro):
    """Merge existing ends without emitting instructions or consuming sources."""

    def __init__(self, book: TimelineBook, sources: tuple[str, ...], into: str):
        self.book = book
        self.sources = sources
        self.into = into

    def preprocess(self, prog) -> None:
        super().preprocess(prog)
        end = self.book.end(self.into)
        for source in self.sources:
            end = self.book.math.maximum(end, self.book.end(source))
        self.book.ends[self.into] = end
        self.book.waiting.discard(self.into)

    def translate(self, prog) -> None:
        pass


class TimelineRuntimeAdvance(Macro):
    """Guard all ends against the static part of an unknown nonnegative advance."""

    def __init__(self, book: TimelineBook, source: DelayRegAuto):
        self.book = book
        self.source = source

    def preprocess(self, prog) -> None:
        if self.book.repeats:
            raise ValueError(
                "repeat per-iteration reference advance cannot be expressed at compile time "
                "when using a runtime register delay"
            )
        auto = self.book.channel_maximum(
            prog, gens=self.source.gens, ros=self.source.ros
        )
        self.book.native_only = True
        try:
            self.source.preprocess(prog)
        finally:
            self.book.native_only = False
        static = self.source.get_time_param("auto_t")
        if static is None:
            static = 0.0
        advance = auto if isinstance(auto, RuntimeEnd) else static
        if isinstance(auto, RuntimeEnd):
            self.source.t_regs["auto_t"] = auto.register
        remaining = [
            self.book.math.floor(end, advance) for end in self.book.ends.values()
        ]
        if all(
            not isinstance(end, RuntimeEnd) and _bounds(end)[1] <= 1e-12
            for end in remaining
        ):
            self.book.decrement(advance)
            return
        limits = prog._time_bounds.source(prog, self.source.time_reg)
        runtime = RuntimeEnd(self.source.time_reg, limits.minimum, limits.maximum)
        self.book.decrement(self.book.math.add(runtime, advance))

    def translate(self, prog) -> None:
        self.source.translate(prog)


class TimelineBoundary(Macro):
    """Reset preprocessing state or emit a linked IR timing boundary."""

    def __init__(self, book: TimelineBook, section: str):
        self.book = book
        self.section = section

    def preprocess(self, prog) -> None:
        super().preprocess(prog)
        if self.section == "reset":
            self.book.ends = {"default": 0.0}
            self.book.waiting.clear()
            self.book.repeats.clear()
            self.book.native_only = False
            self.book.runtime_gens.clear()
            self.book.runtime_ros.clear()
            self.book.pinned.clear()

    def translate(self, prog) -> None:
        if self.section in ("body_start", "body_end"):
            prog.record_meta(
                type="TIMING_BOUNDARY", name=self.section, info={}, regs={}
            )


class TimelineAdvance(TimedMacro):
    def __init__(self, book, timeline, duration, *, tag=None, aligned=False):
        super().__init__(tag=tag)
        self.book = book
        self.timeline = timeline
        self.duration = duration
        self.aligned = aligned

    def preprocess(self, prog) -> None:
        if isinstance(self.duration, str):
            limits = prog._time_bounds.source(prog, self.duration)
            duration = self.book.math.snapshot(
                RuntimeEnd(self.duration, limits.minimum, limits.maximum)
            )
        else:
            duration = (
                step_time(prog, self.duration)
                if self.aligned
                else align_end(prog, self.duration)
            )
        previous = self.book.end(self.timeline, placement=True)
        end = (
            self.book.math.add(previous, duration)
            if isinstance(previous, RuntimeEnd) or isinstance(duration, RuntimeEnd)
            else previous + duration
        )
        self.book.set_end(self.timeline, self.book.math.maximum(0.0, end))
        self.resolved_time = end

    def list_time_params(self) -> list[str]:
        return ["t"]

    def get_time_param(self, name: str) -> QickParam:
        if name != "t":
            raise RuntimeError("timeline advances expose only the t parameter")
        if isinstance(self.resolved_time, RuntimeEnd):
            raise RuntimeError("runtime timeline end is not an affine time parameter")
        return (
            self.resolved_time
            if isinstance(self.resolved_time, QickParam)
            else QickParam(self.resolved_time)
        )

    def translate(self, prog) -> None:
        pass


class TimelineSync(Delay):
    def __init__(
        self,
        book: TimelineBook,
        timeline: str | None,
        *,
        sources: tuple[str, ...] | None = None,
    ):
        super().__init__(t=0, auto=False)
        self.book = book
        self.timeline = timeline
        self.sources = sources

    def preprocess(self, prog) -> None:
        self.t = (
            self.book.maximum()
            if self.timeline is None
            else self.book.end(self.timeline)
        )
        if self.sources is not None:
            self.t = 0.0
            for name in self.sources:
                self.t = self.book.math.maximum(self.t, self.book.end(name))
        if isinstance(self.t, RuntimeEnd):
            self.t_regs["t"] = self.t.register
            self.book.native_only = True
            try:
                prog.decrement_timestamps(self.t.minimum / get_fclk(prog))
            finally:
                self.book.native_only = False
            self.book.decrement(self.t)
        else:
            super().preprocess(prog)


class TimelineRepeatEnter(TimelineSync):
    """Synchronize the parent, then isolate this iteration's local namespace."""

    def __init__(self, book: TimelineBook, timeline: str, counter: str | None = None):
        super().__init__(book, timeline)
        self.counter = counter

    def preprocess(self, prog) -> None:
        super().preprocess(prog)
        if self.counter is not None:
            prog.add_reg(self.counter)
        parent = self.timeline
        if parent is None:
            raise ValueError("repeat requires a parent timeline")
        external = self.book.ends.copy()
        del external[parent]
        initial_external = external.copy()
        self.book.repeats.append(
            RepeatFrame(
                parent,
                external,
                self.book.waiting.copy(),
                {ch: prog.get_timestamp(gen_ch=ch) for ch in prog.gen_chs},
                {ch: prog.get_timestamp(ro_ch=ch) for ch in prog.ro_chs},
                pulse_guards={ch: _PulseGuard(ch) for ch in prog.gen_chs},
                defined=prog._time_bounds.defined.copy(),
                initial_external=initial_external,
                entry_math=self.book.math,
                initial_runtime_gens=self.book.runtime_gens.copy(),
                initial_runtime_ros=self.book.runtime_ros.copy(),
            )
        )
        self.book.ends = {parent: 0.0}
        self.book.waiting.clear()


class TimelineRepeatExit(TimelineSync):
    """Close one iteration, then rebase the outer namespace by all iterations."""

    def __init__(
        self, book: TimelineBook, count: int, maximum: int, *, variable: bool = False
    ):
        super().__init__(book, None)
        self.count = count
        self.maximum = maximum
        self.variable = variable

    def preprocess(self, prog) -> None:
        super().preprocess(prog)
        frame = self.book.repeats.pop()
        if self.count == 0:
            prog._time_bounds.defined.intersection_update(frame.defined)
        for ch, guard in frame.pulse_guards.items():
            repeated = guard.repeated(self.count, self.maximum)
            repeated.require((frame.native_gens[ch],), repeated.limits)
            if self.book.repeats:
                self.book.repeats[-1].pulse_guards[ch].then(repeated)
        variable = self.variable or frame.variable
        if variable and _bounds(frame.advance)[0] < -1e-12:
            raise ValueError(
                "repeat cannot prove minimum reference advance at every sweep point"
            )
        generated_gens = {
            ch: self.book.channel_end(prog, ch, generator=True)
            for ch in frame.written_gens
        }
        generated_ros = {
            ch: self.book.channel_end(prog, ch, generator=False)
            for ch in frame.written_ros
        }
        if variable:
            frame.promote(prog, generated_gens, generated_ros)
        # The static envelope serves native overlap proof only; runtime tails
        # below restore exact relative values for shell and native consumers.
        self.book.native_only = True
        try:
            frame.rebase_native(prog, self.count)
        finally:
            self.book.native_only = False
        if self.book.repeats:
            self.book.repeats[-1].written_gens.update(frame.written_gens)
            self.book.repeats[-1].written_ros.update(frame.written_ros)
        self.book.ends = frame.external
        self.book.waiting = frame.waiting
        # Pulse summaries already include every inner iteration's debit.
        static: dict[str, Time] = {
            name: end
            for name, end in self.book.ends.items()
            if not isinstance(end, RuntimeEnd)
        }
        runtime: dict[str, Time] = {}
        for name, end in self.book.ends.items():
            if isinstance(end, RuntimeEnd):
                upper = time_bounds(prog, frame.initial_external[name])[1]
                debit = _bounds(frame.advance)[0] * self.count
                runtime[name] = (
                    RuntimeEnd(
                        end.register, 0, max(0, round((upper - debit) * get_fclk(prog)))
                    )
                    if variable
                    else self.book.math.floor(end, frame.advance * self.count)
                )
        self.book.ends = static
        self.book.decrement(frame.advance * self.count, observe_pulses=False)
        self.book.ends.update(runtime)
        if variable:
            self.book.waiting.update(
                name
                for name, end in self.book.ends.items()
                if time_bounds(prog, end)[1] > 1e-12
            )
            if self.book.repeats:
                self.book.repeats[-1].variable = True
        self.book.ends[frame.parent] = 0.0
        self._restore_native(prog, frame, variable=variable)

    def _restore_native(self, prog, frame: RepeatFrame, *, variable: bool) -> None:
        for tails, initial, current, written, generator in (
            (
                frame.tail_gens,
                frame.initial_runtime_gens,
                self.book.runtime_gens,
                frame.written_gens,
                True,
            ),
            (
                frame.tail_ros,
                frame.initial_runtime_ros,
                self.book.runtime_ros,
                frame.written_ros,
                False,
            ),
        ):
            if variable:
                for ch, end in tails.items():
                    upper = (
                        end.maximum
                        if ch in written
                        else max(
                            0,
                            end.maximum
                            - round(
                                _bounds(frame.advance)[0] * self.count * get_fclk(prog)
                            ),
                        )
                    )
                    current[ch] = RuntimeEnd(end.register, 0, upper)
            else:
                for ch, end in initial.items():
                    if ch not in written:
                        remaining = self.book.math.floor(
                            end, frame.advance * self.count
                        )
                        if isinstance(remaining, RuntimeEnd):
                            current[ch] = remaining
                        else:
                            current.pop(ch, None)
            for ch, end in current.items():
                self.book.native_timestamp(
                    prog,
                    time_bounds(prog, end)[1],
                    gen_ch=ch if generator else None,
                    ro_ch=None if generator else ch,
                )


class ModuleEmission(TimedMacro):
    """One occurrence of a declared module, lowered through the shared compiler.

    owner is the borrowed declaration object, macros are fresh native operations,
    and tag identifies this occurrence's first timed operation in get_time_param.
    Construction and timing placement are owned by TimelineProgramV2.emit.
    """

    def __init__(self, owner: Module, macros: list[Macro], tag: str | None):
        super().__init__(tag=tag)
        self.owner = owner
        self.macros = macros

    def preprocess(self, prog) -> None:
        """Resolve internal events at one origin; only owner duration moves its end."""
        for macro in self.macros:
            source = macro.source if isinstance(macro, PreparedTimelineMacro) else macro
            if isinstance(
                source,
                (TimelinePlacedPulse, TimelinePlacedConfig, TimelinePlacedTrigger),
            ):
                start = source.book.end(source.timeline)
                macro.preprocess(prog)
                source.book.ends[source.timeline] = start
            else:
                macro.preprocess(prog)

    def translate(self, prog) -> None:
        """Translate lowered operations through the program's shared compiler."""
        for macro in self.macros:
            macro.translate(prog)

    def get_time_param(self, name: str) -> QickParam:
        """Return first-event timing; absent events/parameters raise RuntimeError."""
        for macro in self.macros:
            if isinstance(macro, TimedMacro) and macro.list_time_params():
                return macro.get_time_param(name)
        raise RuntimeError("this emission has no timed operations")

    def list_time_params(self) -> list[str]:
        """List first-event parameter names, or [] for an untimed emission."""
        for macro in self.macros:
            if isinstance(macro, TimedMacro) and macro.list_time_params():
                return macro.list_time_params()
        return []


class NativeTimelineDelay(TimedMacro):
    """Keep QICK auto-delay channel selection while consuming runtime tails."""

    def __init__(self, book: TimelineBook, source: Delay):
        super().__init__(tag=getattr(source, "tag", None))
        self.book = book
        self.source = source

    def preprocess(self, prog) -> None:
        auto = self.book.channel_maximum(
            prog, gens=self.source.gens, ros=self.source.ros
        )
        if not isinstance(auto, RuntimeEnd):
            self.source.preprocess(prog)
            return
        actual = self.book.math.add(auto, step_time(prog, self.source.t))
        self.book.native_only = True
        try:
            self.source.preprocess(prog)
        finally:
            self.book.native_only = False
        self.source.t_regs["t"] = actual.register
        self.book.decrement(actual)

    def translate(self, prog) -> None:
        self.source.translate(prog)

    def get_time_param(self, name: str) -> QickParam:
        return self.source.get_time_param(name)

    def list_time_params(self) -> list[str]:
        return self.source.list_time_params()


class PreparedTimelineMacro(TimedMacro):
    """Keep cycle arithmetic at the native operation's execution position."""

    def __init__(self, book: TimelineBook, source: Macro):
        super().__init__(tag=getattr(source, "tag", None))
        self.book = book
        self.source = (
            NativeTimelineDelay(book, source)
            if isinstance(source, Delay) and source.auto
            else source
        )

    def preprocess(self, prog) -> None:
        self.math = RuntimeMath(prog)
        previous = self.book.math if self.book.preparing else None
        self.book.math = self.math
        self.book.preparing += 1
        try:
            self.source.preprocess(prog)
            timing_source = (
                self.source.source
                if isinstance(
                    self.source,
                    (
                        NativeTimelineDelay,
                        TimelineRuntimeAdvance,
                        TimelineObservedPulse,
                        TimelinePlacedConfig,
                    ),
                )
                else self.source
            )
            if isinstance(timing_source, TimedMacro):
                prog._time_bounds.time_parameters(timing_source)
            prog._time_bounds.activate(self.source)
        finally:
            self.book.preparing -= 1
            if previous is not None:
                self.book.math = previous

    def translate(self, prog) -> None:
        previous = prog._time_origin
        try:
            prog._time_origin = self.math
            self.math.translate(prog)
            prog._time_origin = self.source
            self.source.translate(prog)
        finally:
            prog._time_origin = previous

    def list_time_params(self) -> list[str]:
        return (
            self.source.list_time_params()
            if isinstance(self.source, TimedMacro)
            else []
        )

    def get_time_param(self, name: str) -> QickParam:
        if not isinstance(self.source, TimedMacro):
            raise RuntimeError("this operation has no time parameters")
        return self.source.get_time_param(name)


class TimeProducer(Macro):
    def __init__(
        self,
        book: TimelineBook,
        register: str,
        limits: CycleBounds,
        macros: list[Macro],
        dependencies: tuple[str, ...],
        dependency_ranges: Mapping[str, CycleBounds],
    ):
        self.register = register
        self.limits = limits
        self.macros = macros
        self.dependencies = dependencies
        self.dependency_ranges = dict(dependency_ranges)
        self.prepared = [PreparedTimelineMacro(book, macro) for macro in macros]

    def preprocess(self, prog) -> None:
        for macro in self.prepared:
            macro.preprocess(prog)
        prog._time_bounds.defined.add(self.register)

    def translate(self, prog) -> None:
        for macro in self.prepared:
            macro.translate(prog)


class BranchCases:
    """Collect native macro lists without executing any case at build time."""

    def __init__(self, prog):
        self.prog = prog
        self.cases: list[list[Macro]] = []
        self.active = False

    @contextmanager
    def case(self) -> Generator[None]:
        """Append one sweep-selected case; nested case() calls raise ValueError."""
        if self.active:
            raise ValueError("branch.case cannot overlap another case")
        self.active = True
        start = len(self.prog.macro_list)
        try:
            yield
        finally:
            self.cases.append(self.prog.macro_list[start:])
            del self.prog.macro_list[start:]
            self.active = False


def _same_times(values: list[Time]) -> bool:
    first = values[0]
    return all(
        value is first
        or (
            not isinstance(value, RuntimeEnd)
            and not isinstance(first, RuntimeEnd)
            and all(abs(v) < 1e-12 for v in _bounds(value - first))
        )
        for value in values
    )


class TimelineBranch(Macro):
    def __init__(
        self, book: TimelineBook, by: str, cases: list[list[Macro]], name: str
    ):
        self.book = book
        self.by = by
        self.cases = cases
        self.name = name
        self.prepared: list[list[PreparedTimelineMacro]] = []

    def preprocess(self, prog) -> None:
        if self.by not in prog.loop_dict:
            raise ValueError(
                f"{self.by}: Branch requires a sweep selector, not measurement feedback"
            )
        if len(self.cases) != prog.loop_dict[self.by]:
            raise ValueError("Branch must provide one case per selector point")
        initial = self.book.ends.copy()
        waiting = self.book.waiting.copy()
        initial_gens = {ch: prog.get_timestamp(gen_ch=ch) for ch in prog.gen_chs}
        initial_ros = {ch: prog.get_timestamp(ro_ch=ch) for ch in prog.ro_chs}
        runtime_gens, runtime_ros = (
            self.book.runtime_gens.copy(),
            self.book.runtime_ros.copy(),
        )
        outer_used = self.book.used
        initial_defined = prog._time_bounds.defined.copy()
        saved_pins = self.book.pinned.copy()
        self.book.pinned.update(prog._timeline_live_registers())
        frames = self._save_frames()
        definitions: list[set[str]] = []
        results: list[dict[str, Time]] = []
        native_gens: list[dict[int, Time]] = []
        native_ros: list[dict[int, Time]] = []
        paths = []
        self.prepared = []
        for case in self.cases:
            self._restore_frames(frames)
            prog._time_bounds.defined = initial_defined.copy()
            self.book.ends, self.book.waiting = initial.copy(), waiting.copy()
            self.book.runtime_gens, self.book.runtime_ros = (
                runtime_gens.copy(),
                runtime_ros.copy(),
            )
            self.book.used = set()
            for ch, value in initial_gens.items():
                self.book.native_timestamp(prog, value, gen_ch=ch)
            for ch, value in initial_ros.items():
                self.book.native_timestamp(prog, value, ro_ch=ch)
            self.prepared.append(self._prepare_case(prog, case, initial.keys()))
            results.append({name: self.book.ends[name] for name in initial})
            definitions.append(prog._time_bounds.defined.copy())
            native_gens.append(
                {
                    ch: self.book.channel_end(prog, ch, generator=True)
                    for ch in prog.gen_chs
                }
            )
            native_ros.append(
                {
                    ch: self.book.channel_end(prog, ch, generator=False)
                    for ch in prog.ro_chs
                }
            )
            paths.append(self._save_frames())
            self.book.pinned.update(prog._timeline_live_registers())
            if outer_used is not None:
                outer_used.update(self.book.used & initial.keys())
        self._merge_frames(paths)
        prog._time_bounds.defined = set.intersection(*definitions)
        self.book.used, self.book.waiting = outer_used, waiting
        self.tails = [RuntimeMath(prog) for _ in results]
        self.book.ends = {}
        for name in initial:
            values = [result[name] for result in results]
            self.book.ends[name] = self._merge_values(prog, values)
            if not _same_times(values):
                self.book.waiting.add(name)
        self._merge_native(prog, native_gens, results, generator=True)
        self._merge_native(prog, native_ros, results, generator=False)
        self.book.pinned = saved_pins

    def _prepare_case(
        self, prog, case: list[Macro], incoming
    ) -> list[PreparedTimelineMacro]:
        prepared = [PreparedTimelineMacro(self.book, macro) for macro in case]
        for macro in prepared:
            macro.preprocess(prog)
        local = self.book.ends.keys() - incoming
        used = (self.book.used or set()) | local
        closer = PreparedTimelineMacro(
            self.book, TimelineSync(self.book, None, sources=tuple(sorted(used)))
        )
        closer.preprocess(prog)
        prepared.append(closer)
        return prepared

    def _merge_values(self, prog, values: list[Time]) -> Time:
        if _same_times(values):
            return values[0]
        limits = [time_bounds(prog, value) for value in values]
        merged = RuntimeEnd(
            prog.add_reg(),
            round(min(lo for lo, _ in limits) * get_fclk(prog)),
            round(max(hi for _, hi in limits) * get_fclk(prog)),
        )
        for tail, value in zip(self.tails, values, strict=True):
            operand = tail.cycles(value)
            tail.operations.append(WriteReg(dst=merged.register, src=operand.register))
        return merged

    def _merge_native(
        self,
        prog,
        snapshots: list[dict[int, Time]],
        results: list[dict[str, Time]],
        *,
        generator: bool,
    ) -> None:
        channels = self.book.runtime_gens if generator else self.book.runtime_ros
        channels.clear()
        for ch in snapshots[0]:
            values = [snapshot[ch] for snapshot in snapshots]
            matching = next(
                (
                    name
                    for name in self.book.ends
                    if all(
                        _same_times([snapshot[ch], result[name]])
                        for snapshot, result in zip(snapshots, results, strict=True)
                    )
                ),
                None,
            )
            value = (
                self.book.ends[matching]
                if matching is not None
                else self._merge_values(prog, values)
            )
            if isinstance(value, RuntimeEnd):
                channels[ch] = value
            self.book.native_timestamp(
                prog,
                time_bounds(prog, value)[1],
                gen_ch=ch if generator else None,
                ro_ch=None if generator else ch,
            )

    def _save_frames(self):
        return [
            (
                frame.advance,
                frame.variable,
                frame.written_gens.copy(),
                frame.written_ros.copy(),
                deepcopy(frame.pulse_guards),
            )
            for frame in self.book.repeats
        ]

    def _restore_frames(self, snapshots) -> None:
        for frame, (advance, variable, gens, ros, guards) in zip(
            self.book.repeats, snapshots, strict=True
        ):
            frame.advance, frame.variable = advance, variable
            frame.written_gens, frame.written_ros = gens.copy(), ros.copy()
            frame.pulse_guards = deepcopy(guards)

    def _merge_frames(self, paths) -> None:
        for index, frame in enumerate(self.book.repeats):
            snapshots = [path[index] for path in paths]
            advances = [snapshot[0] for snapshot in snapshots]
            frame.advance = (
                advances[0]
                if _same_times(advances)
                else min(_bounds(value)[0] for value in advances)
            )
            frame.variable = any(
                snapshot[1] for snapshot in snapshots
            ) or not _same_times(advances)
            frame.written_gens = set.union(*(snapshot[2] for snapshot in snapshots))
            frame.written_ros = set.union(*(snapshot[3] for snapshot in snapshots))
            for ch in frame.pulse_guards:
                guards = [snapshot[4][ch] for snapshot in snapshots]
                frame.pulse_guards[ch] = _PulseGuard(
                    ch,
                    tuple(value for guard in guards for value in guard.shifts),
                    tuple(value for guard in guards for value in guard.floors),
                    tuple(value for guard in guards for value in guard.limits),
                )

    def translate(self, prog) -> None:
        prog.record_meta(
            type="BRANCH_START", name=self.name, info={}, regs={"compare_reg": self.by}
        )
        for index in range(len(self.cases) - 1):
            CondJump(
                label=f"{self.name}_case_{index}",
                arg1=self.by,
                test="Z",
                op="-",
                arg2=index,
            ).translate(prog)
        Jump(label=f"{self.name}_case_{len(self.cases) - 1}").translate(prog)
        for index, (case, tail) in enumerate(
            zip(self.prepared, self.tails, strict=True)
        ):
            prog.record_meta(
                type="BRANCH_CASE_START", name=str(index), info={}, regs={}
            )
            Label(label=f"{self.name}_case_{index}").translate(prog)
            for macro in case:
                macro.translate(prog)
            tail.translate(prog)
            prog.record_meta(type="BRANCH_CASE_END", name=str(index), info={}, regs={})
            Jump(label=f"{self.name}_end").translate(prog)
        prog.record_meta(type="BRANCH_END", name=self.name, info={}, regs={})
        Label(label=f"{self.name}_end").translate(prog)
