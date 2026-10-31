"""Preprocess timeline operations and translate their flag-free cycle plans."""

# tolerated(file-size): Branch/Repeat preparation shares TimelineBook frame and
# provenance state; snapshot/restore wiring must follow the same preparation order.

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace

from qick.asm_v2 import (
    CondJump,
    Delay,
    Jump,
    Label,
    Macro,
    OpenLoop,
    QickParam,
    TimedMacro,
    Wait,
    WriteReg,
)

from ._timeline_bounds import CycleBounds, FixedLoopIndex
from ._timeline_guard import PulseGuard as _PulseGuard
from ._timeline_guard import bounds as _bounds
from ._timeline_outputs import (
    TimelineObservedPulse,
    TimelinePlacedConfig,
    TimelinePlacedPulse,
    TimelinePlacedTrigger,
)
from ._timeline_runtime import (
    DeferredCoverage,
    EventEnd,
    RuntimeEnd,
    RuntimeMath,
    SelectedEventEnd,
    Time,
    _known_maximum,
    align_end,
    step_time,
    time_bounds,
)
from ._timeline_selected import TimelineSelectedPulse
from ._timeline_state import BranchProvenance, RepeatFrame, RepeatFrameSnapshot, TimelineBook
from .macro.delay import DelayRegAuto
from .macro.structure import MarkedCloseLoop, MarkedOpenLoop
from .macro.table import TableSelection
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
        if self.book.case_lineage:
            parents, _ = self.book.case_lineage[-1]
            parents.update(dict.fromkeys(self.targets, self.source))

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
        for parents, joined in self.book.case_lineage:
            joined.update(source for source in self.sources if source in parents)

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
        runtime = self.book.math.source(self.source.time_reg)
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
            self.book.cycle_axes.clear()
            self.book.source_expressions.clear()
            self.book.branch_provenance.clear()
            self.book.shell_wait_bound = None
            self.book.waiting.clear()
            self.book.repeats.clear()
            self.book.native_only = False
            self.book.runtime_gens.clear()
            self.book.runtime_ros.clear()
            self.book.native_expressions.clear()
            self.book.pinned.clear()
        elif self.section == "body_end":
            self.book.check_shell_wait(prog)

    def translate(self, prog) -> None:
        if self.section in ("body_start", "body_end"):
            prog.record_meta(
                type="TIMING_BOUNDARY", name=self.section, info={}, regs={}
            )


class TimelineShellWait(Wait):
    """Project only the final WAIT literal, leaving native reference debits alone."""

    def __init__(self, book: TimelineBook, t: float):
        super().__init__(t=t, auto=True, gens=False, ros=True, no_warn=True)
        self.book = book
        self.offset = t

    def preprocess(self, prog) -> None:
        bound = self.book.shell_wait_bound
        self.auto = bound is None
        self.t = self.offset if bound is None else self.offset + bound / get_fclk(prog)
        super().preprocess(prog)


class TimelineAdvance(TimedMacro):
    def __init__(self, book, timeline, duration, *, tag=None, aligned=False):
        super().__init__(tag=tag)
        self.book = book
        self.timeline = timeline
        self.duration = duration
        self.aligned = aligned
        self.event_ends: tuple[EventEnd | SelectedEventEnd, ...] = ()

    def preprocess(self, prog) -> None:
        if isinstance(self.duration, str):
            duration = self.book.math.snapshot(self.book.math.source(self.duration))
        else:
            duration = step_time(prog, self.duration) if self.aligned else self.duration
        previous = self.book.end(self.timeline, placement=True)
        events = self._events(previous)
        end = self._finish(prog, events[0], duration)
        for event in events[1:]:
            # max(ceil(Ei+post)) equals ceil(max(Ei)+post).
            end = self.book.math.maximum(end, self._finish(prog, event, duration))
        self.book.set_end(self.timeline, self.book.math.maximum(0.0, end))
        self.resolved_time = end

    def _events(self, previous: Time) -> list[EventEnd | SelectedEventEnd]:
        """Keep the raw fractional maximum envelope before adding any post."""
        events: list[EventEnd | SelectedEventEnd] = []
        # A selected event already pairs the chosen waveform with its post.
        # An unposted origin is not another child, including for negative posts.
        if not any(isinstance(event, SelectedEventEnd) for event in self.event_ends):
            events.append(EventEnd(previous, 0.0))
        for event in self.event_ends:
            for index, candidate in enumerate(events):
                if isinstance(event, SelectedEventEnd) or isinstance(
                    candidate, SelectedEventEnd
                ):
                    continue
                same_origin = event.origin is candidate.origin or (
                    isinstance(event.origin, (int, float))
                    and isinstance(candidate.origin, (int, float))
                    and event.origin == candidate.origin
                )
                if (
                    not same_origin
                    or isinstance(event.offset, RuntimeEnd)
                    or isinstance(candidate.offset, RuntimeEnd)
                ):
                    continue
                lo, hi = _bounds(event.offset - candidate.offset)
                if lo >= -1e-12 or hi <= 1e-12:
                    larger = event if lo >= -1e-12 else candidate
                    events[index] = EventEnd(
                        larger.origin, larger.offset, candidate.rounded + event.rounded
                    )
                    break
            else:
                # Crossing fractional sweeps must retain their tails until post
                # has been added; a runtime max here would round too early.
                events.append(event)
        return events

    def _finish_selected(self, event: SelectedEventEnd, duration: Time) -> RuntimeEnd:
        if isinstance(duration, RuntimeEnd):
            raise ValueError("selected post aggregate requires a scalar or swept post")
        offsets = tuple(
            offset + post + duration
            for offset, post in zip(event.offsets, event.posts, strict=True)
        )
        physical_offsets = event.offsets
        completed_offsets = offsets
        if not isinstance(event.origin, RuntimeEnd):
            physical_offsets = tuple(event.origin + offset for offset in event.offsets)
            completed_offsets = tuple(event.origin + offset for offset in offsets)
        if isinstance(event.origin, RuntimeEnd):
            selected = self.book.math.selected_end(event.selector, completed_offsets)
            selected_result = self.book.math.add(
                event.origin, selected, target=selected.register
            )
        else:
            selected_result = self.book.math.selected_end(
                event.selector, completed_offsets
            )
        debt = self.book.math.coverage_debt(physical_offsets, completed_offsets)
        if debt == 0:
            return replace(
                selected_result, covers=(*selected_result.covers, *event.rounded)
            )
        return replace(
            selected_result,
            deferred=(
                *selected_result.deferred,
                *(DeferredCoverage(boundary, debt) for boundary in event.rounded),
            ),
        )

    def _finish(self, prog, event: EventEnd | SelectedEventEnd, duration: Time) -> Time:
        if isinstance(event, SelectedEventEnd):
            return self._finish_selected(event, duration)
        fractional: float | QickParam = 0.0
        registers: list[RuntimeEnd] = []
        for value in (event.origin, event.offset, duration):
            if isinstance(value, RuntimeEnd):
                registers.append(value)
            else:
                fractional += value
        result: Time = fractional if self.aligned else align_end(prog, fractional)
        for register in registers:
            result = self.book.math.add(register, result)
        if (
            isinstance(result, RuntimeEnd)
            and event.rounded
            and time_bounds(prog, duration)[0] >= 0
        ):
            covers = [*result.covers, *event.rounded]
            for previous in event.rounded:
                if isinstance(previous, RuntimeEnd):
                    covers.extend(previous.covers)
            result = replace(result, covers=tuple(covers))
        return result

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
                initial_native_expressions=self.book.snapshot_native_expressions(prog),
                initial_source_expressions=self.book.source_expressions.copy(),
            )
        )
        self.book.ends = {parent: 0.0}
        self.book.waiting.clear()


class TimelineRepeatExit(TimelineSync):
    """Close one iteration, then rebase the outer namespace by all iterations."""

    def __init__(
        self,
        book: TimelineBook,
        count: int,
        maximum: int,
        *,
        variable: bool = False,
        iterations: int | str = 1,
        axis: str,
    ):
        super().__init__(book, None)
        self.axis = axis
        self.iterations = iterations
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
        frame.restore_expressions(
            prog,
            self.book.repeat_count(prog, self.iterations, self.count, self.maximum),
            generated_gens,
            generated_ros,
            axis=self.axis,
        )

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
    emission_name identifies this occurrence in timing-resource errors.
    Construction and timing placement are owned by TimelineProgramV2.emit.
    """

    def __init__(
        self, owner: Module, macros: list[Macro], tag: str | None, *, emission_name: str
    ):
        super().__init__(tag=tag)
        self.emission_name = emission_name
        self.owner = owner
        self.macros = macros
        self.finishers: list[PreparedTimelineMacro] = []

    def preprocess(self, prog) -> None:
        """Collect actual ends at a common origin; an advance adds post then rounds."""
        previous_owner = prog._timing_table_owner
        prog._timing_table_owner = self.emission_name
        try:
            pending: dict[str, list[EventEnd | SelectedEventEnd]] = {}
            self.finishers = []
            for macro in self.macros:
                source = (
                    macro.source if isinstance(macro, PreparedTimelineMacro) else macro
                )
                if isinstance(
                    source,
                    (
                        TimelinePlacedPulse,
                        TimelinePlacedConfig,
                        TimelinePlacedTrigger,
                        TimelineSelectedPulse,
                    ),
                ):
                    start = source.book.end(source.timeline)
                    macro.preprocess(prog)
                    pending.setdefault(source.timeline, []).append(source.event_end)
                    source.book.ends[source.timeline] = start
                else:
                    if isinstance(source, TimelineAdvance):
                        source.event_ends = tuple(pending.pop(source.timeline, ()))
                    macro.preprocess(prog)
            for timeline, ends in pending.items():
                advance = TimelineAdvance(prog._timeline_book, timeline, 0.0)
                advance.event_ends = tuple(ends)
                finisher = PreparedTimelineMacro(prog._timeline_book, advance)
                finisher.preprocess(prog)
                self.finishers.append(finisher)
        finally:
            prog._timing_table_owner = previous_owner

    def translate(self, prog) -> None:
        """Translate lowered operations through the program's shared compiler."""
        for macro in [*self.macros, *self.finishers]:
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
        previous_origin = prog._time_origin
        self.book.math = self.math
        self.book.preparing += 1
        try:
            prog._time_origin = self.source
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
            self.book.observe_write(prog, self.source)
        finally:
            prog._time_origin = previous_origin
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


class FixedRepeatLoop(MarkedOpenLoop):
    """Native fixed loop paired with the exact writer that closes its counter.

    index is Repeat-owned metadata; closing is the native matching backedge.
    The original QICK preprocess/expand behavior is unchanged. Keeping both
    identities lets the range proof reject unrelated counter writes even when
    this loop is nested inside a module emission rather than in loop_dict.
    """

    def __init__(self, name: str, count: int):
        super().__init__(name=name, n=count)
        self.index = FixedLoopIndex(name, name, count)
        self.closing = MarkedCloseLoop()


class PreparedTimelineOpenLoop(PreparedTimelineMacro, OpenLoop):
    """Preserve QICK loop discovery while tracking the actual write origin."""

    def __init__(self, book: TimelineBook, source: OpenLoop):
        super().__init__(book, source)
        self.name = source.name
        self.n = source.n


class TimeProducer(Macro):
    def __init__(
        self,
        register: str,
        limits: CycleBounds,
        macros: list[Macro],
        dependencies: tuple[str, ...],
        dependency_ranges: Mapping[str, CycleBounds],
        selection: TableSelection | None = None,
    ):
        self.selection = selection
        self.register = register
        self.limits = limits
        self.macros = macros
        self.dependencies = dependencies
        self.dependency_ranges = dict(dependency_ranges)
        self.prepared: list[PreparedTimelineMacro] = []

    def preprocess(self, prog) -> None:
        self.prepared = [
            PreparedTimelineMacro(prog._timeline_book, macro) for macro in self.macros
        ]
        prog._time_bounds.capture_inputs(
            prog, self.register, self.dependencies, self.dependency_ranges
        )
        for macro in self.prepared:
            macro.preprocess(prog)
        prog._time_bounds.defined.add(self.register)
        prog._timeline_book.observe_selection(prog, self.register, self.selection)

    def translate(self, prog) -> None:
        for macro in self.prepared:
            macro.translate(prog)


class BranchCases:
    """Collect native macro lists without executing any case at build time."""

    def __init__(self, prog):
        self.prog = prog
        self.cases: dict[int, list[Macro]] = {}
        self.active = False

    @contextmanager
    def case(self, index: int) -> Generator[None]:
        """Record the zero-based selector point; omitted points do nothing.

        Cases may be declared in any order. Duplicate indices and nested cases
        raise ValueError. Non-integer or out-of-range indices fail at compile.
        """
        if self.active:
            raise ValueError("branch.case cannot overlap another case")
        if index in self.cases:
            raise ValueError(f"duplicate branch case index: {index}")
        self.active = True
        start = len(self.prog.macro_list)
        try:
            yield
        finally:
            self.cases[index] = self.prog.macro_list[start:]
            del self.prog.macro_list[start:]
            self.active = False


def _same_times(values: Sequence[Time]) -> bool:
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


@dataclass
class BranchCaseState:
    """Suspend a prepared path until the whole branch's closing mode is known."""

    ends: dict[str, Time]
    waiting: set[str]
    used: set[str]
    runtime_gens: dict[int, RuntimeEnd]
    runtime_ros: dict[int, RuntimeEnd]
    gens: dict[int, Time]
    ros: dict[int, Time]
    provenance: BranchProvenance
    frames: list[RepeatFrameSnapshot]
    defined: set[str]
    static_ends: dict[str, Time] | None


class TimelineBranch(Macro):
    def __init__(
        self, book: TimelineBook, by: str, cases: dict[int, list[Macro]], name: str
    ):
        self.book = book
        self.by = by
        self.indexed_cases = cases
        # Producer and parallel-effect audits visit declared bodies before prepare.
        self.cases: list[list[Macro]] = list(cases.values())
        self.name = name
        self.prepared: list[list[PreparedTimelineMacro]] = []

    def preprocess(self, prog) -> None:
        if self.by not in prog.loop_dict:
            raise ValueError(
                f"{self.by}: Branch requires a sweep selector, not measurement feedback"
            )
        count = prog.loop_dict[self.by]
        for index in self.indexed_cases:
            if type(index) is not int or not 0 <= index < count:
                raise ValueError(
                    f"{self.by}: branch case index {index!r} must be an integer in 0..{count - 1}"
                )
        self.cases = [self.indexed_cases.get(index, []) for index in range(count)]
        initial = self.book.ends.copy()
        self.book.begin_branch_sources(prog)
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
        paths: list[list[RepeatFrameSnapshot]] = []
        states: list[BranchCaseState] = []
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
            self.prepared.append(self._prepare_case(prog, case))
            parents, joined = self.book.case_lineage.pop()
            states.append(
                BranchCaseState(
                    self.book.ends.copy(),
                    self.book.waiting.copy(),
                    (self.book.used or set()).copy(),
                    self.book.runtime_gens.copy(),
                    self.book.runtime_ros.copy(),
                    {ch: prog.get_timestamp(gen_ch=ch) for ch in prog.gen_chs},
                    {ch: prog.get_timestamp(ro_ch=ch) for ch in prog.ro_chs},
                    self.book.snapshot_branch_provenance(prog),
                    self._save_frames(),
                    prog._time_bounds.defined.copy(),
                    self._static_case_ends(initial, parents, joined),
                )
            )
            self.book.pinned.update(prog._timeline_live_registers())
        static_ends = self._static_branch_ends(initial, states)
        for prepared, state in zip(self.prepared, states, strict=True):
            self._restore_case(prog, state)
            if static_ends is None:
                used = state.used | (state.ends.keys() - initial.keys())
                closer = PreparedTimelineMacro(
                    self.book, TimelineSync(self.book, None, sources=tuple(sorted(used)))
                )
                closer.preprocess(prog)
                prepared.append(closer)
            self.book.record_branch_sources(prog)
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
        self._merge_frames(prog, paths)
        prog._time_bounds.defined = set.intersection(*definitions)
        self.book.finish_branch_sources(prog, self.by)
        self.book.used, self.book.waiting = outer_used, waiting
        self.tails = [RuntimeMath(prog) for _ in results]
        self.book.ends = {}
        for name in initial:
            values = [result[name] for result in results]
            self.book.ends[name] = (
                static_ends[name]
                if static_ends is not None
                else self._merge_values(prog, values)
            )
            if static_ends is None and not _same_times(values):
                self.book.waiting.add(name)
        self._merge_native(prog, native_gens, results, generator=True)
        self._merge_native(prog, native_ros, results, generator=False)
        self.book.pinned = saved_pins

    def _prepare_case(self, prog, case: list[Macro]) -> list[PreparedTimelineMacro]:
        self.book.case_lineage.append(({}, set()))
        prepared = [PreparedTimelineMacro(self.book, macro) for macro in case]
        for macro in prepared:
            macro.preprocess(prog)
        return prepared

    def _static_case_ends(
        self, initial: dict[str, Time], parents: dict[str, str], joined: set[str]
    ) -> dict[str, Time] | None:
        used = (self.book.used or set()) | (self.book.ends.keys() - initial.keys())
        if any(isinstance(self.book.ends[name], RuntimeEnd) for name in used):
            return None
        ends = {name: self.book.ends[name] for name in initial}
        for local in self.book.ends.keys() - initial.keys():
            if local in joined:
                continue
            ancestor = parents[local]
            while ancestor not in initial:
                ancestor = parents[ancestor]
            merged = _known_maximum(ends[ancestor], self.book.ends[local])
            if merged is None:
                return None
            ends[ancestor] = merged
        return ends

    def _static_branch_ends(
        self, initial: dict[str, Time], states: list[BranchCaseState]
    ) -> dict[str, Time] | None:
        ends = initial.copy()
        used = set.union(*(state.used & initial.keys() for state in states))
        for state in states:
            if state.static_ends is None:
                return None
            for name in used:
                value = state.static_ends[name]
                if isinstance(value, RuntimeEnd):
                    return None
                merged = _known_maximum(ends[name], value)
                if merged is None:
                    return None
                ends[name] = merged
        return ends

    def _restore_case(self, prog, state: BranchCaseState) -> None:
        self._restore_frames(state.frames)
        self.book.ends, self.book.waiting = state.ends.copy(), state.waiting.copy()
        self.book.used = state.used.copy()
        self.book.runtime_gens = state.runtime_gens.copy()
        self.book.runtime_ros = state.runtime_ros.copy()
        prog._time_bounds.defined = state.defined.copy()
        for ch, value in state.gens.items():
            self.book.native_timestamp(prog, value, gen_ch=ch)
        for ch, value in state.ros.items():
            self.book.native_timestamp(prog, value, ro_ch=ch)
        self.book.source_expressions = state.provenance.sources.copy()
        self.book.native_expressions = state.provenance.native.copy()

    def _merge_values(self, prog, values: list[Time]) -> Time:
        if _same_times(values):
            return values[0]
        limits = [time_bounds(prog, value) for value in values]
        merged = RuntimeEnd(
            prog.add_reg(),
            round(min(lo for lo, _ in limits) * get_fclk(prog)),
            round(max(hi for _, hi in limits) * get_fclk(prog)),
            expression=self.book.selected_expression(prog, self.by, values),
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

    def _save_frames(self) -> list[RepeatFrameSnapshot]:
        return [
            RepeatFrameSnapshot(
                advance=frame.advance,
                variable=frame.variable,
                written_gens=frame.written_gens.copy(),
                written_ros=frame.written_ros.copy(),
                pulse_guards=deepcopy(frame.pulse_guards),
                expression_advance=frame.expression_advance,
            )
            for frame in self.book.repeats
        ]

    def _restore_frames(self, snapshots: list[RepeatFrameSnapshot]) -> None:
        self.book.restore_branch_sources()
        for frame, snapshot in zip(self.book.repeats, snapshots, strict=True):
            frame.advance = snapshot.advance
            frame.expression_advance = snapshot.expression_advance
            frame.variable = snapshot.variable
            frame.written_gens = snapshot.written_gens.copy()
            frame.written_ros = snapshot.written_ros.copy()
            frame.pulse_guards = deepcopy(snapshot.pulse_guards)

    def _merge_frames(self, prog, paths: list[list[RepeatFrameSnapshot]]) -> None:
        for index, frame in enumerate(self.book.repeats):
            snapshots = [path[index] for path in paths]
            frame.merge_expression_advance(prog, self.by, snapshots)
            advances = [snapshot.advance for snapshot in snapshots]
            frame.advance = (
                advances[0]
                if _same_times(advances)
                else min(_bounds(value)[0] for value in advances)
            )
            frame.variable = any(
                snapshot.variable for snapshot in snapshots
            ) or not _same_times(advances)
            frame.written_gens = set.union(*(path.written_gens for path in snapshots))
            frame.written_ros = set.union(*(path.written_ros for path in snapshots))
            for ch in frame.pulse_guards:
                guards = [snapshot.pulse_guards[ch] for snapshot in snapshots]
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
