"""Native QICK programs with timeline ends relative to reference time."""

import math
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from qick.asm_v2 import (
    ConfigReadout,
    Delay,
    Macro,
    Pulse,
    QickParam,
    TimedMacro,
    Trigger,
)

from .base import MyProgramV2
from .ir import TimingLayout, TimingLoop
from .macro.delay import DelayRegAuto
from .modules.base import Module
from .modules.util import get_fclk
from .resources import ProgramResourcesMixin


def _bounds(value: float | QickParam) -> tuple[float, float]:
    if isinstance(value, QickParam):
        return (
            value.start + sum(min(0.0, s) for s in value.spans.values()),
            value.start + sum(max(0.0, s) for s in value.spans.values()),
        )
    return value, value


def _maximum(a: float | QickParam, b: float | QickParam, name: str):
    lower, upper = _bounds(a - b)
    if lower >= -1e-12:
        return a
    if upper <= 1e-12:
        return b
    raise ValueError(
        f"{name}: cannot prove timeline maximum at every sweep point; "
        "use delay_timeline to arrange noncrossing ends"
    )


def _align_end(prog, value: float | QickParam) -> float | QickParam:
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
    if isinstance(cycles, QickParam):
        cycles = QickParam(
            ceil_tick(cycles.start), {k: ceil_tick(v) for k, v in cycles.spans.items()}
        )
    else:
        cycles = ceil_tick(cycles)
    return _step_time(prog, cycles / get_fclk(prog))


def _step_time(prog, value: float | QickParam) -> float | QickParam:
    if not isinstance(value, QickParam):
        return value
    # Use QICK public conversion/rounding APIs, never inspect raw sweep internals.
    param = QickParam(value.start, value.spans.copy())
    prog.us2cycles(param)
    return param.get_rounded(loop_counts=prog.loop_dict)


def _repeat_minimum(name: str, n: int | str, range_hint: tuple[int, int] | None) -> int:
    """Validate the public count contract and return its guaranteed minimum."""
    if isinstance(n, str):
        if (
            not isinstance(range_hint, tuple)
            or len(range_hint) != 2
            or any(type(value) is not int or value < 0 for value in range_hint)
            or range_hint[0] > range_hint[1]
        ):
            raise ValueError("register repeat requires a valid nonnegative range_hint")
        if n == name:
            raise ValueError("repeat n must differ from its counter register name")
        return range_hint[0]
    if type(n) is not int or n < 0:
        raise ValueError("repeat n must be a nonnegative integer or register name")
    if range_hint is not None:
        raise ValueError("integer repeat does not accept range_hint")
    return n


@dataclass
class _RepeatFrame:
    parent: str
    external: dict[str, float | QickParam]
    waiting: set[str]
    native_gens: dict[int, float | QickParam]
    native_ros: dict[int, float | QickParam]
    advance: float | QickParam = 0.0
    variable: bool = False
    written_gens: set[int] = field(default_factory=set)
    written_ros: set[int] = field(default_factory=set)

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


class _TimelineBook:
    """Store only relative ends, including the ordinary default timeline."""

    def __init__(self):
        self.ends: dict[str, float | QickParam] = {"default": 0.0}
        self.waiting: set[str] = set()
        self.repeats: list[_RepeatFrame] = []
        self.native_only = False

    def end(self, name: str, *, placement: bool = False) -> float | QickParam:
        if name not in self.ends:
            raise ValueError(f"{name}: unknown timeline; declare it with fork in _body")
        if placement and name in self.waiting:
            raise ValueError(
                f"{name}: timeline can only wait after register repeat; "
                "sync it or use join/fork to establish a conservative scheduling point"
            )
        return self.ends[name]

    def decrement(self, t: float | QickParam) -> None:
        if self.native_only:
            return
        if self.repeats:
            self.repeats[-1].advance += t
        self.ends = {
            name: _maximum(0.0, end - t, name) for name, end in self.ends.items()
        }
        self.waiting = {
            name for name in self.waiting if _bounds(self.ends[name])[1] > 1e-12
        }

    def maximum(self) -> float | QickParam:
        end: float | QickParam = 0.0
        for name, candidate in self.ends.items():
            end = _maximum(end, candidate, name)
        return end


class _Fork(Macro):
    """Copy one declared end to fresh names at this preprocessing position."""

    def __init__(self, book: _TimelineBook, source: str, targets: tuple[str, ...]):
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


class _Join(Macro):
    """Merge existing ends without emitting instructions or consuming sources."""

    def __init__(self, book: _TimelineBook, sources: tuple[str, ...], into: str):
        self.book = book
        self.sources = sources
        self.into = into

    def preprocess(self, prog) -> None:
        super().preprocess(prog)
        end = self.book.end(self.into)
        for source in self.sources:
            end = _maximum(end, self.book.end(source), self.into)
        self.book.ends[self.into] = end
        self.book.waiting.discard(self.into)

    def translate(self, prog) -> None:
        pass


class _RuntimeAdvance(Macro):
    """Guard all ends against the static part of an unknown nonnegative advance."""

    def __init__(self, book: _TimelineBook, source: DelayRegAuto):
        self.book = book
        self.source = source

    def preprocess(self, prog) -> None:
        if self.book.repeats:
            raise ValueError(
                "repeat per-iteration reference advance cannot be expressed at compile time "
                "when using a runtime register delay"
            )
        ends = self.book.ends.copy()
        waiting = self.book.waiting.copy()
        # The native consumer resolves its rounded static time. Suspend only
        # timeline bookkeeping until the proof, not native channel timestamps.
        self.book.ends.clear()
        self.book.waiting.clear()
        try:
            self.source.preprocess(prog)
            static = self.source.get_time_param("auto_t")
            if static is None:
                static = 0.0
            for name, end in ends.items():
                if _bounds(end - static)[1] > 1e-12:
                    raise ValueError(
                        f"{name}: runtime register advance cannot prove end <= "
                        "static advance at every sweep point; first sync this timeline"
                    )
        except Exception:
            self.book.ends = ends
            self.book.waiting = waiting
            raise
        self.book.ends = dict.fromkeys(ends, 0.0)

    def translate(self, prog) -> None:
        self.source.translate(prog)


class _Boundary(Macro):
    """Reset preprocessing state or emit a linked IR timing boundary."""

    def __init__(self, book: _TimelineBook, section: str):
        self.book = book
        self.section = section

    def preprocess(self, prog) -> None:
        super().preprocess(prog)
        if self.section == "reset":
            self.book.ends = {"default": 0.0}
            self.book.waiting.clear()
            self.book.repeats.clear()
            self.book.native_only = False

    def translate(self, prog) -> None:
        if self.section in ("body_start", "body_end"):
            prog.record_meta(
                type="TIMING_BOUNDARY", name=self.section, info={}, regs={}
            )


class _Advance(TimedMacro):
    def __init__(self, book, timeline, duration, *, tag=None, aligned=False):
        super().__init__(tag=tag)
        self.book = book
        self.timeline = timeline
        self.duration = duration
        self.aligned = aligned

    def preprocess(self, prog) -> None:
        duration = (
            _step_time(prog, self.duration)
            if self.aligned
            else _align_end(prog, self.duration)
        )
        end = self.book.end(self.timeline, placement=True) + duration
        self.book.ends[self.timeline] = _maximum(0.0, end, self.timeline)
        self.resolved_time = end if isinstance(end, QickParam) else QickParam(end)

    def list_time_params(self) -> list[str]:
        return ["t"]

    def get_time_param(self, name: str) -> QickParam:
        if name != "t":
            raise RuntimeError("timeline advances expose only the t parameter")
        return self.resolved_time

    def translate(self, prog) -> None:
        pass


class _Sync(Delay):
    def __init__(self, book: _TimelineBook, timeline: str | None):
        super().__init__(t=0, auto=False)
        self.book = book
        self.timeline = timeline

    def preprocess(self, prog) -> None:
        self.t = (
            self.book.maximum()
            if self.timeline is None
            else self.book.end(self.timeline)
        )
        super().preprocess(prog)


class _RepeatEnter(_Sync):
    """Synchronize the parent, then isolate this iteration's local namespace."""

    def __init__(self, book: _TimelineBook, timeline: str, counter: str | None = None):
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
        self.book.repeats.append(
            _RepeatFrame(
                parent,
                external,
                self.book.waiting.copy(),
                {ch: prog.get_timestamp(gen_ch=ch) for ch in prog.gen_chs},
                {ch: prog.get_timestamp(ro_ch=ch) for ch in prog.ro_chs},
            )
        )
        self.book.ends = {parent: 0.0}
        self.book.waiting.clear()


class _RepeatExit(_Sync):
    """Close one iteration, then rebase the outer namespace by all iterations."""

    def __init__(self, book: _TimelineBook, count: int, *, variable: bool = False):
        super().__init__(book, None)
        self.count = count
        self.variable = variable

    def preprocess(self, prog) -> None:
        super().preprocess(prog)
        frame = self.book.repeats.pop()
        variable = self.variable or frame.variable
        if variable and _bounds(frame.advance)[0] < -1e-12:
            raise ValueError(
                "repeat cannot prove minimum reference advance at every sweep point"
            )
        # Rebase native timestamps separately so external ends undergo exactly
        # one pointwise floor proof and last-iteration output tails are preserved.
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
        self.book.decrement(frame.advance * self.count)
        if variable:
            self.book.waiting.update(
                name for name, end in self.book.ends.items() if _bounds(end)[1] > 1e-12
            )
            if self.book.repeats:
                self.book.repeats[-1].variable = True
        self.book.ends[frame.parent] = 0.0


class _PlacedPulse(Pulse):
    def __init__(self, book: _TimelineBook, timeline: str, source: Pulse):
        super().__init__(ch=source.ch, name=source.name, tag=source.tag)
        self.book = book
        self.timeline = timeline
        self.source = source
        self.relative_t: float | QickParam | Literal["auto"] = source.t

    def preprocess(self, prog) -> None:
        end = self.book.end(self.timeline, placement=True)
        if self.relative_t == "auto":
            t = _maximum(end, prog.get_timestamp(gen_ch=self.ch), self.timeline)
        else:
            t = end + self.relative_t
        previous_end = prog.get_timestamp(gen_ch=self.ch)
        self.source.t = t
        self.source.preprocess(prog)
        actual_t = self.source.get_time_param("t")
        if _bounds(actual_t - previous_end)[0] < -1e-12:
            raise ValueError(
                f"{self.timeline}: channel {self.ch} pulse overlap; cannot prove "
                "start >= previous end at every sweep point; use delay_timeline"
            )
        self.book.ends[self.timeline] = _align_end(
            prog, actual_t + prog.pulses[self.name].get_length()
        )

    def translate(self, prog) -> None:
        self.source.translate(prog)

    def get_time_param(self, name: str) -> QickParam:
        return self.source.get_time_param(name)

    def list_time_params(self) -> list[str]:
        return self.source.list_time_params()


class _PlacedConfig(ConfigReadout):
    def __init__(self, book: _TimelineBook, timeline: str, source: ConfigReadout):
        super().__init__(ch=source.ch, name=source.name, tag=source.tag)
        self.book = book
        self.timeline = timeline
        self.source = source
        self.relative_t: float | QickParam = source.t

    def preprocess(self, prog) -> None:
        self.source.t = self.book.end(self.timeline, placement=True) + self.relative_t
        self.source.preprocess(prog)
        self.book.ends[self.timeline] = _maximum(
            0.0, self.source.get_time_param("t"), self.timeline
        )

    def translate(self, prog) -> None:
        self.source.translate(prog)

    def get_time_param(self, name: str) -> QickParam:
        return self.source.get_time_param(name)

    def list_time_params(self) -> list[str]:
        return self.source.list_time_params()


class _PlacedTrigger(Trigger):
    def __init__(self, book: _TimelineBook, timeline: str, source: Trigger):
        super().__init__(
            ros=source.ros,
            tts=source.tts,
            pins=source.pins,
            width=source.width,
            ddr4=source.ddr4,
            mr=source.mr,
            tag=source.tag,
        )
        self.book = book
        self.timeline = timeline
        self.relative_t: float | QickParam = source.t

    def preprocess(self, prog) -> None:
        self.t = self.book.end(self.timeline, placement=True) + self.relative_t
        super().preprocess(prog)
        end = self.get_time_param("t") + self.get_time_param("width")
        for ch in self.ros:
            end = _maximum(end, prog.get_timestamp(ro_ch=ch), self.timeline)
        self.book.ends[self.timeline] = _align_end(prog, end)


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
            if isinstance(macro, (_PlacedPulse, _PlacedConfig, _PlacedTrigger)):
                start = macro.book.end(macro.timeline)
                macro.preprocess(prog)
                macro.book.ends[macro.timeline] = start
            else:
                macro.preprocess(prog)

    def translate(self, prog) -> None:
        """Translate lowered operations through the program's shared compiler."""
        for macro in self.macros:
            macro.translate(prog)

    def get_time_param(self, name: str) -> QickParam:
        """Return first-event timing; absent events/parameters raise RuntimeError."""
        for macro in self.macros:
            if isinstance(macro, TimedMacro):
                return macro.get_time_param(name)
        raise RuntimeError("this emission has no timed operations")

    def list_time_params(self) -> list[str]:
        """List first-event parameter names, or [] for an untimed emission."""
        for macro in self.macros:
            if isinstance(macro, TimedMacro):
                return macro.list_time_params()
        return []


class TimelineProgramV2(MyProgramV2, ProgramResourcesMixin):
    """Author native QICK hooks with optional timeline ends (times in us).

    Constructor follows MyProgramV2: reps, initial_delay, relax_delay. Native
    operations retain QICK timing. Named outputs update their relative ends;
    sync and body completion move reference time. Each compile resets ends.
    All instructions share the existing IR and acquisition implementations.
    Simulation frontend migration is separate.
    """

    def __init__(self, *args, **kwargs):
        self._timeline_book = _TimelineBook()
        self._declared_modules: list[Module] = []
        self._declaring = False
        self._building_body = False
        super().__init__(*args, **kwargs)

    def make_program(self) -> None:
        """Build the QICK shell with linked body markers and all-timeline closure.

        Body closure precedes body_end and final_wait/final_delay. Loop hooks,
        counters, cleanup and subroutines retain AveragerProgramV2's order.
        """
        # QICK 0.2.436 provides no public shell boundary hooks.
        self._reset_resources()
        self._declared_modules.clear()
        self.append_macro(_Boundary(self._timeline_book, "reset"))
        self.set_ext_counter(addr=self.COUNTER_ADDR)
        self._declaring = True
        try:
            self._initialize(None)
        finally:
            self._declaring = False
        self.append_macro(_Boundary(self._timeline_book, "initialize_end"))
        self.delay_auto(self.initial_delay)
        for name, count, before, _after in self.loops:
            self.open_loop(count, name=name)
            if before is not None:
                self.extend_macros(before)
        self.append_macro(_Boundary(self._timeline_book, "body_start"))
        self._building_body = True
        try:
            self._body(None)
        finally:
            self._building_body = False
        self.append_macro(_Sync(self._timeline_book, None))
        self.append_macro(_Boundary(self._timeline_book, "body_end"))
        self.wait_auto(self.final_wait, no_warn=True)
        if self.final_delay is not None:
            self.delay_auto(self.final_delay)
        self.inc_ext_counter(addr=self.COUNTER_ADDR)
        for _name, _count, _before, after in self.loops[::-1]:
            if after is not None:
                self.extend_macros(after)
            self.close_loop()
        self._cleanup(None)
        self.end()
        for name, asm in self.subroutines.items():
            self.label(name)
            self.extend_macros(asm)
            self.ret()

    @property
    def timing_layout(self) -> TimingLayout:
        """Return linked body addresses and outer-to-inner axes after compile.

        Missing or duplicate compiler markers raise ValueError. Body end includes
        timeline closure but precedes shell timing; shell timing belongs to shot.
        """
        return TimingLayout.from_metadata(
            self.meta_infos,
            tuple(TimingLoop(name, count) for name, count, _, _ in self.loops),
        )

    def declare(self, module: Module) -> None:
        """Declare a borrowed module in _initialize; repeat declarations are harmless.

        Do not mutate it afterward. Outside _initialize raises ValueError;
        unsupported module types raise NotImplementedError.
        """
        if not self._declaring:
            raise ValueError("declare must be called in _initialize")
        if any(module is declared for declared in self._declared_modules):
            return
        module.declare(self)
        self._declared_modules.append(module)

    def emit(
        self, module: Module, *, timeline: str | None = None, tag: str | None = None
    ) -> None:
        """Emit a declared module once, with a fresh timing/tag occurrence.

        Omitted timeline selects ordinary default, not reference time. Internal
        events share its current end; rule-a duration advances that end only.
        DelayAuto is native reference advance and rejects explicit timelines.
        Undeclared objects raise ValueError; owner lowering failures propagate.
        """
        if not any(module is declared for declared in self._declared_modules):
            raise ValueError(f"{module.name}: module must be declared before emit")
        start = len(self.macro_list)
        try:
            module.lower(self, timeline=timeline, tag=tag)
            macros = self.macro_list[start:]
        finally:
            del self.macro_list[start:]
        self.append_macro(ModuleEmission(module, macros, tag))

    def advance_module(
        self,
        duration: float | QickParam,
        *,
        timeline: str | None,
        tag: str | None = None,
    ) -> None:
        """Add an owner's already tProc-aligned rule-a duration in us to its end.

        None selects default. Unlike arbitrary delay_timeline inputs, this value
        is not ceiled again. tag names the resolved end via get_time_param.
        """
        self.append_macro(
            _Advance(
                self._timeline_book,
                timeline if timeline is not None else "default",
                duration,
                tag=tag,
                aligned=True,
            )
        )

    def delay_timeline(
        self, t: float | QickParam, *, timeline: str = "default", tag: str | None = None
    ) -> None:
        """Add t us (ceiled to tProc cycles) to a timeline without moving reference.

        tag exposes the resulting end as parameter 't'. Crossing zero or an
        unprovable sweep floor raises ValueError during compile.
        """
        self.append_macro(_Advance(self._timeline_book, timeline, t, tag=tag))

    def set_timestamp(
        self,
        val: float | QickParam,
        gen_ch: int | None = None,
        ro_ch: int | None = None,
    ) -> None:
        """Set a native channel timestamp in us using QICK's channel validation.

        During repeat preprocessing, record channels with newly scheduled events.
        Their last-iteration tails must not receive the external-duration debit.
        """
        super().set_timestamp(val, gen_ch=gen_ch, ro_ch=ro_ch)
        if self._timeline_book.repeats and not self._timeline_book.native_only:
            frame = self._timeline_book.repeats[-1]
            if gen_ch is not None:
                frame.written_gens.add(gen_ch)
            if ro_ch is not None:
                frame.written_ros.add(ro_ch)

    def decrement_timestamps(self, t: float | QickParam | None) -> None:
        """Rebase all relative ends when QICK moves reference by t us.

        None is native auto with no selected channels and changes no ends.
        A sweep must prove a uniform sign for each zero floor, otherwise compile
        raises ValueError naming the timeline and suggesting delay_timeline.
        """
        super().decrement_timestamps(t)
        if t is not None:
            self._timeline_book.decrement(t)

    def delay_reg_auto(self, time_reg: str, *, gens=True, ros=True) -> None:
        """Advance native reference by selected static ends plus register cycles.

        time_reg contains nonnegative tProc cycles. Every timeline, including
        default, must prove end <= static advance at every sweep point, otherwise
        compile raises ValueError suggesting sync. Proven ends become zero.
        """
        super().delay_reg_auto(time_reg, gens=gens, ros=ros)
        source = self.macro_list.pop()
        self.append_macro(_RuntimeAdvance(self._timeline_book, source))

    def fork(self, source: str, *, into: str | list[str]) -> None:
        """Declare new timeline names in _body, copying source end at this point.

        source must already exist; into is one new name or a list of distinct
        new names. A wait-only source supplies its latest bound; new timelines
        permit conservative placement while that source stays wait-only.
        Unknown source or duplicate targets raise ValueError during
        compile. Calling outside _body raises ValueError immediately. This emits
        no instructions and does not advance reference. Each compile resets names.
        """
        if not self._building_body:
            raise ValueError("fork must be called in _body")
        targets = (into,) if isinstance(into, str) else tuple(into)
        self.append_macro(_Fork(self._timeline_book, source, targets))

    @contextmanager
    def repeat(
        self,
        name: str,
        n: int | str,
        *,
        timeline: str = "default",
        range_hint: tuple[int, int] | None = None,
    ) -> Generator[None]:
        """Repeat a body n times on parent timeline (times remain in us).

        Entry syncs the parent. Each iteration closes parent and local ends;
        exit discards local names and leaves parent zero. Fixed counts exactly
        rebase outer ends. Register counts rebase by range minimum and leave
        nonzero outer ends wait-only: join/fork establish ordinary conservative
        scheduling points; full reference consumption restores placement. Only
        parent and locally forked names may be used inside. name is
        the native loop/counter name. n is a nonnegative integer or existing
        register name. Register counts require an author-guaranteed closed
        nonnegative integer range_hint; integer counts forbid range_hint. No
        hardware range check is added. Invalid counts/ranges raise ValueError;
        unknown/inaccessible names fail during compile. Zero counts still sync
        the parent but discard body operations. Runtime register reference delays
        inside the body raise ValueError because iteration duration is unknown.
        """
        minimum = _repeat_minimum(name, n, range_hint)
        if n == 0:
            self.sync(timeline=timeline)
            start = len(self.macro_list)
            try:
                yield
            finally:
                del self.macro_list[start:]
            return
        variable = isinstance(n, str)
        self.append_macro(
            _RepeatEnter(self._timeline_book, timeline, name if variable else None)
        )
        if isinstance(n, str):
            self.open_inner_loop(name, name, n, range_hint=range_hint)
        else:
            self.open_loop(n, name=name)
        try:
            yield
        finally:
            self.append_macro(
                _RepeatExit(self._timeline_book, minimum, variable=variable)
            )
            if isinstance(n, str):
                self.close_inner_loop(name, name, n)
            else:
                self.close_loop()

    def join(self, *sources: str, into: str) -> None:
        """Merge existing sources and target into target's latest end (us).

        Sources remain unchanged. Wait-only sources contribute their latest
        bounds; the target becomes ordinary and permits conservative placement.
        This emits no instructions and does not move reference. Unknown names
        or crossing swept ends raise ValueError
        during compile; declare names with fork before joining them.
        """
        self.append_macro(_Join(self._timeline_book, sources, into))

    def sync(self, *, timeline: str = "default") -> None:
        """Move reference to the chosen end and floor every rebased end at zero.

        Only default exists initially; other names require fork in _body.
        Fully consumed wait-only bounds become ordinary zero ends again.
        Unknown names or unprovable swept floors raise ValueError during compile.
        """
        self.append_macro(_Sync(self._timeline_book, timeline))

    def send_readoutconfig(
        self, ch: int, name: str, *args, timeline: str | None = None, **kwargs
    ) -> None:
        """Send an instantaneous config at timeline end+t us, or native t if omitted.

        Native args are t=0, tag=None. A timeline end becomes the scheduled config
        instant; config length and unrelated ADC windows do not contribute.
        Wait-only timelines reject placement even at t=0; first sync the bound
        or establish an ordinary scheduling point with join/fork.
        Declaration and timing errors propagate.
        """
        super().send_readoutconfig(ch, name, *args, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            self.append_macro(_PlacedConfig(self._timeline_book, timeline, source))

    def pulse_from_regs(
        self, ch: int, name: str, *, timeline: str | None = None, **kwargs
    ) -> None:
        """Play runtime words at native t or timeline end+t, advancing a named end.

        kwargs are t (us), tag, freq_reg and gain_reg containing final hardware
        words. Native patching and template/register validation are unchanged.
        """
        super().pulse_from_regs(ch, name, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            self.append_macro(_PlacedPulse(self._timeline_book, timeline, source))

    def send_readoutconfig_from_regs(
        self, ch: int, name: str, *, timeline: str | None = None, **kwargs
    ) -> None:
        """Send runtime ADC words at native t or timeline end+t (us).

        kwargs are t, tag and freq_reg. A timeline end becomes the scheduled config
        instant, independent of config length. Wait-only timelines reject placement
        even at t=0; first sync the bound or establish an ordinary scheduling
        point with join/fork. Native patching and declaration/register validation
        are unchanged.
        """
        super().send_readoutconfig_from_regs(ch, name, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            self.append_macro(_PlacedConfig(self._timeline_book, timeline, source))

    def trigger(self, *args, timeline: str | None = None, **kwargs) -> None:
        """Trigger ADC/tt/pins at native t or timeline end+t and advance named end.

        Native args select ros/tts/pins, t, width (us), ddr4/mr and tag. Named end
        includes ADC integration and trigger width, rounded up to tProc cycles.
        Named t=None raises ValueError at entry; native special values are unchanged.
        """
        super().trigger(*args, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            if source.t is None:
                raise ValueError("timeline trigger does not support t=None")
            self.append_macro(_PlacedTrigger(self._timeline_book, timeline, source))

    def pulse(
        self, ch: int, name: str, *args, timeline: str | None = None, **kwargs
    ) -> None:
        """Play a pulse at native t or timeline end+t, advancing a named end.

        Native args are t=0 (us or 'auto'), tag=None. Named 'auto' takes the
        provably later timeline/channel end. Named end includes waveform length,
        rounded up to tProc cycles. Crossing swept maxima raise ValueError with
        a delay_timeline hint; native declaration errors propagate.
        """
        super().pulse(ch, name, *args, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            self.append_macro(_PlacedPulse(self._timeline_book, timeline, source))
