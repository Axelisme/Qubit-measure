"""Native QICK programs with timeline ends relative to reference time."""

import math
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


class _TimelineBook:
    """Store only relative ends, including the ordinary default timeline."""

    def __init__(self):
        self.ends: dict[str, float | QickParam] = {"default": 0.0}

    def end(self, name: str) -> float | QickParam:
        if name not in self.ends:
            raise ValueError(f"{name}: unknown timeline; declare it with fork in _body")
        return self.ends[name]

    def decrement(self, t: float | QickParam) -> None:
        self.ends = {
            name: _maximum(0.0, end - t, name) for name, end in self.ends.items()
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
            name in self.book.ends for name in self.targets
        ):
            raise ValueError("fork requires new, distinct timeline names")
        self.book.ends.update(dict.fromkeys(self.targets, end))

    def translate(self, prog) -> None:
        pass


class _RuntimeAdvance(Macro):
    """Guard all ends against the static part of an unknown nonnegative advance."""

    def __init__(self, book: _TimelineBook, source: DelayRegAuto):
        self.book = book
        self.source = source

    def preprocess(self, prog) -> None:
        ends = self.book.ends.copy()
        # The native consumer resolves its rounded static time. Suspend only
        # timeline bookkeeping until the proof, not native channel timestamps.
        self.book.ends.clear()
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
        end = self.book.end(self.timeline) + duration
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


class _PlacedPulse(Pulse):
    def __init__(self, book: _TimelineBook, timeline: str, source: Pulse):
        super().__init__(ch=source.ch, name=source.name, tag=source.tag)
        self.book = book
        self.timeline = timeline
        self.source = source
        self.relative_t: float | QickParam | Literal["auto"] = source.t

    def preprocess(self, prog) -> None:
        end = self.book.end(self.timeline)
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
        self.source.t = self.book.end(self.timeline) + self.relative_t
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
        self.t = self.book.end(self.timeline) + self.relative_t
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
        new names. Unknown source or duplicate targets raise ValueError during
        compile. Calling outside _body raises ValueError immediately. This emits
        no instructions and does not advance reference. Each compile resets names.
        """
        if not self._building_body:
            raise ValueError("fork must be called in _body")
        targets = (into,) if isinstance(into, str) else tuple(into)
        self.append_macro(_Fork(self._timeline_book, source, targets))

    def sync(self, *, timeline: str = "default") -> None:
        """Move reference to the chosen end and floor every rebased end at zero.

        Only default exists initially; other names require fork in _body.
        Unknown names or unprovable swept floors raise ValueError during compile.
        """
        self.append_macro(_Sync(self._timeline_book, timeline))

    def send_readoutconfig(
        self, ch: int, name: str, *args, timeline: str | None = None, **kwargs
    ) -> None:
        """Send an instantaneous config at timeline end+t us, or native t if omitted.

        Native args are t=0, tag=None. A timeline end becomes the scheduled config
        instant; config length and unrelated ADC windows do not contribute.
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
        instant, independent of config length. Native patching and declaration/
        register validation are unchanged.
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
