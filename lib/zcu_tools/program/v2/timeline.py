"""Native QICK programs with independent named scheduling cursors."""

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
from .macro.delay import DelayRegAuto
from .modules.base import Module
from .modules.util import get_fclk, round_timestamp
from .resources import ProgramResourcesMixin


@dataclass
class _TimelineState:
    offset: float | QickParam = 0.0
    gen_ends: list[float | QickParam] = field(default_factory=list)
    ro_ends: list[float | QickParam] = field(default_factory=list)


class _TimelineBook:
    """Keep preprocessing state shared by all macros in one program."""

    def __init__(self):
        self.states: dict[str, _TimelineState] = {}

    def state(self, name: str) -> _TimelineState:
        return self.states.setdefault(name, _TimelineState())

    def decrement(self, t: float | QickParam) -> None:
        for state in self.states.values():
            state.offset = max(0.0, state.offset - t)
            state.gen_ends = [max(0.0, end - t) for end in state.gen_ends]
            state.ro_ends = [max(0.0, end - t) for end in state.ro_ends]


class _RuntimeAdvance(Macro):
    """Guard unknown nonnegative runtime advances without adding a runtime cursor."""

    def __init__(self, book: _TimelineBook, source: DelayRegAuto):
        self.book = book
        self.source = source

    def preprocess(self, prog) -> None:
        offsets = {name: state.offset for name, state in self.book.states.items()}
        has_end = (
            prog.get_max_timestamp(gens=self.source.gens, ros=self.source.ros)
            is not None
        )
        self.source.preprocess(prog)
        static = self.source.get_time_param("auto_t") if has_end else 0.0
        for name, offset in offsets.items():
            difference = offset - static
            if isinstance(difference, QickParam):
                # QickParam's ordinary range comparator is not pointwise for
                # independent sweeps. An affine maximum includes every positive span.
                upper = difference.start + sum(
                    max(0.0, span) for span in difference.spans.values()
                )
            else:
                upper = difference
            if not upper <= 0:
                raise ValueError(
                    f"{name}: runtime register advance cannot prove offset <= "
                    "static advance; use sync(timeline=...) or delay_auto(timeline=...)"
                )
        for state in self.book.states.values():
            state.offset = 0.0

    def translate(self, prog) -> None:
        self.source.translate(prog)


class _Boundary(Macro):
    """Preprocess-only lifecycle marker; never occupies instruction memory."""

    def __init__(self, book: _TimelineBook, section: str):
        self.book = book
        self.section = section

    def preprocess(self, prog) -> None:
        if self.section == "reset":
            self.book.states.clear()
        super().preprocess(prog)

    def translate(self, prog) -> None:
        pass


class _Advance(TimedMacro):
    def __init__(
        self,
        book: _TimelineBook,
        timeline: str,
        source: Delay,
        *,
        aligned: bool = False,
    ):
        super().__init__(tag=source.tag)
        self.book = book
        self.timeline = timeline
        self.source = source
        self.aligned = aligned

    def preprocess(self, prog) -> None:
        state = self.book.state(self.timeline)
        ends = [state.offset]
        if self.source.auto:
            if self.source.gens:
                ends.extend(state.gen_ends)
            if self.source.ros:
                ends.extend(state.ro_ends)
        offset = max(ends) + self.source.t
        if self.aligned:
            # Both the cursor and owner duration are already integral tProc ticks.
            # Recover that integer, rather than ceiling multiplication noise again.
            cycles = offset * get_fclk(prog)
            if isinstance(cycles, QickParam):
                cycles.start = round(cycles.start)
                cycles.spans = {
                    name: round(span) for name, span in cycles.spans.items()
                }
            else:
                cycles = round(cycles)
            offset = cycles / get_fclk(prog)
        else:
            offset = round_timestamp(prog, offset)
        state.offset = offset
        self.resolved_time = (
            offset if isinstance(offset, QickParam) else QickParam(offset)
        )

    def list_time_params(self) -> list[str]:
        return ["t"]

    def get_time_param(self, name: str) -> QickParam:
        if name != "t":
            raise RuntimeError("named advances expose only the t parameter")
        return self.resolved_time

    def translate(self, prog) -> None:
        pass


class _Sync(Delay):
    def __init__(self, book: _TimelineBook, timeline: str):
        super().__init__(t=0, auto=False)
        self.book = book
        self.timeline = timeline

    def preprocess(self, prog) -> None:
        self.t = self.book.state(self.timeline).offset
        super().preprocess(prog)


class _PlacedPulse(Pulse):
    def __init__(self, book: _TimelineBook, timeline: str, source: Pulse):
        super().__init__(ch=source.ch, name=source.name, tag=source.tag)
        self.book = book
        self.timeline = timeline
        self.source = source
        self.relative_t: float | QickParam | Literal["auto"] = source.t

    def preprocess(self, prog) -> None:
        state = self.book.state(self.timeline)
        if self.relative_t == "auto":
            self.t = max(state.offset, prog.get_timestamp(gen_ch=self.ch))
        else:
            self.t = state.offset + self.relative_t
        self.source.t = self.t
        self.source.preprocess(prog)
        state.gen_ends.append(prog.get_timestamp(gen_ch=self.ch))

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
        state = self.book.state(self.timeline)
        self.t = state.offset + self.relative_t
        self.source.t = self.t
        self.source.preprocess(prog)
        state.ro_ends.append(self.t)

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
        state = self.book.state(self.timeline)
        self.t = state.offset + self.relative_t
        super().preprocess(prog)
        state.ro_ends.extend(prog.get_timestamp(ro_ch=ch) for ch in self.ros)


class ModuleEmission(TimedMacro):
    """One declared module's emission, with independent native timing macros.

    owner is the declaration object. macros are its lowered native operations in
    execution order. tag is this occurrence's optional public timing name; timing
    parameters delegate to the first native timed operation. Module objects are
    never themselves QICK macros. Construction is owned by TimelineProgramV2.emit.
    """

    def __init__(self, owner: Module, macros: list[Macro], tag: str | None):
        super().__init__(tag=tag)
        self.owner = owner
        self.macros = macros

    def preprocess(self, prog) -> None:
        """Resolve this occurrence's native timing on the compiling program."""
        for macro in self.macros:
            macro.preprocess(prog)

    def translate(self, prog) -> None:
        """Translate lowered operations through the program's shared compiler."""
        for macro in self.macros:
            macro.translate(prog)

    def get_time_param(self, name: str) -> QickParam:
        """Return first-event timing by name; missing events/parameters raise RuntimeError."""
        for macro in self.macros:
            if isinstance(macro, TimedMacro):
                return macro.get_time_param(name)
        raise RuntimeError("this emission has no timed operations")

    def list_time_params(self) -> list[str]:
        """List first-event parameter names, or an empty list for an untimed emission."""
        for macro in self.macros:
            if isinstance(macro, TimedMacro):
                return macro.list_time_params()
        return []


class TimelineProgramV2(MyProgramV2, ProgramResourcesMixin):
    """Author native QICK hooks with optional named timelines (times in us).

    Constructor controls follow MyProgramV2: reps, initial_delay and relax_delay.
    Author hooks follow QICK. All instructions use the shared IR compiler.
    Named scheduling is resolved during preprocessing, not in author hooks.
    Each compilation starts with empty named cursors/ends. Unnamed operations
    retain native timing semantics. Simulation frontend support is separate.
    """

    def __init__(self, *args, **kwargs):
        self._timeline_book = _TimelineBook()
        self._declared_modules: list[Module] = []
        self._declaring = False
        super().__init__(*args, **kwargs)

    def make_program(self) -> None:
        """Build the native QICK shell with instruction-free lifecycle markers.

        Loop order, hooks, counters, final wait/delay and subroutines follow
        AveragerProgramV2. Markers reset cursors at preprocess and identify the
        initialization/body boundaries without changing native instructions.
        """
        # QICK 0.2.436 has no public boundary hooks. Keep its shell order here
        # so markers surround only author initialization/body, not final timing.
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
        self._body(None)
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

    def declare(self, module: Module) -> None:
        """Declare module resources explicitly in _initialize.

        The object is borrowed and must not be mutated after declaration.
        Re-declaring the same object is harmless. Calling outside _initialize
        raises ValueError. Unsupported module types raise NotImplementedError.
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
        """Emit a declared module once and advance its timeline in us.

        None means the real QICK reference; a name means bookkeeping only.
        Every call creates a fresh ModuleEmission. tag identifies this occurrence
        for get_time_param, independently of legacy module tags/block settings.
        Undeclared objects raise ValueError; owner declaration/lowering failures
        propagate. Unsupported modules raise NotImplementedError.
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
        """Advance by an owner's already tProc-aligned total_length in us.

        Owner lowering must supply an integral number of tProc ticks, including
        each sweep step. None emits a native reference delay; a name adds aligned
        ticks without rounding them upward a second time. Named native delay
        remains the entry point for arbitrary, unrounded durations. tag optionally
        names this advance for public timing inspection.
        """
        if timeline is None:
            self.delay(duration, tag=tag)
        else:
            self.append_macro(
                _Advance(
                    self._timeline_book,
                    timeline,
                    Delay(t=duration, auto=False, tag=tag),
                    aligned=True,
                )
            )

    def decrement_timestamps(self, t: float | QickParam | None) -> None:
        """Rebase QICK channels and named cursors/ends by t us, clamping at zero.

        QICK invokes this during preprocessing of native delay/auto/resync.
        None denotes native auto timing with no channels and leaves cursors
        unchanged. Sweeps use QICK range comparisons, not elementwise maxima.
        """
        super().decrement_timestamps(t)
        if t is not None:
            self._timeline_book.decrement(t)

    # Let QICK bind native arguments and construct each macro. Replace only
    # named calls through public macro_list, preserving native positional calls.
    def delay(self, *args, timeline: str | None = None, **kwargs) -> None:
        """Advance reference time, or a named cursor, by t us.

        args/kwargs are QICK delay(t, tag=None). Named advances round the final
        cursor up to a tProc cycle and emit no instructions. For a named tag,
        get_time_param(tag, 't') reports that resolved cursor. Omit timeline
        for native delay/tag semantics.
        """
        super().delay(*args, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            self.append_macro(_Advance(self._timeline_book, timeline, source))

    def delay_auto(self, *args, timeline: str | None = None, **kwargs) -> None:
        """Align reference time, or a named cursor, to selected ends plus t us.

        args/kwargs are QICK delay_auto(t=0, gens=True, ros=True, tag=None).
        Named alignment takes max(cursor, all prior generator/readout ends on
        that timeline), filtered by gens/ros. It rounds only the final offset
        up to a tProc cycle and emits no instructions. Ends survive advances.
        Maxima and zero floors use QICK QickParam comparison semantics (range
        comparisons, not elementwise maxima for crossing sweeps). Named tags
        report the resolved cursor via get_time_param(tag, 't'). Without
        timeline, retain native timing and tag behavior. A register-name t uses
        delay_reg_auto and must contain nonnegative tProc cycles. Runtime register
        time rejects named timelines and tags at entry; default compilation
        requires the named-cursor proof described by delay_reg_auto.
        """
        super().delay_auto(*args, **kwargs)
        source = self.macro_list[-1]
        if isinstance(source.t, str):
            self.macro_list.pop()
            if timeline is not None:
                raise ValueError(
                    "runtime register time supports only the default timeline"
                )
            if source.tag is not None:
                raise ValueError("runtime register time cannot have a tag")
            self.delay_reg_auto(source.t, gens=source.gens, ros=source.ros)
        elif timeline is not None:
            self.macro_list.pop()
            self.append_macro(_Advance(self._timeline_book, timeline, source))

    def delay_reg_auto(self, time_reg: str, *, gens=True, ros=True) -> None:
        """Advance the default reference by selected static ends plus register cycles.

        time_reg must hold nonnegative tProc cycles. Each named cursor must be
        provably <= the static advance at every sweep point, otherwise compilation
        raises ValueError with its name and a sync/delay_auto hint. Proven cursors
        become zero; channel ends retain only the static decrement. QICK register
        validation is unchanged. No runtime offset or extra instruction is added.
        """
        super().delay_reg_auto(time_reg, gens=gens, ros=ros)
        source = self.macro_list.pop()
        self.append_macro(_RuntimeAdvance(self._timeline_book, source))

    def wait_auto(self, *args, **kwargs) -> None:
        """Wait using native QICK arguments; timeline= is unsupported.

        args/kwargs are wait_auto(t=0, gens=False, ros=True, tag=None,
        no_warn=False). Times are in us; selection includes all channel
        timestamps, even named emissions. Unsupported keywords raise TypeError.
        """
        super().wait_auto(*args, **kwargs)

    def sync(self, *, timeline: str) -> None:
        """Move reference time to the named cursor using one native inc_ref.

        All named cursors/ends and native channel timestamps are rebased, with
        zero as their lower bound. A previously unused timeline starts at zero.
        """
        self.append_macro(_Sync(self._timeline_book, timeline))

    def send_readoutconfig(
        self, ch: int, name: str, *args, timeline: str | None = None, **kwargs
    ) -> None:
        """Send config name on ADC ch, optionally relative to a named cursor.

        args/kwargs are QICK send_readoutconfig's t=0 (us) and tag=None.
        Named configuration is instantaneous: its end is offset+t, regardless
        of config length, and is selected by ros in named delay_auto. It does
        not advance the cursor or borrow an ADC integration window. Without
        timeline, retain QICK configuration/timing/tag behavior.
        """
        super().send_readoutconfig(ch, name, *args, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            self.append_macro(_PlacedConfig(self._timeline_book, timeline, source))

    def pulse_from_regs(
        self, ch: int, name: str, *, timeline: str | None = None, **kwargs
    ) -> None:
        """Play a runtime-word pulse, optionally on a named timeline.

        kwargs are the inherited t (us), tag, freq_reg and gain_reg. Registers
        contain final hardware words. Template validation and QICK failures
        propagate. Named playback retains the runtime patch before port emission.
        """
        super().pulse_from_regs(ch, name, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            self.append_macro(_PlacedPulse(self._timeline_book, timeline, source))

    def send_readoutconfig_from_regs(
        self, ch: int, name: str, *, timeline: str | None = None, **kwargs
    ) -> None:
        """Send runtime ADC words, optionally relative to a named cursor.

        kwargs are inherited t (us), tag and freq_reg. The named config remains
        an instantaneous ros event; runtime patching is retained. Invalid
        templates/registers propagate QICK errors.
        """
        super().send_readoutconfig_from_regs(ch, name, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            self.append_macro(_PlacedConfig(self._timeline_book, timeline, source))

    def trigger(self, *args, timeline: str | None = None, **kwargs) -> None:
        """Pulse ADC/time-tagger/pin triggers using native QICK arguments.

        args/kwargs are trigger(ros=None, tts=None, pins=None, t=0, width=None,
        ddr4=False, mr=False, tag=None). ros/tts/pins select logical channels,
        width is in us, ddr4/mr select capture buffers, and tag names the event.
        With timeline, t is relative to its cursor; ADC integration ends are
        recorded for ros alignment without advancing the cursor. Named t=None
        raises ValueError at entry. Without timeline, native s_out_time
        semantics are retained. Native declaration/argument failures propagate.
        """
        super().trigger(*args, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            if source.t is None:
                raise ValueError("named trigger does not support t=None")
            self.append_macro(_PlacedTrigger(self._timeline_book, timeline, source))

    def pulse(
        self, ch: int, name: str, *args, timeline: str | None = None, **kwargs
    ) -> None:
        """Emit a declared pulse, optionally relative to a named cursor.

        args/kwargs are QICK pulse's t=0 (us or 'auto') and tag=None.
        Named t='auto' selects max(cursor, generator timestamp), using QICK
        QickParam range comparison semantics. Named emission does not advance
        the cursor. Channel/pulse checks, dummypulse creation and tags retain
        QICK behavior; omitted timeline is entirely native.
        """
        super().pulse(ch, name, *args, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            self.append_macro(_PlacedPulse(self._timeline_book, timeline, source))
