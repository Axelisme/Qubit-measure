"""Native QICK programs with independent named scheduling cursors."""

from dataclasses import dataclass, field
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import (
    AveragerProgramV2,
    ConfigReadout,
    Delay,
    Macro,
    Pulse,
    QickParam,
    TimedMacro,
    Trigger,
)

from .modules.util import round_timestamp
from .resources import ProgramResourcesMixin


class _BinaryProgram(TypedDict):
    """QICK public compiled memories: packed pmem/wmem and optional int32 dmem."""

    pmem: list[list[int]]
    wmem: list[list[int]] | None
    dmem: NDArray[np.int32] | None


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
    def __init__(self, book: _TimelineBook, timeline: str, source: Delay):
        super().__init__(tag=source.tag)
        self.book = book
        self.timeline = timeline
        self.source = source

    def preprocess(self, prog) -> None:
        state = self.book.state(self.timeline)
        ends = [state.offset]
        if self.source.auto:
            if self.source.gens:
                ends.extend(state.gen_ends)
            if self.source.ros:
                ends.extend(state.ro_ends)
        offset = round_timestamp(prog, max(ends) + self.source.t)
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
        self.relative_t: float | QickParam | Literal["auto"] = source.t

    def preprocess(self, prog) -> None:
        state = self.book.state(self.timeline)
        if self.relative_t == "auto":
            self.t = max(state.offset, prog.get_timestamp(gen_ch=self.ch))
        else:
            self.t = state.offset + self.relative_t
        super().preprocess(prog)
        state.gen_ends.append(prog.get_timestamp(gen_ch=self.ch))


class _PlacedConfig(ConfigReadout):
    def __init__(self, book: _TimelineBook, timeline: str, source: ConfigReadout):
        super().__init__(ch=source.ch, name=source.name, tag=source.tag)
        self.book = book
        self.timeline = timeline
        self.relative_t: float | QickParam = source.t

    def preprocess(self, prog) -> None:
        state = self.book.state(self.timeline)
        self.t = state.offset + self.relative_t
        super().preprocess(prog)
        state.ro_ends.append(self.t)


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


class TimelineProgramV2(AveragerProgramV2, ProgramResourcesMixin):
    """Author native QICK hooks with optional named timelines (times in us).

    Constructor arguments and author hooks follow AveragerProgramV2. Named
    scheduling is resolved during preprocessing, not when author hooks run.
    Each compilation starts with empty named cursors/ends. Native compile()
    retains QICK's binary cache; set binprog=None then call compile() to rebuild.
    Unnamed operations retain native behavior.
    """

    # QICK accepts None to disable these shell operations despite inferring
    # non-optional types from the defaults in its unannotated constructor.
    initial_delay: float | None
    final_wait: float | None
    binprog: _BinaryProgram | None

    def __init__(self, *args, **kwargs):
        self._timeline_book = _TimelineBook()
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
        self.append_macro(_Boundary(self._timeline_book, "reset"))
        self.set_ext_counter(addr=self.COUNTER_ADDR)
        self._initialize(self.cfg)
        self.append_macro(_Boundary(self._timeline_book, "initialize_end"))
        if self.initial_delay is not None:
            self.delay_auto(self.initial_delay)
        for name, count, before, _after in self.loops:
            self.open_loop(count, name=name)
            if before is not None:
                self.extend_macros(before)
        self.append_macro(_Boundary(self._timeline_book, "body_start"))
        self._body(self.cfg)
        self.append_macro(_Boundary(self._timeline_book, "body_end"))
        if self.final_wait is not None:
            self.wait_auto(self.final_wait, no_warn=True)
        if self.final_delay is not None:
            self.delay_auto(self.final_delay)
        self.inc_ext_counter(addr=self.COUNTER_ADDR)
        for _name, _count, _before, after in self.loops[::-1]:
            if after is not None:
                self.extend_macros(after)
            self.close_loop()
        self._cleanup(self.cfg)
        self.end()
        for name, asm in self.subroutines.items():
            self.label(name)
            self.extend_macros(asm)
            self.ret()

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
        timeline, retain native timing and tag behavior.
        """
        super().delay_auto(*args, **kwargs)
        if timeline is not None:
            source = self.macro_list.pop()
            self.append_macro(_Advance(self._timeline_book, timeline, source))

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
