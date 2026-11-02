"""Place timeline-managed outputs while retaining native QICK consumers."""

from fractions import Fraction
from typing import Literal

from qick.asm_v2 import ConfigReadout, Pulse, QickParam, Trigger

from ._timeline_guard import PulseGuard as _PulseGuard
from ._timeline_guard import bounds as _bounds
from ._timeline_runtime import (
    EventEnd,
    NativeLength,
    RuntimeEnd,
    Time,
    align_end,
    optional_time_bounds,
    placement_time,
)
from ._timeline_state import TimelineBook
from .modules.util import get_fclk
from .table_param import TableParam


def _static_ticks(prog, value: float | QickParam) -> int | None:
    if isinstance(value, QickParam):
        if value.is_sweep():
            return None
        value = value.start
    return round(value * get_fclk(prog))


def _pulse_static_cycles(
    prog, ch: int, name: str, start: float | QickParam
) -> Fraction | None:
    ticks = _static_ticks(prog, start)
    length = prog.pulses[name].get_length()
    if ticks is None or isinstance(length, QickParam) and length.is_sweep():
        return None
    scalar = length.start if isinstance(length, QickParam) else length
    clock = prog.soccfg["gens"][ch]["f_fabric"]
    return Fraction(ticks) + Fraction(round(scalar * clock)) * Fraction(
        str(get_fclk(prog))
    ) / Fraction(str(clock))


def _trigger_static_cycles(
    prog, start: float | QickParam, width: float | QickParam
) -> Fraction | None:
    ticks, width_ticks = _static_ticks(prog, start), _static_ticks(prog, width)
    if ticks is None or width_ticks is None:
        return None
    return Fraction(ticks + width_ticks)


class TimelineObservedPulse(Pulse):
    """Delegate native pulse behavior, observing only its public timing inputs."""

    def __repr__(self) -> str:
        # QICK eagerly formats macros even when debug logging is disabled.
        return f"{type(self).__name__}()"

    def __init__(self, book: TimelineBook, source: Pulse):
        # Directly constructed QICK Pulse macros may omit the optional tag.
        super().__init__(
            ch=source.ch, name=source.name, tag=getattr(source, "tag", None)
        )
        self.book = book
        self.source = source
        self.runtime_start: RuntimeEnd | None = None

    def native_length(self, prog) -> NativeLength:
        """Identify this pulse's resolved generator-count lattice."""
        return NativeLength(
            prog.pulses[self.name].get_length(),
            prog.soccfg["gens"][self.ch]["f_fabric"],
        )

    def record(self, prog, t, limit: float | QickParam | None = None) -> None:
        if self.book.repeats and not self.book.loop_timelines:
            length = prog.pulses[self.name].get_length()
            self.book.repeats[-1].pulse_guards[self.ch].then(
                _PulseGuard(
                    self.ch, (length,), (t + length,), () if limit is None else (limit,)
                )
            )

    def preprocess(self, prog) -> None:
        # Auto is resolved once during preprocess. Repeated instructions retain
        # that scheduled time; the validation must not reschedule their outputs.
        t = (
            prog.get_timestamp(gen_ch=self.ch)
            if self.source.t == "auto"
            else self.source.t
        )
        if isinstance(t, TableParam):
            start = self.book.math.cycles(t)
            self.runtime_start = start
            self.source.t = self.book.math.preprocess_time(start)
            self.source.preprocess(prog)
            self.source.t_regs["t"] = start.register
            # QICK accumulates the unrounded timestamp across native auto
            # pulses. Keep that table after preprocessing its scalar stand-in;
            # nearest applies to each placement, not to each duration.
            actual_end = t + prog.pulses[self.name].get_length()
            self.book.native_timestamp(prog, actual_end, gen_ch=self.ch)
            finish = self.book.math.cycles(actual_end)
            self.book.runtime_gens[self.ch] = finish
            self.book.record_native(prog, self.ch, finish, generator=True)
            self.record(prog, t)
            return
        self.source.preprocess(prog)
        if self.source.t != "auto":
            self.book.runtime_gens.pop(self.ch, None)
        if self.source.t == "auto" and self.ch in self.book.runtime_gens:
            start = self.book.runtime_gens[self.ch]
            self.runtime_start = start
            self.source.t_regs["t"] = self.book.math.cycles(start).register
            self.book.runtime_gens[self.ch] = self.book.math.add(
                start,
                align_end(
                    prog, prog.pulses[self.name].get_length(), self.native_length(prog)
                ),
            )
            self.book.record_native(
                prog, self.ch, self.book.runtime_gens[self.ch], generator=True
            )
        self.record(prog, t)

    def translate(self, prog) -> None:
        self.source.translate(prog)

    def get_time_param(self, name: str) -> QickParam:
        if name == "t" and self.runtime_start is not None:
            return self.book.math.time_parameter(self.runtime_start)
        return self.source.get_time_param(name)

    def list_time_params(self) -> list[str]:
        return self.source.list_time_params()


class TimelinePlacedPulse(TimelineObservedPulse):
    def __init__(self, book: TimelineBook, timeline: str, source: Pulse):
        if isinstance(source, TimelineObservedPulse):
            source = source.source
        super().__init__(book, source)
        self.timeline = timeline
        self.relative_t: float | QickParam | Literal["auto"] = source.t
        self.event_end = EventEnd(0.0, 0.0)

    def preprocess(self, prog) -> None:
        end = self.book.end(self.timeline)
        previous = self.book.channel_end(prog, self.ch, generator=True)
        if self.relative_t == "auto":
            end = self.book.math.maximum(end, previous)
        if isinstance(self.relative_t, TableParam):
            end = self.book.math.cycles(end)
        if isinstance(end, RuntimeEnd):
            scheduled = self.book.math.add(
                end,
                0.0
                if self.relative_t == "auto"
                else placement_time(prog, self.relative_t),
            )
            start = self.book.math.event_time(scheduled, 0.0)
            end_limits = optional_time_bounds(prog, start)
            previous_limits = optional_time_bounds(prog, previous)
            if (
                end_limits is not None
                and previous_limits is not None
                and self.relative_t != "auto"
                and not self.book.math.covers(start, previous)
            ):
                raise ValueError(
                    f"{self.timeline}: channel {self.ch} pulse overlap; cannot prove runtime start >= previous end"
                )
            self.book.check_loop_pulse(prog, self.ch, start)
            self.runtime_start = start
            self.source.t = self.book.math.preprocess_time(start)
            self.source.preprocess(prog)
            self.source.t_regs["t"] = "s_out_time"
            finish = self.book.math.add(
                scheduled,
                align_end(
                    prog, prog.pulses[self.name].get_length(), self.native_length(prog)
                ),
            )
            offset = (
                0.0
                if self.relative_t == "auto"
                else placement_time(prog, self.relative_t)
            )
            length = prog.pulses[self.name].get_length()
            if isinstance(length, QickParam) and not length.is_sweep():
                length = length.start
            self.event_end = EventEnd(
                end,
                offset + length,
                (finish,),
                _pulse_static_cycles(prog, self.ch, self.name, offset),
                self.native_length(prog),
            )
            self.book.set_end(self.timeline, finish)
            self.book.runtime_gens[self.ch] = finish
            self.book.record_native(prog, self.ch, finish, generator=True)
            limits = optional_time_bounds(prog, start)
            if limits is not None:
                self.record(prog, limits[1], limits[0])
            return
        t = (
            end
            if self.relative_t == "auto"
            else end + placement_time(prog, self.relative_t)
        )
        previous_end = prog.get_timestamp(gen_ch=self.ch)
        self.source.t = t
        self.source.preprocess(prog)
        actual_t = self.source.get_time_param("t")
        if (
            optional_time_bounds(prog, previous) is not None
            and _bounds(actual_t - previous_end)[0] < -1e-12
        ):
            raise ValueError(
                f"{self.timeline}: channel {self.ch} pulse overlap; cannot prove "
                "start >= previous end at every sweep point; use timeline_inc"
            )
        self.record(prog, t, actual_t)
        actual_end = actual_t + prog.pulses[self.name].get_length()
        finish = align_end(prog, actual_end, self.native_length(prog))
        self.event_end = EventEnd(
            0.0,
            actual_end,
            (finish,),
            _pulse_static_cycles(prog, self.ch, self.name, actual_t),
            self.native_length(prog),
        )
        self.book.set_end(self.timeline, finish)
        if isinstance(actual_end, TableParam):
            # Native auto playback consumes QICK's nearest timestamp, not the
            # ceiled timeline boundary retained in event_end.rounded.
            self.book.runtime_gens[self.ch] = self.book.math.cycles(actual_end)
        elif isinstance(finish, RuntimeEnd):
            self.book.runtime_gens[self.ch] = finish
        else:
            self.book.runtime_gens.pop(self.ch, None)
        self.book.record_native(prog, self.ch, finish, generator=True)


class TimelinePlacedConfig(ConfigReadout):
    def __repr__(self) -> str:
        # QICK eagerly formats macros even when debug logging is disabled.
        return f"{type(self).__name__}()"

    def __init__(self, book: TimelineBook, timeline: str, source: ConfigReadout):
        super().__init__(ch=source.ch, name=source.name, tag=source.tag)
        self.book = book
        self.timeline = timeline
        self.source = source
        self.relative_t: float | QickParam = source.t
        self.event_end = EventEnd(0.0, 0.0)
        self.runtime_start: RuntimeEnd | None = None

    def preprocess(self, prog) -> None:
        origin = self.book.end(self.timeline)
        if isinstance(origin, RuntimeEnd):
            start = self.book.math.event_time(
                origin, placement_time(prog, self.relative_t)
            )
            self.event_end = EventEnd(origin, placement_time(prog, self.relative_t))
            self.runtime_start = start
            self.source.t = self.book.math.preprocess_time(start)
            self.source.preprocess(prog)
            self.source.t_regs["t"] = "s_out_time"
            finish = self.book.math.add(origin, placement_time(prog, self.relative_t))
            self.book.set_end(self.timeline, self.book.math.maximum(0.0, finish))
            return
        self.source.t = origin + placement_time(prog, self.relative_t)
        self.source.preprocess(prog)
        self.event_end = EventEnd(0.0, self.source.get_time_param("t"))
        self.book.set_end(
            self.timeline, self.book.math.maximum(0.0, self.source.get_time_param("t"))
        )

    def translate(self, prog) -> None:
        self.source.translate(prog)

    def get_time_param(self, name: str) -> QickParam:
        if name == "t" and self.runtime_start is not None:
            return self.book.math.time_parameter(self.runtime_start)
        return self.source.get_time_param(name)

    def list_time_params(self) -> list[str]:
        return self.source.list_time_params()


class TimelineObservedTrigger(Trigger):
    """Retain native Trigger outputs, observing its quantized ADC tail evidence."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __init__(self, book: TimelineBook, source: Trigger):
        super().__init__(
            ros=source.ros,
            tts=source.tts,
            pins=source.pins,
            t=source.t,
            width=source.width,
            ddr4=source.ddr4,
            mr=source.mr,
            tag=source.tag,
        )
        self.book = book

    def preprocess(self, prog) -> None:
        super().preprocess(prog)
        if self.t_regs["t"] is None:
            self.book.unknown_native_adcs.update(self.ros)
            return
        actual_t = self.get_time_param("t")
        for ch in self.ros:
            length = prog.ro_chs[ch]["length"] / prog.soccfg["readouts"][ch]["f_output"]
            self.book.record_native(prog, ch, actual_t + length, generator=False)


class TimelinePlacedTrigger(Trigger):
    def __repr__(self) -> str:
        # QICK eagerly formats macros even when debug logging is disabled.
        return f"{type(self).__name__}()"

    def get_time_param(self, name: str) -> QickParam:
        if name == "t" and self.runtime_start is not None:
            return self.book.math.time_parameter(self.runtime_start)
        return super().get_time_param(name)

    def __init__(self, book: TimelineBook, timeline: str, source: Trigger):
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
        self.relative_width: float | QickParam | None = source.width
        self.event_end = EventEnd(0.0, 0.0)
        self.runtime_start: RuntimeEnd | None = None

    def preprocess(self, prog) -> None:
        origin = self.book.end(self.timeline)
        if isinstance(self.relative_t, TableParam):
            origin = self.book.math.cycles(origin)
        if self.relative_width is not None:
            self.width = placement_time(prog, self.relative_width)
        if isinstance(origin, RuntimeEnd):
            start = self.book.math.event_time(
                origin, placement_time(prog, self.relative_t)
            )
            for ch in self.ros or ():
                self.book.check_adc(prog, self.timeline, ch, start)
            self.runtime_start = start
            self.t = self.book.math.preprocess_time(start)
            super().preprocess(prog)
            self.t_regs["t"] = "s_out_time"
            duration = self.get_time_param("width")
            if not duration.is_sweep():
                duration = duration.start
            scheduled = self.book.math.add(
                origin, placement_time(prog, self.relative_t)
            )
            end = self.book.math.add(scheduled, duration)
            for ch in self.ros:
                length = (
                    prog.ro_chs[ch]["length"] / prog.soccfg["readouts"][ch]["f_output"]
                )
                ro_end = self.book.math.add(scheduled, align_end(prog, length))
                self.book.runtime_ros[ch] = ro_end
                self.book.record_native(prog, ch, ro_end, generator=False)
            offset = placement_time(prog, self.relative_t)
            self.event_end = EventEnd(
                origin,
                offset + duration,
                static_cycles=_trigger_static_cycles(
                    prog, offset, self.get_time_param("width")
                ),
            )
            self.book.set_end(self.timeline, end)
            return
        self.t = origin + placement_time(prog, self.relative_t)
        for ch in self.ros or ():
            self.book.check_adc(prog, self.timeline, ch, self.t)
        super().preprocess(prog)
        actual_t = self.get_time_param("t")
        end: Time = actual_t + self.get_time_param("width")
        for ch in self.ros:
            length = prog.ro_chs[ch]["length"] / prog.soccfg["readouts"][ch]["f_output"]
            # QICK's timestamp uses the requested trigger time, not its rounded placement.
            ro_end = actual_t + length
            self.book.native_timestamp(prog, ro_end, ro_ch=ch)
            self.book.record_native(prog, ch, ro_end, generator=False)
        self.event_end = EventEnd(
            0.0,
            end,
            static_cycles=_trigger_static_cycles(
                prog, actual_t, self.get_time_param("width")
            ),
        )
        self.book.set_end(
            self.timeline,
            end if isinstance(end, RuntimeEnd) else align_end(prog, end),
        )
