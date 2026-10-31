"""Place timeline-managed outputs while retaining native QICK consumers."""

from typing import Literal

from qick.asm_v2 import ConfigReadout, Pulse, QickParam, Trigger

from ._timeline_guard import PulseGuard as _PulseGuard
from ._timeline_guard import bounds as _bounds
from ._timeline_runtime import (
    EventEnd,
    RuntimeEnd,
    Time,
    align_end,
    optional_time_bounds,
    placement_time,
)
from ._timeline_state import TimelineBook


class TimelineObservedPulse(Pulse):
    """Delegate native pulse behavior, observing only its public timing inputs."""

    def __init__(self, book: TimelineBook, source: Pulse):
        # Directly constructed QICK Pulse macros may omit the optional tag.
        super().__init__(
            ch=source.ch, name=source.name, tag=getattr(source, "tag", None)
        )
        self.book = book
        self.source = source
        self.runtime_start: RuntimeEnd | None = None

    def record(self, prog, t, limit: float | QickParam | None = None) -> None:
        if self.book.repeats:
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
        self.source.preprocess(prog)
        if self.source.t == "auto" and self.ch in self.book.runtime_gens:
            start = self.book.runtime_gens[self.ch]
            self.runtime_start = start
            self.source.t_regs["t"] = start.register
            self.book.runtime_gens[self.ch] = self.book.math.add(
                start, align_end(prog, prog.pulses[self.name].get_length())
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
        end = self.book.end(self.timeline, placement=True)
        previous = self.book.channel_end(prog, self.ch, generator=True)
        if self.relative_t == "auto":
            end = self.book.math.maximum(end, previous)
        if isinstance(end, RuntimeEnd):
            end_limits = optional_time_bounds(prog, end)
            previous_limits = optional_time_bounds(prog, previous)
            if (
                end_limits is not None
                and previous_limits is not None
                and self.relative_t != "auto"
                and not (
                    (
                        _bounds(self.relative_t)[0] >= -1e-12
                        and self.book.math.covers(end, previous)
                    )
                    or end_limits[0] + _bounds(self.relative_t)[0]
                    >= previous_limits[1] - 1e-12
                )
            ):
                raise ValueError(
                    f"{self.timeline}: channel {self.ch} pulse overlap; cannot prove runtime start >= previous end"
                )
            if self.relative_t == "auto":
                start = end
            else:
                start = self.book.math.add(end, placement_time(prog, self.relative_t))
            self.runtime_start = start
            self.source.t = self.book.math.preprocess_time(start)
            self.source.preprocess(prog)
            self.source.t_regs["t"] = self.book.math.cycles(start).register
            finish = self.book.math.add(
                start, align_end(prog, prog.pulses[self.name].get_length())
            )
            offset = (
                0.0
                if self.relative_t == "auto"
                else placement_time(prog, self.relative_t)
            )
            self.event_end = EventEnd(
                end, offset + prog.pulses[self.name].get_length(), (finish,)
            )
            self.book.set_end(self.timeline, finish)
            self.book.runtime_gens[self.ch] = finish
            self.book.record_native(prog, self.ch, finish, generator=True)
            limits = optional_time_bounds(prog, start)
            if limits is not None:
                self.record(prog, limits[1], limits[0])
            return
        t = end if self.relative_t == "auto" else placement_time(prog, end + self.relative_t)
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
                "start >= previous end at every sweep point; use delay_timeline"
            )
        self.record(prog, t, actual_t)
        actual_end = actual_t + prog.pulses[self.name].get_length()
        finish = align_end(prog, actual_end)
        self.event_end = EventEnd(0.0, actual_end, (finish,))
        self.book.set_end(self.timeline, finish)
        if isinstance(finish, RuntimeEnd):
            self.book.runtime_gens[self.ch] = finish
        else:
            self.book.runtime_gens.pop(self.ch, None)
        self.book.record_native(prog, self.ch, finish, generator=True)


class TimelinePlacedConfig(ConfigReadout):
    def __init__(self, book: TimelineBook, timeline: str, source: ConfigReadout):
        super().__init__(ch=source.ch, name=source.name, tag=source.tag)
        self.book = book
        self.timeline = timeline
        self.source = source
        self.relative_t: float | QickParam = source.t
        self.event_end = EventEnd(0.0, 0.0)
        self.runtime_start: RuntimeEnd | None = None

    def preprocess(self, prog) -> None:
        origin = self.book.end(self.timeline, placement=True)
        if isinstance(origin, RuntimeEnd):
            start = self.book.math.add(origin, placement_time(prog, self.relative_t))
            self.event_end = EventEnd(origin, placement_time(prog, self.relative_t))
            self.runtime_start = start
            self.source.t = self.book.math.preprocess_time(start)
            self.source.preprocess(prog)
            self.source.t_regs["t"] = start.register
            self.book.set_end(self.timeline, self.book.math.maximum(0.0, start))
            return
        self.source.t = placement_time(prog, origin + self.relative_t)
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


class TimelinePlacedTrigger(Trigger):
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
        self.event_end = EventEnd(0.0, 0.0)
        self.runtime_start: RuntimeEnd | None = None

    def preprocess(self, prog) -> None:
        origin = self.book.end(self.timeline, placement=True)
        if self.width is not None:
            self.width = placement_time(prog, self.width)
        if isinstance(origin, RuntimeEnd):
            start = self.book.math.add(origin, placement_time(prog, self.relative_t))
            self.runtime_start = start
            self.t = self.book.math.preprocess_time(start)
            super().preprocess(prog)
            self.t_regs["t"] = start.register
            duration = self.get_time_param("width")
            end = self.book.math.add(start, duration)
            for ch in self.ros:
                length = (
                    prog.ro_chs[ch]["length"] / prog.soccfg["readouts"][ch]["f_output"]
                )
                duration = self.book.math.maximum(duration, length)
                ro_end = self.book.math.add(start, align_end(prog, length))
                self.book.runtime_ros[ch] = ro_end
                self.book.record_native(prog, ch, ro_end, generator=False)
                end = self.book.math.maximum(end, ro_end)
            offset = placement_time(prog, self.relative_t)
            self.event_end = EventEnd(
                origin,
                self.book.math.add(duration, offset)
                if isinstance(duration, RuntimeEnd)
                else offset + duration,
            )
            self.book.set_end(self.timeline, end)
            return
        self.t = placement_time(prog, origin + self.relative_t)
        super().preprocess(prog)
        actual_t = self.get_time_param("t")
        end: Time = actual_t + self.get_time_param("width")
        for ch in self.ros:
            length = prog.ro_chs[ch]["length"] / prog.soccfg["readouts"][ch]["f_output"]
            # QICK's timestamp uses the requested trigger time, not its rounded placement.
            ro_end = actual_t + length
            self.book.native_timestamp(prog, ro_end, ro_ch=ch)
            self.book.record_native(prog, ch, ro_end, generator=False)
            end = self.book.math.maximum(end, ro_end)
        self.event_end = EventEnd(0.0, end)
        self.book.set_end(
            self.timeline,
            end if isinstance(end, RuntimeEnd) else align_end(prog, end),
        )
