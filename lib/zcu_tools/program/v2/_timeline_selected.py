"""Selected runtime playback, retaining raw fractional waveform ends."""

from dataclasses import dataclass

from qick.asm_v2 import QickParam, QickRawParam, TimedMacro

from ._timeline_guard import PulseGuard, bounds
from ._timeline_runtime import (
    RuntimeEnd,
    SelectedEventEnd,
    Time,
    placement_time,
    time_bounds,
)
from ._timeline_state import TimelineBook
from .macro.pluse_reg import PulseByReg
from .macro.table import TableSelection
from .macro.wmem import PulseFromLengthReg


@dataclass(frozen=True)
class PlaybackChoice:
    """One declared waveform and its separate post-delay in us.

    name is a registered pulse. length_cycles replaces only the const/flat
    segment's generator length when not None. post is never part of playback.
    """

    name: str
    post: float | QickParam = 0.0
    length_cycles: int | None = None


@dataclass(frozen=True)
class SelectedPlayback:
    """Framework metadata pairing a playback producer with its selector.

    selector is the loader-owned captured zero-based index. register is the
    actual address/length register read by playback. choices are in selector
    order. relation certifies that table and its exact writers; None is rejected
    by timeline playback. Native playback needs no selection metadata.
    """

    selector: str
    register: str
    choices: tuple[PlaybackChoice, ...]
    relation: TableSelection | None = None


class TimelineSelectedPulse(TimedMacro):
    """Place a runtime-address/length pulse and report its raw selected end.

    source retains the native playback instructions. selection describes the
    declared candidates; book and timeline supply the common emission origin.
    Invalid selector/producer provenance or channel overlap fails at compile.
    """

    def __init__(
        self,
        book: TimelineBook,
        timeline: str,
        source: PulseByReg | PulseFromLengthReg,
        selection: SelectedPlayback,
    ):
        super().__init__()
        self.book = book
        self.timeline = timeline
        self.source = source
        self.selection = selection
        self.relative_t: float | QickParam = source.t
        self.event_end = SelectedEventEnd(0.0, selection.selector, (), ())

    def _lengths(self, prog) -> tuple[float | QickParam, ...]:
        lengths: list[float | QickParam] = []
        relation = self.selection.relation
        if relation is None:
            raise ValueError("selected playback requires a table relation")
        if (relation.register, relation.selector) != (
            self.selection.register,
            self.selection.selector,
        ):
            raise ValueError("selected playback relation does not match its registers")
        actual_words: list[int] = []
        for choice in self.selection.choices:
            pulse = prog.pulses[choice.name]
            length = pulse.get_length()
            if (
                isinstance(self.source, PulseFromLengthReg)
                and choice.length_cycles is not None
            ):
                actual_words.append(choice.length_cycles)
                names = pulse.get_wavenames(exclude_special=False)
                segment = names[1] if self.source.flat_top else names[0]
                raw = prog.waves[prog.wave2idx[segment]].length
                fixed = raw.start if isinstance(raw, QickRawParam) else raw
                length += (choice.length_cycles - fixed) / prog.soccfg["gens"][
                    self.source.ch
                ]["f_fabric"]
            elif isinstance(self.source, PulseByReg):
                actual_words.append(
                    prog.wave2idx[pulse.get_wavenames(exclude_special=False)[0]]
                )
            else:
                raise ValueError(
                    "runtime length playback requires generator-cycle choices"
                )
            lengths.append(length)
        if tuple(actual_words) != relation.table.values:
            raise ValueError(
                f"{relation.register}: selected playback choices do not match the declared table"
            )
        return tuple(lengths)

    def preprocess(self, prog) -> None:
        selection = self.selection
        if selection.relation is None:
            raise ValueError(
                f"{selection.register}: selected playback requires a table relation"
            )
        prog._time_bounds.require_selection(prog, selection.relation)
        origin = self.book.end(self.timeline, placement=True)
        previous = self.book.channel_end(prog, self.source.ch, generator=True)
        offset = placement_time(prog, self.relative_t)
        start: Time
        if isinstance(origin, RuntimeEnd):
            start = self.book.math.add(origin, offset)
            raw_origin = origin
        else:
            start = placement_time(prog, origin + self.relative_t)
            raw_origin = 0.0
            offset = start
        if (
            not (
                bounds(self.relative_t)[0] >= 0
                and self.book.math.covers(origin, previous)
            )
            and time_bounds(prog, start)[0] < time_bounds(prog, previous)[1] - 1e-12
        ):
            raise ValueError(
                f"{self.timeline}: channel {self.source.ch} pulse overlap; cannot prove runtime start >= previous end"
            )
        self.source.t = time_bounds(prog, start)[1]
        self.source.preprocess(prog)
        if isinstance(start, RuntimeEnd):
            self.source.t_regs["t"] = start.register
        else:
            self.source.t = start
            self.source.preprocess(prog)
        lengths = self._lengths(prog)
        offsets = tuple(offset + length for length in lengths)
        physical = self.book.math.selected_end(selection.selector, offsets)
        finish = self.book.math.add(raw_origin, physical, target=physical.register)
        self.event_end = SelectedEventEnd(
            raw_origin,
            selection.selector,
            offsets,
            tuple(c.post for c in selection.choices),
            (finish,),
        )
        self.book.set_end(self.timeline, finish)
        self.book.runtime_gens[self.source.ch] = finish
        self.book.record_native(prog, self.source.ch, finish, generator=True)
        self.book.native_timestamp(
            prog, time_bounds(prog, finish)[1], gen_ch=self.source.ch
        )
        if self.book.repeats:
            lo = time_bounds(prog, start)[0]
            hi = time_bounds(prog, finish)[1]
            self.book.repeats[-1].pulse_guards[self.source.ch].then(
                PulseGuard(self.source.ch, shifts=(), floors=(hi,), limits=(lo,))
            )

        if isinstance(self.source, PulseByReg):
            # Address preparation is the final part of this pulse's arithmetic
            # plan. No other plan runs before its contiguous WPORT consumption.
            storage = tuple(
                self.book.math.scratch(index)
                for index in range(self.source.address_scratch_count)
            )
            operations = self.source.prepare_addresses(prog, storage)
            self.source.validate_address_storage(prog)
            self.book.math.operations.extend(operations)

    def translate(self, prog) -> None:
        self.source.translate(prog)

    def list_time_params(self) -> list[str]:
        return self.source.list_time_params()

    def get_time_param(self, name: str) -> QickParam:
        return self.source.get_time_param(name)
