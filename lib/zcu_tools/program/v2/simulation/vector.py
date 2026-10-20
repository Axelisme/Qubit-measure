"""Array Module interpretation; Python dispatch scales with slots, not points."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from math import prod

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickParam, QickRawParam

from zcu_tools.core.simulation import (
    EventSlot,
    PulseEnvelope,
    ReadoutSlot,
    TimelineTemplate,
)

from ..modular import ModularProgramV2
from ..modules.base import Module
from ..modules.computed_pulse import ComputedPulse
from ..modules.control import Branch, Repeat
from ..modules.delay import Delay, DelayAuto, Join, SoftDelay
from ..modules.dmem import LoadValue, LoadWord, ScanWith
from ..modules.pulse import Pulse, TableLengthPulse
from ..modules.readout import MuxReadout, PulseReadout, TablePulseReadout
from ..modules.reset import BathReset, NoneReset, PulseReset, TwoPulseReset
from ..modules.util import merge_max_length, round_timestamp
from ..modules.waveform import FlatTopWaveformCfg
from ..utils import is_qick_param, readout_freqs_from_words
from ._envelope_pool import EnvelopePool
from ._mux import mux_parameters
from .templates import FlatTopEnvelopes, select_candidates, take_slot
from .timeline import TimelineInterpreter

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
Time = float | QickParam
_REPS_ERROR = "simulation requires reps not to change the timeline"


@dataclass
class _Register:
    values: IntArray
    words: BoolArray
    valid: BoolArray
    depends_reps: bool | BoolArray = False


@dataclass(frozen=True)
class _PulseMetadata:
    duration: FloatArray
    frequency: FloatArray
    phase: FloatArray
    gain: FloatArray
    envelopes: tuple[PulseEnvelope | None, ...]
    block_length: Time | None


@dataclass(frozen=True)
class _ComputedPulseTable:
    """Candidate-by-point matrices at the Module's most recent scalar offset."""

    local_time: float
    gen_ch: int
    start_offsets: FloatArray
    duration: FloatArray
    frequency: FloatArray
    phase: FloatArray
    gain: FloatArray
    envelopes: tuple[PulseEnvelope | None, ...]
    block_length: float


@dataclass
class _Read:
    trigger: FloatArray
    frequency: FloatArray
    phase: FloatArray
    generators: IntArray
    active: BoolArray


class VectorInterpreter:
    """Interpret a compiled program once over its C-order non-reps grid.

    run_templates returns detached numeric structural classes. Unsupported Module
    operations, invalid registers/layouts and reps-dependent controls raise before
    any backend call. The scalar interpreter supplies shared envelope extraction
    and validation only; it never evaluates hardware points here.

    flat_tops owns flat-top envelope lowering.
    """

    def __init__(
        self, prog: ModularProgramV2, *, envelope_pool: EnvelopePool | None = None
    ) -> None:
        """Borrow compiled controls, optionally reusing exact DAC envelopes.

        envelope_pool retains immutable records only; all registers, point arrays
        and name aliases remain local to this preparation.
        """
        self._prog = prog
        self._diagnostic = TimelineInterpreter(prog, envelope_pool=envelope_pool)
        self._names = tuple(n for n in prog.loop_dict if n != "reps")
        self._dims = tuple(prog.loop_dict[n] for n in self._names)
        self._size = prod(self._dims)
        grid = (
            np.indices(self._dims, dtype=np.int64).reshape(len(self._dims), self._size)
            if self._dims
            else np.empty((0, 1), dtype=np.int64)
        )
        self._grid = dict(zip(self._names, grid, strict=True))
        self._reps = prog.loop_dict["reps"]
        self._active = np.ones(self._size, dtype=np.bool_)
        self._presence = self._active.copy()
        self._registers = {
            n: _Register(
                v.copy(), np.zeros(self._size, dtype=np.bool_), self._active.copy()
            )
            for n, v in self._grid.items()
        }
        self._registers["reps"] = _Register(
            np.zeros(self._size, dtype=np.int64),
            np.zeros(self._size, dtype=np.bool_),
            self._active.copy(),
            self._reps > 1,
        )
        self._ref = self._zeros()
        self._gen_end = {ch: self._zeros() for ch in prog.gen_chs}
        self._ro_end = {ch: self._zeros() for ch in prog.ro_chs}
        self._events: list[tuple[EventSlot, BoolArray]] = []
        self._reads: dict[int, list[_Read]] = {ch: [] for ch in prog.ro_chs}
        self._iterations: list[tuple[str, int]] = []
        # Module identity is valid only within this compiled interpretation.
        self._pulse_metadata: dict[Pulse, _PulseMetadata] = {}
        self._computed_tables: dict[ComputedPulse, _ComputedPulseTable] = {}
        self._structures: list[IntArray] = []
        self.flat_tops = FlatTopEnvelopes()

    def _zeros(self) -> FloatArray:
        return np.zeros(self._size, dtype=np.float64)

    def _scalar(self, value: Time) -> FloatArray:
        if not is_qick_param(value):
            return np.full(self._size, float(value), dtype=np.float64)
        result = np.full(self._size, value.start, dtype=np.float64)
        for name, span in value.spans.items():
            count = self._prog.loop_dict[name]
            if name == "reps":
                if span and count > 1:
                    raise ValueError(_REPS_ERROR)
                continue
            if count > 1:
                result += span * self._grid[name] / (count - 1)
        return result

    def _tick(self, value: Time) -> FloatArray:
        cycles = self._prog.us2cycles(deepcopy(value))
        if isinstance(cycles, QickRawParam):
            cycles.to_steps(self._prog.loop_dict)
            return self._scalar(cycles / float(self._prog.tproccfg["f_time"]))
        return self._scalar(float(self._prog.cycles2us(cycles)))

    def _align(
        self, extra: FloatArray | None = None, *, gens: bool = True, ros: bool = True
    ) -> None:
        ends = []
        if gens:
            ends.extend(self._gen_end.values())
        if ros:
            ends.extend(self._ro_end.values())
        if ends:
            delay = np.maximum(0, np.maximum.reduce(ends) - self._ref)
            if extra is not None:
                delay += extra
            clock = float(self._prog.tproccfg["f_time"])
            self._ref += np.where(self._active, np.rint(delay * clock) / clock, 0)

    def _advance(self, t: Time) -> None:
        self._ref += np.where(self._active, self._tick(t), 0)

    def _register(
        self, name: str, *, word: bool = False, timeline: bool = True
    ) -> _Register:
        reg = self._registers.get(name)
        if reg is None or np.any(self._active & ~reg.valid):
            raise ValueError(f"undefined register {name!r}")
        if np.any(self._active & (reg.words != word)):
            raise ValueError(f"register {name!r}: unsupported word/scalar consumer")
        if timeline and np.any(self._active & reg.depends_reps):
            raise ValueError(_REPS_ERROR)
        return reg

    def _write(
        self,
        name: str,
        values: IntArray,
        *,
        word: bool = False,
        depends_reps: bool | BoolArray = False,
    ) -> None:
        old = self._registers.get(name)
        if old is None:
            old = _Register(
                np.zeros(self._size, dtype=np.int64),
                np.zeros(self._size, dtype=np.bool_),
                np.zeros(self._size, dtype=np.bool_),
            )
        self._registers[name] = _Register(
            np.where(self._active, values, old.values),
            np.where(self._active, word, old.words),
            old.valid | self._active,
            np.where(self._active, depends_reps, old.depends_reps),
        )

    def _index(self, name: str, count: int) -> IntArray:
        index = self._register(name).values
        if np.any(self._active & ((index < 0) | (index >= count))):
            raise ValueError(f"table index out of bounds via register {name!r}")
        return np.where(self._active, index, 0)

    def _sequence(self, modules: Sequence[Module], t: Time) -> Time:
        for module in modules:
            try:
                t = self.run(module, t)
            except (ValueError, KeyError, IndexError) as exc:
                if str(exc) == _REPS_ERROR:
                    raise
                raise ValueError(
                    f"{type(module).__name__}({module.name!r}) at loops {tuple(self._prog.loop_dict)}: {exc}"
                ) from exc
        return t

    def run(self, module: Module, t: Time) -> Time:
        """Schedule module on the active mask at symbolic processor offset t (us).

        Return the next symbolic offset in us. The enclosing sequence owns the
        reference clock and mask; unsupported semantic operations raise ValueError.
        """
        if type(module) is Pulse:
            return self._pulse(module, t)
        if type(module) is PulseReadout:
            return self._readout(module, t)
        if type(module) is MuxReadout:
            return self._mux_readout(module, t)
        if type(module) in (SoftDelay, Delay, DelayAuto, Join):
            return self._delay(module, t)
        return self._control(module, t)

    def _delay(self, module: Module, t: Time) -> Time:
        if type(module) is SoftDelay:
            return round_timestamp(self._prog, t + module.delay)
        if type(module) is Delay:
            self._advance(round_timestamp(self._prog, t + module.delay))
            return 0.0
        if type(module) is DelayAuto:
            extra = (
                self._register(module.t).values / float(self._prog.tproccfg["f_time"])
                if isinstance(module.t, str)
                else self._scalar(module.t)
            )
            self._align(extra, gens=module.gens, ros=module.ros)
            return 0.0
        if type(module) is Join:
            return self._join(module, t)
        raise ValueError(f"{type(module).__name__} is not a delay Module")

    def _control(self, module: Module, t: Time) -> Time:
        if type(module) in (LoadValue, LoadWord) and isinstance(
            module, (LoadValue, LoadWord)
        ):
            self._load(module)
            return t
        if type(module) is ScanWith:
            return self._repeat(module.repeat_mod, t)
        if type(module) is Repeat:
            return self._repeat(module, t)
        if type(module) is Branch:
            return self._branch(module, t)
        return self._selected_pulse(module, t)

    def _load(self, module: LoadValue | LoadWord) -> None:
        if module.values:
            reg = self._register(module.idx_reg, timeline=False)
            if np.any(
                self._active & ((reg.values < 0) | (reg.values >= len(module.values)))
            ):
                raise ValueError(
                    f"table index out of bounds via register {module.idx_reg!r}"
                )
            values = np.asarray(module.values, dtype=np.int64)
            self._write(
                module.val_reg,
                values[np.where(self._active, reg.values, 0)],
                word=type(module) is LoadWord,
                depends_reps=reg.depends_reps & bool(np.any(values != values[0])),
            )

    def _selected_pulse(self, module: Module, t: Time) -> Time:
        if type(module) is ComputedPulse:
            return self._computed(module, t)
        if type(module) is TableLengthPulse:
            return self._table_length(module, t)
        if type(module) is TablePulseReadout:
            return self._table_readout(module, t)
        return self._reset(module, t)

    def _computed(self, module: ComputedPulse, t: Time) -> Time:
        index = self._index(module.val_reg, len(module.pulse_modules))
        if isinstance(t, QickParam):
            slots = self._computed_slots(module, t)
            slot = select_candidates(slots, index, source=module.name)
            block_length = module.total_length(self._prog)
        else:
            table = self._computed_tables.get(module)
            if table is None or table.local_time != t:
                table = self._computed_table(module, t)
                self._computed_tables[module] = table

            def select(values: FloatArray) -> FloatArray:
                return np.take_along_axis(values, index[None, :], axis=0)[0]

            slot = EventSlot(
                table.gen_ch,
                module.name,
                tuple(self._iterations),
                select(self._ref[None, :] + table.start_offsets),
                select(table.duration),
                select(table.frequency),
                select(table.phase),
                select(table.gain),
                table.envelopes,
                index.copy(),
                self._active.copy(),
            )
            block_length = table.block_length
        self._events.append((slot, self._presence.copy()))
        self._gen_end[slot.gen_ch] = np.where(
            self._active,
            slot.start_us + slot.duration_us,
            self._gen_end[slot.gen_ch],
        )
        return t + block_length

    def _computed_slots(self, module: ComputedPulse, t: Time) -> list[EventSlot]:
        slots = []
        for candidate in module.pulse_modules:
            start = len(self._events)
            self._pulse(candidate, t)
            slots.append(self._events.pop(start)[0])
        return slots

    def _computed_table(self, module: ComputedPulse, t: float) -> _ComputedPulseTable:
        slots = self._computed_slots(module, t)
        # Cache quantized offsets before adding the dynamic reference. Subtracting
        # the reference from an absolute timestamp would change float rounding.
        offsets = []
        for candidate in module.pulse_modules:
            assert candidate.cfg is not None
            offsets.append(self._tick(t + candidate.cfg.pre_delay))
        return _ComputedPulseTable(
            t,
            slots[0].gen_ch,
            np.stack(offsets),
            np.stack([s.duration_us for s in slots]),
            np.stack([s.frequency_mhz for s in slots]),
            np.stack([s.phase_deg for s in slots]),
            np.stack([s.gain for s in slots]),
            tuple(s.envelopes[0] for s in slots),
            module.total_length(self._prog),
        )

    def _reset(self, module: Module, t: Time) -> Time:
        if type(module) is NoneReset:
            return t
        if type(module) is PulseReset:
            return self._pulse(module.reset_pulse, t)
        if type(module) is TwoPulseReset:
            return max(
                self._pulse(module.reset_pulse1, t), self._pulse(module.reset_pulse2, t)
            )
        if type(module) is BathReset:
            end = self._pulse(module.res_pulse, t)
            self._pulse(module.qub_pulse, t)
            return self._pulse(module.pi2_pulse, end)
        raise ValueError(f"{type(module).__name__} simulation unsupported")

    def _repeat(self, module: Repeat, t: Time) -> float:
        self._advance(t)
        self._align()
        n = (
            self._register(module.n).values
            if isinstance(module.n, str)
            else np.full(self._size, module.n, dtype=np.int64)
        )
        if np.any(self._active & (n < 0)):
            raise ValueError("negative repeat count")
        upper = (
            module.n
            if isinstance(module.n, int)
            else (
                module.range_hint[1] if module.range_hint is not None else int(n.max())
            )
        )
        if np.any(self._active & (n > upper)):
            raise ValueError("repeat count exceeds range_hint")
        active = self._active.copy()
        for index in range(upper):
            self._active = active & (index < n)
            self._write(module.counter_reg, np.full(self._size, index, dtype=np.int64))
            self._iterations.append((module.name, index))
            end = self._sequence(module.sub_modules, 0.0)
            self._iterations.pop()
            self._advance(end)
            self._align()
        self._active = active
        self._write(module.counter_reg, n)
        return 0.0

    def _branch(self, module: Branch, t: Time) -> float:
        self._advance(t)
        self._align()
        index = self._index(module.compare_reg, len(module.branches))
        self._structures.append(index)
        active, presence = self._active.copy(), self._presence.copy()
        for selected in np.unique(index[active]):
            self._active = active & (index == selected)
            self._presence = presence & (index == selected)
            end = self._sequence(module.branches[int(selected)], 0.0)
            self._advance(end)
            self._align()
        self._active, self._presence = active, presence
        return 0.0

    def _join(self, module: Join, t: Time) -> Time:
        self._diagnostic.check_join(module)
        queues = [list(branch) for branch in module.join_modules]
        times = [t for _ in queues]

        def minimum(value: Time) -> float:
            return value.minval() if is_qick_param(value) else float(value)

        while any(queues):
            candidates = [i for i, queue in enumerate(queues) if queue]
            index = min(
                candidates,
                key=lambda i: minimum(times[i]),
            )
            times[index] = self._sequence([queues[index].pop(0)], times[index])
        return merge_max_length(*times)

    def _pulse(
        self, module: Pulse, t: Time, *, duration: FloatArray | None = None
    ) -> Time:
        cfg = module.cfg
        if cfg is None:
            return t
        # Runtime table lengths can reuse a Pulse with different durations, so
        # neither read nor populate the static cache for explicit durations.
        reusable = duration is None and bool(self._iterations)
        metadata = self._pulse_metadata.get(module) if reusable else None
        if metadata is None:
            prog = self._prog
            if duration is None:
                duration = self._scalar(
                    prog.get_pulse_param(module.pulse_id, "total_length")
                )
            envelope = self._diagnostic.envelope(module, float(duration[0]))
            if isinstance(cfg.waveform, FlatTopWaveformCfg) and envelope is not None:
                self.flat_tops.record(envelope)
            frequency = self._scalar(prog.get_pulse_param(module.pulse_id, "freq"))
            phase = self._scalar(prog.get_pulse_param(module.pulse_id, "phase"))
            gain = self._scalar(prog.get_pulse_param(module.pulse_id, "gain"))
            envelopes = (envelope,)
            block_length = module.total_length(prog) if module.block_mode else None
            if reusable:
                self._pulse_metadata[module] = _PulseMetadata(
                    duration, frequency, phase, gain, envelopes, block_length
                )
        else:
            duration = metadata.duration
            frequency, phase, gain = metadata.frequency, metadata.phase, metadata.gain
            envelopes, block_length = metadata.envelopes, metadata.block_length
        slot = EventSlot(
            cfg.ch,
            module.name,
            tuple(self._iterations),
            self._ref + self._tick(t + cfg.pre_delay),
            duration.copy(),
            frequency,
            phase,
            gain,
            envelopes,
            np.zeros(self._size, dtype=np.int64),
            self._active.copy(),
        )
        self._events.append((slot, self._presence.copy()))
        self._gen_end[cfg.ch] = np.where(
            self._active, slot.start_us + duration, self._gen_end[cfg.ch]
        )
        return t + block_length if block_length is not None else t

    def _table_length(self, module: TableLengthPulse, t: Time) -> float:
        index = self._index(module.idx_reg, len(module.lengths))
        candidates = [
            module.playback_parameters(self._prog, i)
            for i in range(len(module.lengths))
        ]
        slots = []
        previous = self._gen_end[module.cfg.ch].copy()
        for template, cycles, _ in candidates:
            duration = float(self._prog.cycles2us(cycles, gen_ch=module.cfg.ch))
            duration += self._diagnostic.scalar(
                self._prog.get_pulse_param(template.pulse_id, "total_length")
            ) - self._diagnostic.scalar(
                self._prog.get_pulse_param(template.pulse_id, "length")
            )
            self._pulse(
                template, t, duration=np.full(self._size, duration, dtype=np.float64)
            )
            slots.append(self._events.pop()[0])
        self._events.append(
            (
                select_candidates(slots, index, source=slots[0].source),
                self._presence.copy(),
            )
        )
        self._gen_end[module.cfg.ch] = previous
        durations = np.asarray([c[2] for c in candidates], dtype=np.float64) / float(
            self._prog.tproccfg["f_time"]
        )
        self._align(durations[index])
        return 0.0

    def _frequencies(
        self, module: PulseReadout, words: tuple[IntArray, IntArray] | None
    ) -> tuple[FloatArray, FloatArray]:
        gen = self._scalar(self._prog.get_pulse_param(module.pulse.pulse_id, "freq"))
        ro = self._scalar(self._prog.get_pulse_param(module.adc_name, "freq"))
        gen_word = (
            self._register(module.freq_val, word=True).values
            if module.freq_val
            else None
        )
        ro_word = (
            self._register(module.ro_freq_val, word=True).values
            if module.ro_freq_val
            else None
        )
        if words is not None:
            gen_word, ro_word = words
        cfg = module.cfg
        if gen_word is not None:
            gen = readout_freqs_from_words(
                self._prog.soccfg,
                gen_word,
                kind="gen",
                gen_ch=cfg.gen_ch,
                ro_ch=cfg.ro_ch,
                mixer_freq=cfg.mixer_freq,
                nqz=cfg.nqz,
                paired_ro_words=ro_word,
            )
        if ro_word is not None:
            ro = (
                gen.copy()
                if gen_word is not None
                else readout_freqs_from_words(
                    self._prog.soccfg,
                    ro_word,
                    kind="ro",
                    gen_ch=cfg.gen_ch,
                    ro_ch=cfg.ro_ch,
                    mixer_freq=cfg.mixer_freq,
                    nqz=cfg.nqz,
                    reference_freq_mhz=gen,
                )
            )
        return gen, ro

    def _readout(
        self,
        module: PulseReadout,
        t: Time,
        *,
        words: tuple[IntArray, IntArray] | None = None,
    ) -> Time:
        cfg = module.cfg
        trigger = self._ref + self._tick(t + cfg.trig_offset)
        gen, ro = self._frequencies(module, words)
        self._reads[cfg.ro_ch].append(
            _Read(
                trigger,
                ro,
                self._zeros(),
                np.full(self._size, cfg.gen_ch, dtype=np.int64),
                self._active.copy(),
            )
        )
        self._ro_end[cfg.ro_ch] = np.where(
            self._active,
            trigger
            + self._prog.cycles2us(
                self._prog.ro_chs[cfg.ro_ch]["length"], ro_ch=cfg.ro_ch
            ),
            self._ro_end[cfg.ro_ch],
        )
        self._pulse(module.pulse, t)
        event, presence = self._events.pop()
        gain = event.gain
        if module.gain_val is not None:
            word = self._register(module.gain_val, word=True).values
            gain = np.where(
                word < 2**31, word, word - 2**32
            ) / self._prog.soccfg.get_maxv(cfg.gen_ch)
        self._events.append(
            (replace(event, source=module.name, frequency_mhz=gen, gain=gain), presence)
        )
        return t + module.total_length(self._prog)

    def _mux_readout(self, module: MuxReadout, t: Time) -> Time:
        cfg = module.cfg
        start = self._ref + self._tick(t + cfg.pre_delay)
        duration = self._scalar(self._prog.get_pulse_param(module.name, "total_length"))
        trigger = self._ref + self._tick(t + cfg.trig_offset)
        for tone in mux_parameters(self._prog, module):
            self._events.append(
                (
                    EventSlot(
                        cfg.gen_ch,
                        module.name,
                        tuple(self._iterations),
                        start.copy(),
                        duration.copy(),
                        np.full(self._size, tone.frequency_mhz),
                        np.full(self._size, tone.phase_deg),
                        np.full(self._size, tone.gain),
                        (None,),
                        np.zeros(self._size, dtype=np.int64),
                        self._active.copy(),
                    ),
                    self._presence.copy(),
                )
            )
            self._reads[tone.channel].append(
                _Read(
                    trigger.copy(),
                    np.full(self._size, tone.demodulation_mhz),
                    np.full(self._size, tone.demodulation_phase_deg),
                    np.full(self._size, cfg.gen_ch, dtype=np.int64),
                    self._active.copy(),
                )
            )
            self._ro_end[tone.channel] = np.where(
                self._active,
                trigger
                + self._prog.cycles2us(
                    self._prog.ro_chs[tone.channel]["length"], ro_ch=tone.channel
                ),
                self._ro_end[tone.channel],
            )
        self._gen_end[cfg.gen_ch] = np.where(
            self._active, start + duration, self._gen_end[cfg.gen_ch]
        )
        return t + module.total_length(self._prog)

    def _table_readout(self, module: TablePulseReadout, t: Time) -> Time:
        names = tuple(self._prog.loop_dict)
        axis = names.index(module.idx_reg)
        stride = prod(tuple(self._prog.loop_dict.values())[axis + 1 :])
        flat = np.zeros(self._size, dtype=np.int64)
        for name, count in self._prog.loop_dict.items():
            flat = flat * count + self._registers[name].values
        index = self._registers[module.idx_reg].values
        if (
            module.idx_reg == "reps"
            and self._reps > 1
            and (
                np.any(np.asarray(module.freq_words) != module.freq_words[0])
                or np.any(np.asarray(module.ro_freq_words) != module.ro_freq_words[0])
            )
        ):
            raise ValueError(_REPS_ERROR)
        index = np.where(flat < stride, 0, index)
        words = (
            np.asarray(module.freq_words, dtype=np.int64)[index],
            np.asarray(module.ro_freq_words, dtype=np.int64)[index],
        )
        gen, ro = self._frequencies(module.initial_template, words)
        initial_gen, initial_ro = self._frequencies(module.initial_template, None)
        reps_outer = names.index("reps") <= axis
        if (
            self._reps > 1
            and reps_outer
            and (
                np.any(gen[flat < stride] != initial_gen[flat < stride])
                or np.any(ro[flat < stride] != initial_ro[flat < stride])
            )
        ):
            raise ValueError(_REPS_ERROR)
        start = len(self._events)
        end = self._readout(module.initial_template, t, words=words)
        event, presence = self._events[start]
        self._events[start] = (
            replace(event, frequency_mhz=np.where(flat < stride, initial_gen, gen)),
            presence,
        )
        read = self._reads[module.cfg.ro_ch][-1]
        read.frequency = np.where(flat < stride, initial_ro, ro)
        return end

    def run_templates(self) -> tuple[TimelineTemplate, ...]:
        """Return structural classes covering every non-reps point once.

        Event slots retain expanded Repeat positions with finite inactive entries.
        Branch selections partition classes; ADC reads must match compiled counts.
        """
        end = self._sequence(self._prog.modules, 0.0)
        duration = np.maximum.reduce(
            [
                self._ref + self._tick(end),
                *self._gen_end.values(),
                *self._ro_end.values(),
                *(
                    np.where(slot.active, slot.start_us + slot.duration_us, 0)
                    for slot, _ in self._events
                ),
            ]
        )
        for slot, _ in self._events:
            long = self.flat_tops.long_plateau(slot)
            if long is not None:
                self._structures.append(long.astype(np.int64))
        _, labels = np.unique(
            np.stack(self._structures, axis=1)
            if self._structures
            else np.zeros((self._size, 1), dtype=np.int64),
            axis=0,
            return_inverse=True,
        )
        templates = []
        for label in np.unique(labels):
            points = np.flatnonzero(labels == label).astype(np.int64)
            events = tuple(
                take_slot(slot, points)
                for slot, presence in self._events
                if np.any(presence[points])
            )
            readouts = []
            for ch, cfg in self._prog.ro_chs.items():
                reads = [r for r in self._reads[ch] if np.any(r.active[points])]
                if len(reads) != cfg["trigs"] or any(
                    not np.all(r.active[points]) for r in reads
                ):
                    raise ValueError(
                        "simulation readout layout does not match compiled triggers"
                    )
                readouts.append(
                    ReadoutSlot(
                        ch,
                        int(cfg["length"]),
                        float(self._prog.soccfg["readouts"][ch]["f_output"]),
                        np.stack([r.trigger[points] for r in reads], axis=1),
                        np.stack([r.frequency[points] for r in reads], axis=1),
                        np.stack([r.phase[points] for r in reads], axis=1),
                        np.stack([r.generators[points] for r in reads], axis=1),
                    )
                )
            templates.append(
                TimelineTemplate(
                    points, events, tuple(readouts), duration[points].copy()
                )
            )
        return tuple(self.flat_tops.lower(t) for t in templates)
