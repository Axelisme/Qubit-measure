"""Group compiled output effects into detached numeric point slots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    EventSlot,
    PulseEnvelope,
    ReadoutSlot,
    TimeBase,
    TimelineTemplate,
)

from ...wave_ports import wave_port_conflicts
from ..base import MyProgramV2
from ..ir.io_instructions import DportWriteInst, PortWriteInst, TrigInst
from ..ir.operands import Register, SrcKeyword
from ..utils import generator_frequencies_from_words, readout_frequencies_from_words
from ._compiled_state import BoolArray, IntArray, Machine
from ._envelope_pool import EnvelopeContent, EnvelopePool
from ._mux import static_readout_parameters


@dataclass(frozen=True)
class _EventValues:
    """Per-lane output values before grouping into full-grid event slots.

    Times are int64 common-clock ticks; carrier is MHz, phase is degrees, and gain is the
    normalized signed amplitude. active identifies lanes playing this output.
    """

    start_ticks: IntArray
    duration_ticks: IntArray
    frequency_mhz: NDArray[np.float64]
    phase_deg: NDArray[np.float64]
    gain: NDArray[np.float64]
    active: BoolArray


def _overlaps_window(event: EventSlot, start: IntArray, duration: int) -> BoolArray:
    """Select active events with strictly positive acquisition-window overlap."""
    return (
        event.active
        & (event.start_ticks < start + duration)
        & (event.start_ticks + event.duration_ticks > start)
    )


@dataclass(frozen=True)
class _Config:
    """FIFO update with integer effective time and separate non-time values."""

    start: IntArray
    active: BoolArray
    frequency: NDArray[np.float64]
    phase: NDArray[np.float64]
    position: int


@dataclass(frozen=True)
class _Read:
    """Trigger time and demodulation values without mixed numeric storage."""

    start: IntArray
    frequency: NDArray[np.float64]
    phase: NDArray[np.float64]


class Outputs:
    """Accumulate slots by instruction, without allocating shot event records.

    routes is the SoC's explicit physical (generator, ADC) wiring. Each read uses
    its sole wired generator, or the unique wired generator overlapping its window.
    Each vector lane contributes one shot; inner loops retain iteration slots.
    """

    def __init__(
        self,
        prog: MyProgramV2,
        size: int,
        routes: Sequence[tuple[int, int]],
        pool: EnvelopePool,
        time_base: TimeBase,
    ) -> None:
        self.prog = prog
        self.size = size
        self.routes = tuple(routes)
        self.pool = pool
        self.time_base = time_base
        self.envelopes: dict[tuple[int, int, int], PulseEnvelope] = {}
        self.ports: dict[tuple[int, int | None], tuple[str, int]] = {}
        for ch in prog.gen_chs:
            cfg = prog.soccfg["gens"][ch]
            self._port(int(cfg["tproc_ch"]), cfg.get("tmux_ch"), "gen", ch)
        for ch in prog.ro_chs:
            ctrl = prog.soccfg["readouts"][ch].get("tproc_ctrl")
            if ctrl is not None:
                self._port(
                    int(ctrl), prog.soccfg["readouts"][ch].get("tmux_ch"), "ro", ch
                )
        self.iterations: tuple[tuple[str, int], ...] = ()
        self.loop_positions: dict[str, int] = {}
        self.events: dict[tuple[int, int, tuple[tuple[str, int], ...]], EventSlot] = {}
        # Absolute effective ticks stay separate from frequency/phase values.
        # Guards reject pending FIFO state across shots.
        self.configs: dict[int, list[_Config]] = {}
        self.reads: dict[int, list[_Read]] = {ch: [] for ch in prog.ro_chs}
        # Retain absolute tProc cycles for config admission without
        # subtracting and readding the per-shot origin.
        self.trigger_ticks: dict[int, list[IntArray]] = {ch: [] for ch in prog.ro_chs}
        self.duration = np.zeros(size, dtype=np.int64)
        self.visited = np.zeros(size, dtype=np.bool_)
        self.points = np.arange(size, dtype=np.int64)
        self.origin = np.zeros(size, dtype=np.int64)
        self.read_count: dict[int, IntArray] = {}

    def _ticks(self, cycles: IntArray, clock: float) -> IntArray:
        period = self.time_base.period_ticks(clock)
        if cycles.size:
            self.time_base.validate_ticks(int(cycles.min()) * period)
            self.time_base.validate_ticks(int(cycles.max()) * period)
        return cycles * period

    def _read_duration(self, ch: int) -> int:
        period = self.time_base.period_ticks(
            self.prog.soccfg["readouts"][ch]["f_output"]
        )
        duration = int(self.prog.ro_chs[ch]["length"]) * period
        self.time_base.validate_ticks(duration)
        return duration

    def _port(self, port: int, tmux_ch: int | None, kind: str, ch: int) -> None:
        key = (port, tmux_ch)
        if wave_port_conflicts(
            tmux_ch,
            (used_tmux for used_port, used_tmux in self.ports if used_port == port),
        ):
            raise ValueError(
                f"PortWriteInst WPORT_WR at preopt setup: ambiguous wave output port {port}, tmux_ch={tmux_ch}"
            )
        self.ports[key] = (kind, ch)

    def begin(self, machine: Machine, points: IntArray) -> None:
        """Start the current shot lanes at their actual reference-time origin."""
        self.points = points
        self.origin = machine.ref.copy()
        self.read_count = {
            ch: np.zeros(machine.size, dtype=np.int64) for ch in self.prog.ro_chs
        }
        self.prologue_configs = {ch: len(rows) for ch, rows in self.configs.items()}
        self.current_end = np.zeros(machine.size, dtype=np.int64)
        self.generator_end: dict[int, IntArray] = {}

    def _put(
        self,
        target: NDArray[np.float64] | IntArray,
        values: NDArray[np.float64] | IntArray,
        machine: Machine,
    ) -> None:
        target[self.points[machine.active]] = values[machine.active]

    def wave(
        self, machine: Machine, inst: PortWriteInst, position: int, source: str
    ) -> None:
        """Apply WPORT data at the actual port and current WMEM/register values."""
        port = machine.value(inst.dst)
        if np.any(port != port[0]):
            raise ValueError("lane-dependent wave output port")
        if inst.src == SrcKeyword.WMEM:
            words = machine.wave(inst.addr)
        elif inst.src == SrcKeyword.REG or (
            isinstance(inst.src, Register) and inst.src.canonical_name == "r_wave"
        ):
            words = np.stack(
                [machine.value(Register(f"w{i}")) for i in range(6)], axis=1
            )
        else:
            raise ValueError("unsupported wave port source")
        port_number = int(port[0])
        key = (port_number, None)
        if key not in self.ports:
            tmux = (words[machine.active, 5] >> 8) & 0xFF
            if np.any(tmux != tmux[0]):
                raise ValueError("lane-dependent tmux channel")
            key = (port_number, int(tmux[0]))
            if key not in self.ports:
                raise ValueError(
                    f"unregistered wave output port {port_number}, tmux_ch={key[1]}"
                )
        kind, ch = self.ports[key]
        timestamp = machine.ref + machine.value(
            inst.time if inst.time is not None else Register("s14")
        )
        start = self._ticks(
            timestamp if kind == "ro" else timestamp - self.origin,
            self.prog.tproccfg["f_time"],
        )
        cfg = self.prog.soccfg["gens" if kind == "gen" else "readouts"][ch]
        phase_bits = int(cfg.get("b_phase", 32))
        phase = (words[:, 1] % 2**phase_bits) * 360 / 2**phase_bits
        if kind == "ro":
            updates = self.configs.setdefault(ch, [])
            if updates:
                previous = np.maximum.reduce(
                    [
                        np.where(row.active, row.start, np.iinfo(np.int64).min)
                        for row in updates
                    ]
                )
                start = np.maximum(start, previous)
            frequency = (words[:, 0] % 2**32).astype(np.float64)
            updates.append(
                _Config(start, machine.active.copy(), frequency, phase, position)
            )
            return
        if "mux_tones" in self.prog.gen_chs[ch]:
            self._mux_wave(machine, ch, words, position, source, start)
            return
        self._ordinary_wave(machine, ch, words, position, source, start)

    def _ordinary_wave(
        self,
        machine: Machine,
        ch: int,
        words: IntArray,
        position: int,
        source: str,
        start: IntArray,
    ) -> None:
        cfg = self.prog.soccfg["gens"][ch]
        conf = words[:, 5] & 0xFFFF
        if np.any(machine.active & ((conf & 16) != 0)):
            raise ValueError("phrst=1 requires cross-command DDS phase state")
        if np.any(machine.active & ((conf & 4) != 0)):
            raise ValueError("mode=periodic cannot be represented by finite events")
        if np.any(machine.active & ((conf & 3) == 2)):
            raise ValueError("outsel=input has no numeric baseband contract")
        active = ((conf & 3) != 3) & machine.active
        phase_bits = int(cfg.get("b_phase", 32))
        phase = (words[:, 1] % 2**phase_bits) * 360 / 2**phase_bits
        freq = generator_frequencies_from_words(
            self.prog.soccfg,
            words[:, 0],
            gen_ch=ch,
            rounded_mixer_mhz=float(
                self.prog.gen_chs[ch].get("mixer_freq", {}).get("rounded", 0)
            ),
            absolute_freqs=self.prog.ABSOLUTE_FREQS,
        )
        duration = self._ticks(words[:, 4] & 65535, cfg["f_fabric"])
        # This serial playback model omits dispatcher/CDC/sample pipeline delays.
        start = np.maximum(start, self.generator_end.get(ch, start))
        self.generator_end[ch] = np.where(
            machine.active,
            start + duration,
            self.generator_end.get(ch, np.zeros(machine.size, dtype=np.int64)),
        )
        gain = ((words[:, 3] + 2**31) % 2**32 - 2**31) / float(cfg["maxv"])
        self.current_end = np.maximum(
            self.current_end, np.where(machine.active, start + duration, 0)
        )
        if not np.any(active):
            return
        key = (position, 0, self.iterations)
        self._emit(
            machine,
            key,
            ch,
            source,
            _EventValues(start, duration, freq, phase, gain, active),
        )
        slot = self.events[key]
        candidates = list(slot.envelopes)
        for address, length, conf in np.unique(words[:, [2, 4, 5]], axis=0):
            selected = (
                np.all(words[:, [2, 4, 5]] == (address, length, conf), axis=1)
                & machine.active
            )
            if not np.any(selected):
                continue
            envelope = self._envelope(ch, int(address), int(length), int(conf))
            if envelope not in candidates:
                candidates.append(envelope)
            indexes = self.points[selected & active]
            candidate = candidates.index(envelope)
            slot.envelope_index[indexes] = candidate
        self.events[key] = replace(slot, envelopes=tuple(candidates))

    def _mux_wave(
        self,
        machine: Machine,
        ch: int,
        words: IntArray,
        position: int,
        source: str,
        start: IntArray,
    ) -> None:
        cfg = self.prog.soccfg["gens"][ch]
        for tone, compiled in enumerate(self.prog.gen_chs[ch]["mux_tones"]):
            mask = ((words[:, 5] & (1 << tone)) != 0) & machine.active
            if not np.any(mask):
                continue
            duration = self._ticks(words[:, 4] & 0xFFFFFFFF, cfg["f_fabric"])
            frequency = np.full(
                machine.size, float(compiled["freq_rounded"]), dtype=np.float64
            )
            phase = np.full(
                machine.size, float(compiled["phase_rounded"]), dtype=np.float64
            )
            gain = np.full(
                machine.size, float(compiled["gain_rounded"]), dtype=np.float64
            )
            self._emit(
                machine,
                (position, tone, self.iterations),
                ch,
                source,
                _EventValues(start, duration, frequency, phase, gain, mask),
            )
            self.events[(position, tone, self.iterations)].active[self.points] = mask

    def _emit(
        self,
        machine: Machine,
        key: tuple[int, int, tuple[tuple[str, int], ...]],
        ch: int,
        source: str,
        values: _EventValues,
    ) -> None:
        if key not in self.events:
            zero = np.zeros(self.size, dtype=np.float64)
            self.events[key] = EventSlot(
                ch,
                source,
                self.iterations,
                np.zeros(self.size, dtype=np.int64),
                np.zeros(self.size, dtype=np.int64),
                zero.copy(),
                zero.copy(),
                zero.copy(),
                (None,),
                np.zeros(self.size, dtype=np.int64),
                np.zeros(self.size, dtype=np.bool_),
            )
        slot = self.events[key]
        for target, value in (
            (slot.start_ticks, values.start_ticks),
            (slot.duration_ticks, values.duration_ticks),
            (slot.frequency_mhz, values.frequency_mhz),
            (slot.phase_deg, values.phase_deg),
            (slot.gain, values.gain),
        ):
            self._put(target, np.where(values.active, value, 0), machine)
        slot.active[self.points] = values.active
        self.current_end = np.maximum(
            self.current_end,
            np.where(values.active, values.start_ticks + values.duration_ticks, 0),
        )

    def _envelope(
        self, ch: int, address: int, length: int, conf: int
    ) -> PulseEnvelope | None:
        if conf & 3 in (1, 3):
            if not conf & 8 and conf & 3 == 1:
                raise ValueError("stdysel=last has nonzero final DDS envelope sample")
            return None
        if conf & 3 != 0:
            raise ValueError(f"unmodeled generator output selection {conf & 3}")
        cfg = self.prog.soccfg["gens"][ch]
        spc = int(cfg["samps_per_clk"])
        cache = (ch, address, length)
        if cache in self.envelopes:
            envelope = self.envelopes[cache]
            self._steady_output(conf, envelope)
            return envelope
        start, count = (address & 0xFFFFFF) * spc, (length & 0xFFFF) * spc
        envelopes = self.prog.envelopes[ch]["envs"]
        if not isinstance(envelopes, dict):
            raise ValueError("compiled DAC envelope table is missing")
        for entry in envelopes.values():
            offset = start - int(entry["addr"])
            data = entry["data"]
            if offset >= 0 and offset + count <= len(data):
                segment = data[offset : offset + count]
                maxv = self.prog.soccfg.get_maxv(ch)
                clock = float(cfg["f_fabric"])
                key = EnvelopeContent(
                    segment.dtype.str,
                    segment.shape,
                    segment.tobytes(),
                    maxv,
                    clock,
                    self.time_base.period_ticks(clock * spc),
                    None,
                )
                envelope = self.pool.get(key)
                if envelope is None:
                    values = (segment[:, 0] + 1j * segment[:, 1]) / maxv
                    envelope = PulseEnvelope(
                        self.time_base.period_ticks(clock * spc),
                        tuple(complex(v) for v in values),
                    )
                    self.pool.put(key, envelope)
                self._steady_output(conf, envelope)
                self.envelopes[cache] = envelope
                return envelope
        raise ValueError(f"undefined DAC envelope range [{start}, {start + count})")

    def _steady_output(self, conf: int, envelope: PulseEnvelope) -> None:
        if not conf & 8 and envelope.values[-1] != 0:
            raise ValueError("stdysel=last has nonzero final DAC envelope sample")

    def trigger(self, machine: Machine, inst: DportWriteInst | TrigInst) -> None:
        """Record dport masks or tport sets; clears do not create ADC reads."""
        port = machine.value(inst.dst)
        if np.any(port != port[0]):
            raise ValueError("lane-dependent trigger port")
        if isinstance(inst, TrigInst) and inst.src == "clr":
            return
        timestamp = machine.ref + machine.value(
            inst.time if inst.time is not None else Register("s14")
        )
        start = self._ticks(timestamp - self.origin, self.prog.tproccfg["f_time"])
        for ch in self.prog.ro_chs:
            cfg = self.prog.soccfg["readouts"][ch]
            if int(cfg["trigger_port"]) != int(port[0]):
                continue
            if isinstance(inst, DportWriteInst):
                if cfg["trigger_type"] != "dport":
                    continue
                data = machine.value(inst.data)
                active = machine.active & ((data & (1 << int(cfg["trigger_bit"]))) != 0)
            else:
                if cfg["trigger_type"] != "tport":
                    continue
                active = machine.active.copy()
            if not np.any(active):
                continue
            indexes = self.read_count[ch]
            index = int(indexes[active][0])
            if np.any(indexes[active] != index):
                raise ValueError("lane-dependent trigger layout")
            while len(self.reads[ch]) <= index:
                self.reads[ch].append(
                    _Read(
                        np.zeros(self.size, dtype=np.int64),
                        np.zeros(self.size, dtype=np.float64),
                        np.zeros(self.size, dtype=np.float64),
                    )
                )
                self.trigger_ticks[ch].append(np.zeros(self.size, dtype=np.int64))
            self.trigger_ticks[ch][index][self.points[active]] = timestamp[active]
            read = self.reads[ch][index]
            read.start[self.points[active]] = start[active]
            if cfg.get("tproc_ctrl") is None:
                gen = self.prog.ro_chs[ch]["gen_ch"]
                mixer = float(
                    self.prog.gen_chs[gen].get("mixer_freq", {}).get("rounded", 0)
                )
                frequency, phase = static_readout_parameters(
                    self.prog, ch, mixer_mhz=mixer
                )
                read.frequency[self.points[active]] = frequency
                read.phase[self.points[active]] = phase
            self.read_count[ch] += active
            self.current_end = np.maximum(
                self.current_end,
                np.where(
                    active,
                    start + self._read_duration(ch),
                    0,
                ),
            )

    def finish(self, machine: Machine) -> None:
        """Close a shot, validating per-point output presence and trigger counts."""
        for ch, count in self.read_count.items():
            mismatch = count != self.prog.ro_chs[ch]["trigs"]
            if np.any(mismatch):
                actual = int(count[np.flatnonzero(mismatch)[0]])
                raise ValueError(
                    f"ADC {ch}: static={self.prog.ro_chs[ch]['trigs']}, actual={actual} reads in this shot"
                )
        self._resolve_configs(machine)
        end = self._ticks(machine.ref - self.origin, self.prog.tproccfg["f_time"])
        self._put(self.duration, np.maximum(self.current_end, end), machine)
        self.visited[self.points] = True

    def _resolve_configs(self, machine: Machine) -> None:
        for ch, reads in self.reads.items():
            if self.prog.soccfg["readouts"][ch].get("tproc_ctrl") is None:
                continue
            duration = self._read_duration(ch)
            for index, read in enumerate(reads):
                start = self._ticks(
                    self.trigger_ticks[ch][index][self.points],
                    self.prog.tproccfg["f_time"],
                )
                frequency = np.full(machine.size, np.nan)
                phase = np.full(machine.size, np.nan)
                selected = np.full(machine.size, -1, dtype=np.int64)
                rows = self.configs.get(ch, [])
                for config_index, config in enumerate(rows):
                    active = (config.start <= start) & config.active
                    frequency[active] = config.frequency[active]
                    phase[active] = config.phase[active]
                    selected[active] = config_index
                if np.any(np.isnan(frequency)):
                    lane = int(np.flatnonzero(np.isnan(frequency))[0])
                    raise ValueError(
                        f"ADC {ch} read {index} point {self.points[lane]}: no effective runtime config at window start {start[lane]}"
                    )
                prologue = self.prologue_configs.get(ch, 0)
                if (
                    self.size * self.prog.loop_dict["reps"] > 1
                    and len(rows) > prologue
                    and np.any(selected < prologue)
                ):
                    raise ValueError(
                        f"PortWriteInst at preopt ADC {ch}: cross-shot config live-in"
                    )
                for config in rows:
                    changed = (config.frequency != frequency) | (config.phase != phase)
                    conflict = (
                        config.active
                        & (start < config.start)
                        & (config.start < start + duration)
                        & changed
                    )
                    if np.any(conflict):
                        lane = int(np.flatnonzero(conflict)[0])
                        raise ValueError(
                            f"ADC {ch} read {index} window [{start[lane]}, {start[lane] + duration}): "
                            f"config effective at {config.start[lane]}, "
                            f"PortWriteInst WPORT_WR at preopt {config.position}; "
                            f"initial=(frequency_word={frequency[lane]}, phase={phase[lane]}), "
                            f"updated=(frequency_word={config.frequency[lane]}, phase={config.phase[lane]})"
                        )
                self._put(read.frequency, frequency, machine)
                self._put(read.phase, phase, machine)

    def validate_fifo(self, machine: Machine) -> None:
        """Reject mutable readout FIFO updates pending beyond this shot.

        Prologue-only configurations remain constants. Once a port is written
        in the shot, every read must select a shot-local update and the FIFO
        must drain before the following shot, even for equal-valued updates.
        """
        if self.size * self.prog.loop_dict["reps"] <= 1:
            return
        for ch, rows in self.configs.items():
            prologue = self.prologue_configs.get(ch, 0)
            if len(rows) > prologue and np.any(
                np.maximum.reduce(
                    [
                        np.where(row.active, row.start, np.iinfo(np.int64).min)
                        for row in rows[prologue:]
                    ]
                )
                > self._ticks(machine.ref, self.prog.tproccfg["f_time"])
            ):
                raise ValueError(
                    f"PortWriteInst at preopt ADC {ch}: cross-shot pending config FIFO"
                )

    def template(self) -> TimelineTemplate:
        """Return detached point arrays after resolving every read's physical route."""
        if not np.all(self.visited):
            raise ValueError("compiled loops did not visit every point")
        slots = []
        for ch, reads in self.reads.items():
            cfg = self.prog.soccfg["readouts"][ch]
            starts = np.stack([read.start for read in reads], axis=1)
            frequencies = np.stack([read.frequency for read in reads], axis=1)
            phases = np.stack([read.phase for read in reads], axis=1)
            routes = self._routes(ch, starts)
            if cfg.get("tproc_ctrl") is not None:
                frequencies = self._runtime_frequencies(ch, starts, frequencies, routes)
            slots.append(
                ReadoutSlot(
                    ch,
                    int(self.prog.ro_chs[ch]["length"]),
                    self.time_base.period_ticks(cfg["f_output"]),
                    starts,
                    frequencies,
                    phases,
                    routes,
                )
            )
        return TimelineTemplate(
            np.arange(self.size, dtype=np.int64),
            tuple(
                self.events[key] for key in sorted(self.events, key=self._event_order)
            ),
            tuple(slots),
            self.duration.copy(),
        )

    def _event_order(
        self, key: tuple[int, int, tuple[tuple[str, int], ...]]
    ) -> tuple[tuple[int, int], ...]:
        position, tone, iterations = key
        return tuple(
            (self.loop_positions[name], index) for name, index in iterations
        ) + ((position, tone),)

    def _runtime_frequencies(
        self,
        ch: int,
        starts: IntArray,
        frequencies: NDArray[np.float64],
        routes: IntArray,
    ) -> NDArray[np.float64]:
        result = np.full(starts.shape, np.nan, dtype=np.float64)
        first_carrier = np.zeros(starts.shape, dtype=np.float64)
        duration = self._read_duration(ch)
        for read in range(starts.shape[1]):
            start = starts[:, read]
            words = frequencies[:, read].astype(np.int64)
            for event in self.events.values():
                active = _overlaps_window(event, start, duration) & (
                    event.gen_ch == routes[:, read]
                )
                if not np.any(active):
                    continue
                images = readout_frequencies_from_words(
                    self.prog.soccfg,
                    words,
                    ro_ch=ch,
                    reference_freq_mhz=event.frequency_mhz,
                    flip_downconversion=self.prog.FLIP_DOWNCONVERSION,
                )
                conflict = (
                    active & np.isfinite(result[:, read]) & (result[:, read] != images)
                )
                if np.any(conflict):
                    point = int(np.flatnonzero(conflict)[0])
                    raise ValueError(
                        f"ADC {ch} read {read} point {point}: carriers={[first_carrier[point, read], event.frequency_mhz[point]]}, images={[result[point, read], images[point]]}"
                    )
                result[active, read] = images[active]
                first_carrier[active, read] = event.frequency_mhz[active]
            missing = ~np.isfinite(result[:, read])
            if np.any(missing):
                point = int(np.flatnonzero(missing)[0])
                raise ValueError(
                    f"ADC {ch} read {read} point {point}: no pulse carrier in acquisition window [{start[point]}, {start[point] + duration})"
                )
        return result

    def _routes(self, ch: int, times: IntArray) -> IntArray:
        wired = sorted({gen for gen, adc in self.routes if adc == ch})
        result = np.empty(times.shape, dtype=np.int64)
        if len(wired) == 1:
            result.fill(wired[0])
            return result
        duration = self._read_duration(ch)
        for read in range(times.shape[1]):
            candidates = np.zeros((self.size, len(wired)), dtype=np.bool_)
            for index, gen in enumerate(wired):
                for event in self.events.values():
                    if event.gen_ch == gen:
                        candidates[:, index] |= _overlaps_window(
                            event, times[:, read], duration
                        )
            count = candidates.sum(axis=1)
            if np.any(count != 1):
                point = int(np.flatnonzero(count != 1)[0])
                selected = [
                    g
                    for g, active in zip(wired, candidates[point], strict=True)
                    if active
                ]
                reason = "no wired generator route; " if not wired else ""
                raise ValueError(
                    f"ADC {ch} read {read} point {point}: {reason}generator candidates={selected}, wired={wired}, time=[{times[point, read]}, {times[point, read] + duration})"
                )
            result[:, read] = np.asarray(wired)[np.argmax(candidates, axis=1)]
        return result
