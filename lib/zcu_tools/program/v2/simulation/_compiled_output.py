"""Group compiled output effects into detached numeric point slots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    EventSlot,
    PulseEnvelope,
    ReadoutSlot,
    TimelineTemplate,
)

from ..base import MyProgramV2
from ..ir.io_instructions import DportWriteInst, PortWriteInst
from ..ir.operands import Register, SrcKeyword
from ..utils import generator_frequencies_from_words, readout_frequencies_from_words
from ._compiled_state import IntArray, Machine
from ._envelope_pool import EnvelopeContent, EnvelopePool
from ._mux import static_readout_parameters


class Outputs:
    """Accumulate slots by instruction, without allocating shot event records.

    routes is the SoC's explicit physical (generator, ADC) wiring. Each read uses
    its sole wired generator, or the unique wired generator overlapping its window.
    Multiple visits to one point must produce identical observable outputs.
    """

    def __init__(
        self,
        prog: MyProgramV2,
        size: int,
        routes: Sequence[tuple[int, int]],
        pool: EnvelopePool,
    ) -> None:
        self.prog = prog
        self.size = size
        self.routes = tuple(routes)
        self.pool = pool
        self.envelopes: dict[tuple[int, int, int], PulseEnvelope] = {}
        self.ports: dict[int, tuple[str, int]] = {}
        for ch in prog.gen_chs:
            self._port(int(prog.soccfg["gens"][ch]["tproc_ch"]), "gen", ch)
        for ch in prog.ro_chs:
            ctrl = prog.soccfg["readouts"][ch].get("tproc_ctrl")
            if ctrl is not None:
                self._port(int(ctrl), "ro", ch)
        self.events: dict[tuple[int, int], EventSlot] = {}
        self.configs: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        self.reads: dict[int, list[NDArray[np.float64]]] = {
            ch: [] for ch in prog.ro_chs
        }
        self.duration = np.zeros(size, dtype=np.float64)
        self.visited = np.zeros(size, dtype=np.bool_)
        self.points = np.arange(size, dtype=np.int64)
        self.origin = np.zeros(size, dtype=np.int64)
        self.read_count: dict[int, int] = {}
        self.shot_events: set[tuple[int, int]] = set()

    def _port(self, port: int, kind: str, ch: int) -> None:
        if port in self.ports:
            raise ValueError(
                f"PortWriteInst WPORT_WR at preopt setup: ambiguous wave output port {port}"
            )
        self.ports[port] = (kind, ch)

    def begin(self, machine: Machine, points: IntArray) -> None:
        """Start the current shot lanes at their actual reference-time origin."""
        self.points = points
        self.origin = machine.ref.copy()
        self.read_count = dict.fromkeys(self.prog.ro_chs, 0)
        self.shot_events = set()
        self.current_end = np.zeros(machine.size, dtype=np.float64)

    def _put(
        self, target: NDArray[np.float64], values: NDArray[np.float64], machine: Machine
    ) -> None:
        repeated = self.visited[self.points]
        if np.any(repeated & (target[self.points] != values)):
            changed = ", ".join(sorted(machine.writes))
            raise ValueError(
                f"reps loop changes observable output; carried register/memory state: {changed}"
            )
        target[self.points] = values

    def wave(
        self, machine: Machine, inst: PortWriteInst, position: int, source: str
    ) -> None:
        """Apply WPORT data at the actual port and current WMEM/register values."""
        port = machine.value(inst.dst)
        if np.any(port != port[0]):
            raise ValueError("lane-dependent wave output port")
        kind, ch = self.ports[int(port[0])]
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
        start = (
            machine.ref
            - self.origin
            + machine.value(inst.time if inst.time is not None else Register("s14"))
        ) / float(self.prog.tproccfg["f_time"])
        cfg = self.prog.soccfg["gens" if kind == "gen" else "readouts"][ch]
        bits = int(cfg["b_dds"])
        freq = (words[:, 0] % 2**bits) * float(cfg["f_dds"]) / 2**bits
        phase_bits = int(cfg.get("b_phase", 32))
        phase = (words[:, 1] % 2**phase_bits) * 360 / 2**phase_bits
        if kind == "ro":
            self.configs[ch] = (words[:, 0].astype(np.float64), phase)
            return
        if "mux_tones" in self.prog.gen_chs[ch]:
            self._mux_wave(machine, ch, words, position, source, start)
            return
        freq = generator_frequencies_from_words(
            self.prog.soccfg,
            words[:, 0],
            gen_ch=ch,
            rounded_mixer_mhz=float(
                self.prog.gen_chs[ch].get("mixer_freq", {}).get("rounded", 0)
            ),
            absolute_freqs=self.prog.ABSOLUTE_FREQS,
        )
        duration = (words[:, 4] & 65535) / float(cfg["f_fabric"])
        gain = ((words[:, 3] + 2**31) % 2**32 - 2**31) / float(cfg["maxv"])
        key = (position, 0)
        self._emit(machine, key, ch, source, (start, duration, freq, phase, gain))
        slot = self.events[key]
        candidates = list(slot.envelopes)
        for address, length, conf in np.unique(words[:, [2, 4, 5]], axis=0):
            selected = np.all(words[:, [2, 4, 5]] == (address, length, conf), axis=1)
            envelope = self._envelope(ch, int(address), int(length), int(conf))
            if envelope not in candidates:
                candidates.append(envelope)
            slot.envelope_index[self.points[selected]] = candidates.index(envelope)
        self.events[key] = replace(slot, envelopes=tuple(candidates))

    def _mux_wave(
        self,
        machine: Machine,
        ch: int,
        words: IntArray,
        position: int,
        source: str,
        start: NDArray[np.float64],
    ) -> None:
        cfg = self.prog.soccfg["gens"][ch]
        for tone, compiled in enumerate(self.prog.gen_chs[ch]["mux_tones"]):
            mask = (words[:, 5] & (1 << tone)) != 0
            if not np.any(mask):
                continue
            duration = (words[:, 4] & 0xFFFFFFFF) / float(cfg["f_fabric"])
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
                (position, tone),
                ch,
                source,
                (start, duration, frequency, phase, gain),
            )
            self.events[(position, tone)].active[self.points] = mask

    def _emit(
        self,
        machine: Machine,
        key: tuple[int, int],
        ch: int,
        source: str,
        values: tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        if key not in self.events:
            zero = np.zeros(self.size, dtype=np.float64)
            self.events[key] = EventSlot(
                ch,
                source,
                (),
                zero.copy(),
                zero.copy(),
                zero.copy(),
                zero.copy(),
                zero.copy(),
                (None,),
                np.zeros(self.size, dtype=np.int64),
                np.zeros(self.size, dtype=np.bool_),
            )
        slot = self.events[key]
        for target, value in zip(
            (
                slot.start_us,
                slot.duration_us,
                slot.frequency_mhz,
                slot.phase_deg,
                slot.gain,
            ),
            values,
            strict=True,
        ):
            self._put(target, value, machine)
        slot.active[self.points] = True
        self.shot_events.add(key)
        self.current_end = np.maximum(self.current_end, values[0] + values[1])

    def _envelope(
        self, ch: int, address: int, length: int, conf: int
    ) -> PulseEnvelope | None:
        if conf & 3 == 1:
            return None
        if conf & 3 != 0:
            raise ValueError(f"unmodeled generator output selection {conf & 3}")
        cfg = self.prog.soccfg["gens"][ch]
        spc = int(cfg["samps_per_clk"])
        cache = (ch, address, length)
        if cache in self.envelopes:
            return self.envelopes[cache]
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
                    spc,
                    None,
                )
                envelope = self.pool.get(key)
                if envelope is None:
                    values = (segment[:, 0] + 1j * segment[:, 1]) / maxv
                    envelope = PulseEnvelope(
                        tuple(float(t) for t in np.arange(count) / (clock * spc)),
                        tuple(complex(v) for v in values),
                    )
                    self.pool.put(key, envelope)
                self.envelopes[cache] = envelope
                return envelope
        raise ValueError(f"undefined DAC envelope range [{start}, {start + count})")

    def trigger(self, machine: Machine, inst: DportWriteInst) -> None:
        """Record ADC rising trigger effects, preserving declaration/read order."""
        port, data = machine.value(inst.dst), machine.value(inst.data)
        if np.any(port != port[0]):
            raise ValueError("lane-dependent trigger port")
        start = (
            machine.ref
            - self.origin
            + machine.value(inst.time if inst.time is not None else Register("s14"))
        ) / float(self.prog.tproccfg["f_time"])
        for ch in self.prog.ro_chs:
            cfg = self.prog.soccfg["readouts"][ch]
            active = (data & (1 << int(cfg["trigger_bit"]))) != 0
            if int(cfg["trigger_port"]) != int(port[0]) or not np.any(active):
                continue
            if not np.all(active):
                raise ValueError("lane-dependent trigger layout")
            index = self.read_count[ch]
            if index >= self.prog.ro_chs[ch]["trigs"]:
                raise ValueError(
                    f"ADC {ch}: static={self.prog.ro_chs[ch]['trigs']}, actual={index + 1} reads in this shot"
                )
            if ch not in self.configs and cfg.get("tproc_ctrl") is None:
                gen = self.prog.ro_chs[ch]["gen_ch"]
                mixer = float(
                    self.prog.gen_chs[gen].get("mixer_freq", {}).get("rounded", 0)
                )
                frequency, phase = static_readout_parameters(
                    self.prog, ch, mixer_mhz=mixer
                )
                self.configs[ch] = (
                    np.full(machine.size, frequency, dtype=np.float64),
                    np.full(machine.size, phase, dtype=np.float64),
                )
            frequency, phase = self.configs[ch]
            while len(self.reads[ch]) <= index:
                self.reads[ch].append(np.zeros((self.size, 3), dtype=np.float64))
            read = self.reads[ch][index]
            for col, values in enumerate((start, frequency, phase)):
                self._put(read[:, col], values, machine)
            self.read_count[ch] += 1
            self.current_end = np.maximum(
                self.current_end,
                start + self.prog.ro_chs[ch]["length"] / float(cfg["f_output"]),
            )

    def finish(self, machine: Machine) -> None:
        """Close a shot, validating per-point output presence and trigger counts."""
        for ch, count in self.read_count.items():
            if count != self.prog.ro_chs[ch]["trigs"]:
                raise ValueError(
                    f"ADC {ch}: static={self.prog.ro_chs[ch]['trigs']}, actual={count} reads in this shot"
                )
        end = (machine.ref - self.origin) / float(self.prog.tproccfg["f_time"])
        for pos, slot in self.events.items():
            if (
                np.any(self.visited[self.points] & slot.active[self.points])
                and pos not in self.shot_events
            ):
                raise ValueError("reps loop changes pulse presence")
        self._put(self.duration, np.maximum(self.current_end, end), machine)
        self.visited[self.points] = True

    def template(self) -> TimelineTemplate:
        """Return detached point arrays after resolving every read's physical route."""
        if not np.all(self.visited):
            raise ValueError("compiled loops did not visit every point")
        slots = []
        for ch, reads in self.reads.items():
            cfg = self.prog.soccfg["readouts"][ch]
            values = np.stack(reads, axis=1)
            routes = self._routes(ch, values[:, :, 0])
            if cfg.get("tproc_ctrl") is not None:
                values[:, :, 1] = self._runtime_frequencies(ch, values, routes)
            slots.append(
                ReadoutSlot(
                    ch,
                    int(self.prog.ro_chs[ch]["length"]),
                    float(cfg["f_output"]),
                    values[:, :, 0].copy(),
                    values[:, :, 1].copy(),
                    values[:, :, 2].copy(),
                    routes,
                )
            )
        return TimelineTemplate(
            np.arange(self.size, dtype=np.int64),
            tuple(self.events.values()),
            tuple(slots),
            self.duration.copy(),
        )

    def _runtime_frequencies(
        self, ch: int, values: NDArray[np.float64], routes: IntArray
    ) -> NDArray[np.float64]:
        result = np.full(values.shape[:2], np.nan, dtype=np.float64)
        first_carrier = np.zeros(values.shape[:2], dtype=np.float64)
        cfg = self.prog.soccfg["readouts"][ch]
        duration = self.prog.ro_chs[ch]["length"] / float(cfg["f_output"])
        for read in range(values.shape[1]):
            start = values[:, read, 0]
            words = values[:, read, 1].astype(np.int64)
            for event in self.events.values():
                active = (
                    event.active
                    & (event.gen_ch == routes[:, read])
                    & (event.start_us < start + duration)
                    & (event.start_us + event.duration_us > start)
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

    def _routes(self, ch: int, times: NDArray[np.float64]) -> IntArray:
        wired = sorted({gen for gen, adc in self.routes if adc == ch})
        result = np.empty(times.shape, dtype=np.int64)
        if len(wired) == 1:
            result.fill(wired[0])
            return result
        duration = self.prog.ro_chs[ch]["length"] / float(
            self.prog.soccfg["readouts"][ch]["f_output"]
        )
        for read in range(times.shape[1]):
            candidates = np.zeros((self.size, len(wired)), dtype=np.bool_)
            for index, gen in enumerate(wired):
                for event in self.events.values():
                    if event.gen_ch == gen:
                        candidates[:, index] |= (
                            event.active
                            & (event.start_us < times[:, read] + duration)
                            & (event.start_us + event.duration_us > times[:, read])
                        )
            count = candidates.sum(axis=1)
            if np.any(count != 1):
                point = int(np.flatnonzero(count != 1)[0])
                selected = [
                    g
                    for g, active in zip(wired, candidates[point], strict=True)
                    if active
                ]
                raise ValueError(
                    f"ADC {ch} read {read} point {point}: generator candidates={selected}, wired={wired}, time=[{times[point, read]}, {times[point, read] + duration})"
                )
            result[:, read] = np.asarray(wired)[np.argmax(candidates, axis=1)]
        return result
