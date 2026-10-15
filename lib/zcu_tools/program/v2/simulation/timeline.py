"""Module scheduling and quantized numeric controls, without physical models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
from qick.asm_v2 import QickParam, QickRawParam

from zcu_tools.core.simulation import (
    PointTimeline,
    PulseEnvelope,
    PulseEvent,
    ReadoutWindow,
)

from ..modular import ModularProgramV2
from ..modules.base import Module
from ..modules.computed_pulse import ComputedPulse
from ..modules.control import Branch, Repeat
from ..modules.delay import Delay, DelayAuto, Join, SoftDelay
from ..modules.dmem import LoadValue, LoadWord, ScanWith
from ..modules.pulse import Pulse, TableLengthPulse
from ..modules.readout import PulseReadout, TablePulseReadout
from ..modules.reset import BathReset, NoneReset, PulseReset, TwoPulseReset
from ..modules.util import round_timestamp
from ..modules.waveform import ConstWaveformCfg, FlatTopWaveformCfg
from ..utils import is_qick_param, readout_freq_from_word, readout_freq_from_words
from .dependencies import register_access, validate_loop_hooks, validate_modules


def _minimum(value: float | QickParam) -> float:
    if is_qick_param(value):
        return value.minval()
    return float(value)


class TimelineInterpreter:
    """Interpret a compiled ModularProgramV2 into detached execution timelines.

    prog is borrowed for this preparation only; no Module reaches the returned
    records. Each instance owns its reference clock, channel clocks and registers.
    Unsupported modules raise ValueError naming the module and loop coordinates
    before any backend call. This is a Module interpreter, not an ISA emulator.
    """

    def __init__(self, prog: ModularProgramV2) -> None:
        """Borrow a compiled program and initialize isolated execution state."""
        self.prog = prog
        self.point: dict[str, int] = {}
        self.ref = 0.0
        self.events: list[PulseEvent] = []
        self.reads: dict[int, list[tuple[float, float, float]]] = {}
        self.gen_end: dict[int, float] = {}
        self.ro_end: dict[int, float] = {}
        self.routes: dict[int, int] = {}
        self.registers: dict[str, tuple[int, bool]] = {}
        self.iterations: list[tuple[str, int]] = []
        self.templates: dict[int, tuple[int, int]] = {}
        tables = validate_modules(prog.modules)
        validate_loop_hooks(prog.loops, tables)
        self.table_readouts = {id(module): module for module in tables}

    def scalar(self, value: float | QickParam) -> float:
        """Resolve a scalar/sweep in us, MHz, degrees or normalized gain.

        QickParam values passed here are already quantized where a board grid
        applies. Each span uses its declared hardware-loop zero-based coordinate.
        Unknown loop names fail rather than assuming their first point.
        """
        if not isinstance(value, QickParam):
            return float(value)
        result = float(value.start)
        for name, span in value.spans.items():
            count = self.prog.loop_dict[name]
            result += span * self.point[name] / (count - 1) if count > 1 else 0
        return float(result)

    def tick(self, value: float | QickParam) -> float:
        """Return QICK nearest processor-tick quantization in us at this point."""
        cycles = self.prog.us2cycles(deepcopy(value))
        if isinstance(cycles, QickRawParam):
            cycles.to_steps(self.prog.loop_dict)
            return self.scalar(cycles / float(self.prog.tproccfg["f_time"]))
        return float(self.prog.cycles2us(cycles))

    def run_point(self, point: dict[str, int]) -> PointTimeline:
        """Evaluate one loop coordinate and return owned immutable controls."""
        self.point = point.copy()
        self.registers.update({name: (index, False) for name, index in point.items()})
        self.ref = 0.0
        self.events = []
        self.reads = {ch: [] for ch in self.prog.ro_chs}
        self.gen_end = {ch: 0.0 for ch in self.prog.gen_chs}
        self.ro_end = {ch: 0.0 for ch in self.prog.ro_chs}
        t = self.sequence(self.prog.modules, 0.0)
        windows = tuple(
            ReadoutWindow(
                ch,
                self.routes[ch],
                int(ro["length"]),
                float(self.prog.soccfg["readouts"][ch]["f_output"]),
                tuple(r[0] for r in self.reads[ch]),
                tuple(r[1] for r in self.reads[ch]),
                tuple(r[2] for r in self.reads[ch]),
            )
            for ch, ro in self.prog.ro_chs.items()
        )
        end = max(
            self.ref + self.tick(t),
            *self.gen_end.values(),
            *self.ro_end.values(),
            *(event.start_us + event.duration_us for event in self.events),
        )
        return PointTimeline(tuple(point.items()), tuple(self.events), windows, end)

    def sequence(self, modules, t: float | QickParam) -> float | QickParam:
        """Run a semantic sequence at local us t and return its local end time."""
        for module in modules:
            try:
                t = self.run(module, t)
            except (ValueError, KeyError, IndexError) as exc:
                raise ValueError(
                    f"{type(module).__name__}({module.name!r}) at {self.point}: {exc}"
                ) from exc
        return t

    def run(self, module: Module, t: float | QickParam) -> float | QickParam:
        """Dispatch exact supported Module types; reject unknown subclasses."""
        if type(module) is Pulse:
            return self.pulse(module, t)
        if type(module) is PulseReadout:
            return self.readout(module, t)
        return self.timing(module, t)

    def timing(self, module: Module, t: float | QickParam) -> float | QickParam:
        """Apply explicit delays, reference waits and parallel branch timing."""
        if type(module) is SoftDelay:
            return round_timestamp(self.prog, t + module.delay)
        if type(module) is Delay:
            self.ref += self.tick(round_timestamp(self.prog, t + module.delay))
            return 0.0
        if type(module) is DelayAuto:
            extra = (
                self.prog.cycles2us(self.register(module.t))
                if isinstance(module.t, str)
                else self.scalar(module.t)
            )
            self.align(extra, gens=module.gens, ros=module.ros)
            return 0.0
        if type(module) is Join:
            self.check_join(module)
            return self.join(module, t)
        return self.control(module, t)

    def register(self, name: str, *, word: bool = False) -> int:
        """Read an initialized scalar or exact uint32 word with consumer checking.

        Scalar/word mismatches and undefined registers raise ValueError naming
        the register; scalar consumers never reinterpret unsigned bit patterns.
        """
        if name not in self.registers:
            raise ValueError(f"undefined register {name!r}")
        value, is_word = self.registers[name]
        if word != is_word:
            raise ValueError(f"register {name!r}: unsupported word/scalar consumer")
        return value

    def control(self, module: Module, t: float | QickParam) -> float | QickParam:
        """Interpret table and structured-control Modules without executing ISA."""
        if isinstance(module, (LoadValue, LoadWord)) and type(module) in (
            LoadValue,
            LoadWord,
        ):
            index = self.register(module.idx_reg)
            if not 0 <= index < len(module.values):
                raise ValueError(
                    f"table index {index} out of bounds via register {module.idx_reg!r}"
                )
            self.registers[module.val_reg] = (
                module.values[index],
                type(module) is LoadWord,
            )
            return t
        if type(module) is ScanWith:
            return self.repeat(module.repeat_mod, t)
        if type(module) is Repeat:
            return self.repeat(module, t)
        if type(module) is Branch:
            self.ref += self.tick(t)
            self.align()
            index = self.register(module.compare_reg)
            if not 0 <= index < len(module.branches):
                raise ValueError(
                    f"branch index {index} out of bounds via register {module.compare_reg!r}"
                )
            end = self.sequence(module.branches[index], 0.0)
            self.ref += self.tick(end)
            self.align()
            return 0.0
        return self.selected_pulse(module, t)

    def selected_pulse(self, module: Module, t: float | QickParam) -> float | QickParam:
        """Resolve computed/table pulses and readout templates before playback."""
        if type(module) is ComputedPulse:
            index = self.register(module.val_reg)
            if not 0 <= index < len(module.pulse_modules):
                raise ValueError(
                    f"candidate {index} out of bounds via register {module.val_reg!r}"
                )
            self.pulse(module.pulse_modules[index], t)
            return t + module.total_length(self.prog)
        if type(module) is TableLengthPulse:
            return self.table_length(module, t)
        if type(module) is TablePulseReadout:
            return self.table_readout(module, t)
        return self.reset(module, t)

    def repeat(self, module: Repeat, t: float | QickParam) -> float:
        """Flush pending activity and execute a deterministic register-count loop."""
        self.ref += self.tick(t)
        self.align()
        n = self.register(module.n) if isinstance(module.n, str) else module.n
        if n < 0:
            raise ValueError(f"negative repeat count {n}")
        for index in range(n):
            self.registers[module.counter_reg] = (index, False)
            self.iterations.append((module.name, index))
            end = self.sequence(module.sub_modules, 0.0)
            self.iterations.pop()
            self.ref += self.tick(end)
            self.align()
        self.registers[module.counter_reg] = (n, False)
        return 0.0

    def reset(self, module: Module, t: float | QickParam) -> float | QickParam:
        """Expand reset wrappers into real pulses only; never set a quantum state."""
        if type(module) is NoneReset:
            return t
        if type(module) is PulseReset:
            return self.pulse(module.reset_pulse, t)
        if type(module) is TwoPulseReset:
            return max(
                self.pulse(module.reset_pulse1, t), self.pulse(module.reset_pulse2, t)
            )
        if type(module) is BathReset:
            end = self.pulse(module.res_pulse, t)
            self.pulse(module.qub_pulse, t)
            return self.pulse(module.pi2_pulse, end)
        raise ValueError(f"{type(module).__name__} simulation unsupported")

    def table_length(self, module: TableLengthPulse, t: float | QickParam) -> float:
        """Apply selected generator length and runtime duration, with no max padding."""
        index = self.register(module.idx_reg)
        if not 0 <= index < len(module.lengths):
            raise ValueError(
                f"table index {index} out of bounds via register {module.idx_reg!r}"
            )
        template, cycles, duration_cycles = module.playback_parameters(self.prog, index)
        duration = self.prog.cycles2us(cycles, gen_ch=module.cfg.ch)
        if isinstance(module.cfg.waveform, FlatTopWaveformCfg):
            template_length = self.scalar(
                self.prog.get_pulse_param(template.pulse_id, "length")
            )
            total = self.scalar(
                self.prog.get_pulse_param(template.pulse_id, "total_length")
            )
            duration += total - template_length
        previous_end = self.gen_end.get(module.cfg.ch, 0.0)
        self.pulse(template, t, duration=duration)
        # PulseFromLengthReg deliberately does not update compile-time timestamps.
        # Its duration table, not template length, owns the reference increment.
        self.gen_end[module.cfg.ch] = previous_end
        self.align(self.prog.cycles2us(duration_cycles))
        return 0.0

    def table_readout(
        self, module: TablePulseReadout, t: float | QickParam
    ) -> float | QickParam:
        """Play the current template; only exec-after mutates it for the next point."""
        self.table_readouts[id(module)] = module
        words = self.templates.get(id(module))
        return self.readout(module.initial_template, t, words=words)

    def after_point(self) -> None:
        """Execute supported table-template hooks at actual nested-loop boundaries.

        The innermost hardware loop always closes after a point. An outer hook
        runs only when every inner loop completed. The last entry wraps to entry
        zero before the next outer repetition. State belongs to this interpreter.
        """
        closed: set[str] = set()
        for name, count in reversed(self.prog.loop_dict.items()):
            closed.add(name)
            if self.point[name] != count - 1:
                break
        for key, module in self.table_readouts.items():
            if module.idx_reg in closed:
                next_index = (self.point[module.idx_reg] + 1) % len(module.freq_words)
                self.templates[key] = (
                    module.freq_words[next_index],
                    module.ro_freq_words[next_index],
                )

    def join(self, module: Join, t: float | QickParam) -> float | QickParam:
        """Interleave branches by static minimum time, matching Join emission.

        Each branch has its own local cursor. Swept pulse order stays fixed
        across hardware points. Cross-branch register dependencies are rejected.
        """
        from ..modules.util import merge_max_length

        queues = [list(branch) for branch in module.join_modules]
        times = [t for _ in queues]
        while any(queues):
            candidates = [i for i, queue in enumerate(queues) if queue]
            index = min(candidates, key=lambda i: _minimum(times[i]))
            times[index] = self.sequence([queues[index].pop(0)], times[index])
        return merge_max_length(*times)

    def check_join(self, module: Join) -> None:
        """Reject cross-branch register dependencies before choosing branch order."""
        accesses = [register_access(branch) for branch in module.join_modules]
        for i, (reads, writes) in enumerate(accesses):
            for other_reads, other_writes in accesses[i + 1 :]:
                conflict = writes & (other_reads | other_writes) | reads & other_writes
                if conflict:
                    raise ValueError(
                        f"Join cross-branch register dependency: {sorted(conflict)}"
                    )

    def align(self, extra: float = 0.0, *, gens: bool = True, ros: bool = True) -> None:
        """Advance reference through selected channel timestamps plus extra us."""
        ends = []
        if gens:
            ends.extend(self.gen_end.values())
        if ros:
            ends.extend(self.ro_end.values())
        if ends:
            self.ref += self.tick(max(0.0, max(ends) - self.ref) + extra)

    def pulse(
        self, module: Pulse, t: float | QickParam, *, duration: float | None = None
    ) -> float | QickParam:
        """Emit registered generator controls, preserving nonblocking scheduling."""
        cfg = module.cfg
        if cfg is None:
            return t
        prog = self.prog
        pulse_id = module.pulse_id
        if duration is None:
            duration = self.scalar(prog.get_pulse_param(pulse_id, "total_length"))
        event = PulseEvent(
            cfg.ch,
            self.ref + self.tick(t + cfg.pre_delay),
            duration,
            self.scalar(prog.get_pulse_param(pulse_id, "freq")),
            self.scalar(prog.get_pulse_param(pulse_id, "phase")),
            self.scalar(prog.get_pulse_param(pulse_id, "gain")),
            self._envelope(module, duration),
            source=module.name,
            iterations=tuple(self.iterations),
        )
        self.events.append(event)
        self.gen_end[cfg.ch] = event.start_us + duration
        return t + module.total_length(prog) if module.block_mode else t

    def _envelope(self, module: Pulse, duration: float) -> PulseEnvelope | None:
        cfg = module.cfg
        if cfg is None or isinstance(cfg.waveform, ConstWaveformCfg):
            return None
        ch = cfg.ch
        name = self.prog.pulses[module.pulse_id].params["envelope"]
        envelopes = self.prog.envelopes[ch]["envs"]
        if not isinstance(envelopes, dict):
            raise ValueError("compiled generator envelope table is missing")
        data = envelopes[name]["data"]
        values = (data[:, 0] + 1j * data[:, 1]) / self.prog.soccfg.get_maxv(ch)
        clock = self.prog.soccfg["gens"][ch]
        period = 1 / (clock["f_fabric"] * clock["samps_per_clk"])
        axis = np.arange(len(values)) * period
        if isinstance(cfg.waveform, FlatTopWaveformCfg):
            # QICK splits even ramp clocks and plays its plateau at half gain.
            half = len(values) // (2 * clock["samps_per_clk"]) * clock["samps_per_clk"]
            ramp_us = half * period
            values = np.concatenate((values[:half], [0.5, 0.5], values[-half:]))
            axis = np.concatenate(
                (
                    axis[:half],
                    [ramp_us, duration - ramp_us - period],
                    axis[:half] + duration - ramp_us,
                )
            )
        return PulseEnvelope(
            tuple(float(v) for v in axis), tuple(complex(v) for v in values)
        )

    def frequencies(
        self, module: PulseReadout, words: tuple[int, int] | None
    ) -> tuple[float, float]:
        """Decode final frequency words at their explicit readout consumer.

        Return semantic generator and demod MHz, including mixer and ADC alias
        conventions. Paired words must encode the same semantic frequency.
        """
        cfg = module.cfg
        gen = self.scalar(self.prog.get_pulse_param(module.pulse.pulse_id, "freq"))
        ro = self.scalar(self.prog.get_pulse_param(module.adc_name, "freq"))
        gen_word = (
            self.register(module.freq_val, word=True) if module.freq_val else None
        )
        ro_word = (
            self.register(module.ro_freq_val, word=True) if module.ro_freq_val else None
        )
        if words is not None:
            gen_word, ro_word = words
        if gen_word is not None and ro_word is not None:
            gen = readout_freq_from_words(
                self.prog.soccfg,
                gen_word,
                ro_word,
                gen_ch=cfg.gen_ch,
                ro_ch=cfg.ro_ch,
                mixer_freq=cfg.mixer_freq,
                nqz=cfg.nqz,
            )
            return gen, gen
        if gen_word is not None:
            gen = readout_freq_from_word(
                self.prog.soccfg,
                gen_word,
                kind="gen",
                gen_ch=cfg.gen_ch,
                ro_ch=cfg.ro_ch,
                mixer_freq=cfg.mixer_freq,
                nqz=cfg.nqz,
            )
        if ro_word is not None:
            ro = readout_freq_from_word(
                self.prog.soccfg,
                ro_word,
                kind="ro",
                reference_freq_mhz=gen,
                gen_ch=cfg.gen_ch,
                ro_ch=cfg.ro_ch,
                mixer_freq=cfg.mixer_freq,
                nqz=cfg.nqz,
            )
        return gen, ro

    def readout(
        self,
        module: PulseReadout,
        t: float | QickParam,
        *,
        words: tuple[int, int] | None = None,
    ) -> float | QickParam:
        """Emit independent ADC trigger/window and generator playback."""
        cfg = module.cfg
        self.routes[cfg.ro_ch] = cfg.gen_ch
        trigger = self.ref + self.tick(t + cfg.trig_offset)
        gen_freq, ro_freq = self.frequencies(module, words)
        self.reads[cfg.ro_ch].append((trigger, ro_freq, 0.0))
        self.ro_end[cfg.ro_ch] = trigger + self.prog.cycles2us(
            self.prog.ro_chs[cfg.ro_ch]["length"], ro_ch=cfg.ro_ch
        )
        start = len(self.events)
        self.pulse(module.pulse, t)
        event = self.events[start]
        gain = event.gain
        if module.gain_val is not None:
            word = self.register(module.gain_val, word=True)
            signed = word if word < 2**31 else word - 2**32
            gain = signed / self.prog.soccfg.get_maxv(cfg.gen_ch)
        self.events[start] = replace(
            event, source=module.name, frequency_mhz=gen_freq, gain=gain
        )
        return t + module.total_length(self.prog)
