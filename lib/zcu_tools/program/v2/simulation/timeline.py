"""Module scheduling and quantized numeric controls, without physical models."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from qick.asm_v2 import QickParam

from zcu_tools.core.simulation import (
    PointTimeline,
    PulseEnvelope,
    PulseEvent,
    ReadoutWindow,
)

from ..modular import ModularProgramV2
from ..modules.base import Module
from ..modules.delay import Delay, DelayAuto, Join, SoftDelay
from ..modules.pulse import Pulse
from ..modules.readout import PulseReadout
from ..modules.util import round_timestamp
from ..modules.waveform import ConstWaveformCfg, FlatTopWaveformCfg


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
        return float(self.prog.cycles2us(self.prog.us2cycles(self.scalar(value))))

    def run_point(self, point: dict[str, int]) -> PointTimeline:
        """Evaluate one loop coordinate and return owned immutable controls."""
        self.point = point.copy()
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
            self.ref + self.tick(t), *self.gen_end.values(), *self.ro_end.values()
        )
        return PointTimeline(tuple(point.items()), tuple(self.events), windows, end)

    def sequence(self, modules, t: float) -> float:
        """Run a semantic sequence at local us t and return its local end time."""
        for module in modules:
            try:
                t = self.run(module, t)
            except (ValueError, KeyError, IndexError) as exc:
                raise ValueError(
                    f"{type(module).__name__}({module.name!r}) at {self.point}: {exc}"
                ) from exc
        return t

    def run(self, module: Module, t: float) -> float:
        """Dispatch exact supported Module types; reject unknown subclasses."""
        kind = type(module)
        if isinstance(module, Pulse) and kind is Pulse:
            return self.pulse(module, t)
        if isinstance(module, PulseReadout) and kind is PulseReadout:
            return self.readout(module, t)
        if isinstance(module, SoftDelay) and kind is SoftDelay:
            return self.scalar(round_timestamp(self.prog, t + module.delay))
        if isinstance(module, Delay) and kind is Delay:
            self.ref += self.tick(round_timestamp(self.prog, t + module.delay))
            return 0.0
        if isinstance(module, DelayAuto) and kind is DelayAuto:
            if isinstance(module.t, str):
                raise ValueError("runtime delay simulation unsupported")
            self.align(self.scalar(module.t), gens=module.gens, ros=module.ros)
            return 0.0
        if isinstance(module, Join) and kind is Join:
            return max(self.sequence(branch, t) for branch in module.join_modules)
        raise ValueError(f"{kind.__name__} simulation unsupported")

    def align(self, extra: float = 0.0, *, gens: bool = True, ros: bool = True) -> None:
        """Advance reference through selected channel timestamps plus extra us."""
        ends = []
        if gens:
            ends.extend(self.gen_end.values())
        if ros:
            ends.extend(self.ro_end.values())
        if ends:
            self.ref += self.tick(max(ends) - self.ref + extra)

    def pulse(self, module: Pulse, t: float) -> float:
        """Emit registered generator controls, preserving nonblocking scheduling."""
        cfg = module.cfg
        if cfg is None:
            return t
        prog = self.prog
        pulse_id = module.pulse_id
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
            iterations=(),
        )
        self.events.append(event)
        self.gen_end[cfg.ch] = event.start_us + duration
        return t + self.scalar(module.total_length(prog)) if module.block_mode else t

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

    def readout(self, module: PulseReadout, t: float) -> float:
        """Emit independent ADC trigger/window and generator playback."""
        if any((module.gain_val, module.freq_val, module.ro_freq_val)):
            raise ValueError("runtime word consumer simulation unsupported")
        cfg = module.cfg
        self.routes[cfg.ro_ch] = cfg.gen_ch
        trigger = self.ref + self.tick(t + cfg.trig_offset)
        frequency = self.scalar(self.prog.get_pulse_param(module.adc_name, "freq"))
        self.reads[cfg.ro_ch].append((trigger, frequency, 0.0))
        self.ro_end[cfg.ro_ch] = trigger + self.prog.cycles2us(
            self.prog.ro_chs[cfg.ro_ch]["length"], ro_ch=cfg.ro_ch
        )
        start = len(self.events)
        self.pulse(module.pulse, t)
        self.events[start] = replace(self.events[start], source=module.name)
        return t + self.scalar(module.total_length(self.prog))
