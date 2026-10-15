"""Minimal static PulseReadout frontend; full Module interpretation extends here."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from zcu_tools.core.simulation import (
    AcquireRequest,
    AcquisitionMode,
    PulseEvent,
    ReadoutWindow,
)

from ..modules.readout import PulseReadout
from ..modules.waveform import ConstWaveformCfg

if TYPE_CHECKING:
    from zcu_tools.program.acquisition import ImproveAcquireMixin

    from .binding import SimulationBinding


def build_request(
    prog: ImproveAcquireMixin,
    mode: AcquisitionMode,
    binding: SimulationBinding,
    sequence: int,
) -> AcquireRequest:
    from ..modular import ModularProgramV2

    if not isinstance(prog, ModularProgramV2):
        raise ValueError("simulation requires a ModularProgramV2 frontend")
    if prog.loop_dims is None or prog.avg_level is None:
        prog.compile()
    if prog.loop_dims is None or prog.avg_level is None:
        raise ValueError("compiled acquisition layout is missing")
    events = []
    triggers: dict[int, list[float]] = {ch: [] for ch in prog.ro_chs}
    routes: dict[int, int] = {}
    cursor = 0.0
    for module in prog.modules:
        if type(module) is not PulseReadout or any(
            (module.gain_val, module.freq_val, module.ro_freq_val)
        ):
            raise ValueError(f"unsupported simulation Module: {type(module).__name__}")
        cfg = module.cfg
        routes[cfg.ro_ch] = cfg.gen_ch
        if not isinstance(cfg.waveform, ConstWaveformCfg):
            raise ValueError("simulation currently supports constant readout waveforms")
        values = (
            cfg.freq,
            cfg.gain,
            cfg.phase,
            cfg.pre_delay,
            cfg.post_delay,
            cfg.trig_offset,
            cfg.waveform.length,
        )
        if not all(isinstance(v, (int, float)) for v in values):
            raise ValueError("simulation currently requires scalar readout controls")
        # QICK quantizers own the board grid, including matched generator/ADC DDS.
        start = float(prog.cycles2us(prog.us2cycles(cursor + float(cfg.pre_delay))))
        duration = float(
            prog.cycles2us(
                prog.us2cycles(cfg.waveform.length, gen_ch=cfg.gen_ch),
                gen_ch=cfg.gen_ch,
            )
        )
        frequency = float(
            prog.reg2freq(
                prog.freq2reg(cfg.freq, gen_ch=cfg.gen_ch, ro_ch=cfg.ro_ch),
                gen_ch=cfg.gen_ch,
            )
        )
        events.append(
            PulseEvent(
                cfg.gen_ch,
                start,
                duration,
                frequency,
                float(
                    prog.reg2deg(
                        prog.deg2reg(cfg.phase, gen_ch=cfg.gen_ch), gen_ch=cfg.gen_ch
                    )
                ),
                int(float(cfg.gain) * prog.soccfg.get_maxv(cfg.gen_ch))
                / prog.soccfg.get_maxv(cfg.gen_ch),
            )
        )
        triggers[cfg.ro_ch].append(
            float(prog.cycles2us(prog.us2cycles(cursor + float(cfg.trig_offset))))
        )
        cursor = float(module.total_length(prog)) + cursor
    dims = tuple(int(n) for n in prog.loop_dims)
    windows = tuple(
        ReadoutWindow(
            ch,
            routes[ch],
            int(ro["length"]),
            float(prog.soccfg["readouts"][ch]["f_output"]),
            tuple(triggers[ch]),
        )
        for ch, ro in prog.ro_chs.items()
    )
    if not windows or any(
        window.reads != prog.ro_chs[window.channel]["trigs"] for window in windows
    ):
        raise ValueError("simulation readout layout does not match compiled triggers")
    return AcquireRequest(
        mode,
        dims,
        int(prog.avg_level),
        windows,
        (tuple(events),) * math.prod(dims),
        binding.root_seed,
        sequence,
        binding.rng_version,
    )
