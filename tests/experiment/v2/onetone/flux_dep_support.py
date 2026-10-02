"""Typed FluxDep fixtures shared by acquisition and Notebook contract tests."""

from __future__ import annotations

import numpy as np
from zcu_tools.experiment.v2.onetone.flux_dep import (
    FluxDepCfg,
    FluxDepModuleCfg,
    FluxDepResult,
    FluxDepSweepCfg,
)
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sweep import SweepCfg


def make_result() -> FluxDepResult:
    values = np.linspace(-0.5, 0.5, 9)
    freqs = np.linspace(4.8, 5.4, 7)
    signals = np.asarray(
        np.sin(values[:, None] * 7 + freqs[None, :] * 9)
        + 1j * np.cos(values[:, None] * 3 - freqs[None, :] * 7),
        dtype=np.complex128,
    )
    return FluxDepResult(values, freqs, signals)


def make_cfg() -> FluxDepCfg:
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.2,
        freq=7000.0,
        phase=0.0,
        waveform=ConstWaveformCfg(length=1.0),
    )
    return FluxDepCfg(
        reps=1,
        rounds=1,
        dev={},
        modules=FluxDepModuleCfg(
            readout=PulseReadoutCfg(
                pulse_cfg=pulse,
                ro_cfg=DirectReadoutCfg(
                    ro_ch=0, gen_ch=0, ro_length=1.0, ro_freq=7000.0, trig_offset=0.0
                ),
            ),
        ),
        sweep=FluxDepSweepCfg(
            flux=SweepCfg(start=-0.5, stop=0.5, step=0.125, expts=9),
            freq=SweepCfg(start=4.8, stop=5.4, step=0.1, expts=7),
        ),
    )
