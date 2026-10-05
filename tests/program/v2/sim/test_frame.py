"""Drives at several carriers keep phase-coherent phases within one shot.

The oracle slices each drive into short segments in the qubit frame, where a
drive at ``f`` turns its axis at ``2*pi*(f - f_qubit)`` from t = 0 at the start
of the shot.
"""

from __future__ import annotations

import math

import pytest
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.delay import Delay
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sim import SimParams, bloch
from zcu_tools.program.v2.sim.frame import align_drive_phases
from zcu_tools.program.v2.sim.lowering import lower_point

# Effectively infinite T1/T2: rotations stay unitary.
_SIM = SimParams(
    EJ=8.5,
    EC=1.0,
    EL=0.5,
    flux_period=0.002,
    flux_half=0.001,
    T1=1.0e9,
    T2=1.0e9,
    T2_star=1.0e9,
    bare_rf=7.2,
    g=0.08,
    Ql=5000.0,
    Qi=50000.0,
    snr=10.0,
    pi_gain_len=0.4,
)
_F_QUBIT_MHZ = 4000.0
_SLICES = 2000


def _pulse(name: str, *, gain: float, freq: float, phase: float = 0.0) -> Pulse:
    cfg = PulseCfg(
        waveform=ConstWaveformCfg(length=0.4),
        ch=0,
        nqz=1,
        freq=freq,
        phase=phase,
        gain=gain,
    )
    return Pulse(name, cfg)


def _readout() -> Module:
    return DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=7200.0).build("ro")


def _excited(segments: list[bloch.Segment]) -> float:
    return bloch.excited_population(bloch.evolve(bloch.ground_state(0.0), segments))


def _lowered_excited(modules: list[Module]) -> float:
    lowered = lower_point(
        modules, None, _SIM, _F_QUBIT_MHZ / 1e3, {}, lambda cycles: cycles * 0.01
    )
    return _excited(lowered.segments)


def _sliced_excited(modules: list[Module]) -> float:
    segments: list[bloch.Segment] = []
    clock = 0.0
    for module in modules:
        if isinstance(module, Delay):
            delay = float(module.delay)
            segments.append(bloch.Segment(0.0, 0.0, 0.0, delay, None, None, 0.0))
            clock += delay
        elif isinstance(module, Pulse):
            cfg = module.cfg
            assert cfg is not None
            length = float(cfg.waveform.length)
            omega = math.pi / _SIM.pi_gain_len * float(cfg.gain)
            offset = 2.0 * math.pi * (float(cfg.freq) - _F_QUBIT_MHZ)
            dt = length / _SLICES
            for k in range(_SLICES):
                phase = math.radians(float(cfg.phase)) + offset * (
                    clock + (k + 0.5) * dt
                )
                segments.append(bloch.Segment(omega, 0.0, phase, dt, None, None, 0.0))
            clock += length
    return _excited(segments)


@pytest.mark.parametrize("detuning", [0.3, -0.7, 2.5])
def test_two_carrier_sequence_matches_sliced_qubit_frame(detuning: float) -> None:
    f_drive = _F_QUBIT_MHZ + 0.2
    modules = [
        _pulse("x90", gain=0.5, freq=f_drive),
        Delay("wait1", 0.3),
        _pulse("probe", gain=1.0, freq=f_drive + detuning, phase=30.0),
        Delay("wait2", 0.2),
        _pulse("x90b", gain=0.5, freq=f_drive),
        _readout(),
    ]

    assert _lowered_excited(modules) == pytest.approx(
        _sliced_excited(modules), abs=1e-4
    )


def test_single_carrier_timeline_is_unchanged() -> None:
    drive = bloch.Segment(3.0, -1.0, 0.5, 0.2, None, None, 0.0)
    idle = bloch.Segment(0.0, -1.0, 0.0, 0.7, None, None, 0.0)

    assert align_drive_phases([drive, idle, drive], -1.0) == [drive, idle, drive]
