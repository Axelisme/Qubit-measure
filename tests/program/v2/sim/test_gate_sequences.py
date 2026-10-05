"""Gate-sequence experiments recover injected pulse errors on the simulator.

AllXY selects its gates with ``ComputedPulse`` and the zig-zag scan repeats a
pulse at a swept gain or frequency with a register-driven ``Repeat``; all run
end to end through the real experiment classes on a sim mock soc.  With
``pi_gain_len = 0.4`` a 0.5 µs const pulse needs gain 0.8 for a pi rotation and
0.4 for a pi/2 rotation.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.twotone.allxy import (
    AllXY_Exp,
    AllXYAnalyzeOptions,
    AllXYCfg,
    AllXYModuleCfg,
)
from zcu_tools.experiment.v2.twotone.zigzag_sweep import (
    ZigZagScanAnalyzeOptions,
    ZigZagScanCfg,
    ZigZagScanExp,
    ZigZagScanModuleCfg,
    ZigZagScanSweepCfg,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from ._engine_support import (
    RESET_RELAX_DELAY,
    SIM,
    ground_resonator_frequency_mhz,
    qubit_frequency_mhz,
)

_LENGTH = 0.5
_PI_GAIN = SIM.pi_gain_len / _LENGTH


def _pulse(gain: float) -> PulseCfg:
    return PulseCfg(
        ch=0,
        nqz=1,
        gain=gain,
        freq=qubit_frequency_mhz(),
        phase=0.0,
        waveform=ConstWaveformCfg(length=_LENGTH),
    )


def _readout() -> DirectReadoutCfg:
    return DirectReadoutCfg(
        ro_ch=0, ro_length=1.0, ro_freq=ground_resonator_frequency_mhz()
    )


@contextmanager
def _context() -> Generator[RunContext]:
    soc, soccfg = make_mock_soc(sim=SIM)
    plots = Plots(NonPresentingHost())
    try:
        yield RunContext(soc, soccfg, plots, {}, StopSignal())
    finally:
        plots.finish(present=False)


def _allxy_amplitude_error(gain_scale: float) -> float:
    cfg = AllXYCfg(
        reps=100,
        rounds=1,
        modules=AllXYModuleCfg(
            X90_pulse=_pulse(_PI_GAIN / 2 * gain_scale),
            X180_pulse=_pulse(_PI_GAIN * gain_scale),
            readout=_readout(),
        ),
        relax_delay=RESET_RELAX_DELAY,
    )
    exp = AllXY_Exp()
    with _context() as context:
        result = exp.run(cfg, context=context)
        analysis = exp.analyze(
            RunRecord(cfg, result), AllXYAnalyzeOptions(), plots=context.plots
        )
    return analysis.amplitude_error


def test_allxy_reports_injected_gain_error() -> None:
    calibrated = _allxy_amplitude_error(1.0)
    over_driven = _allxy_amplitude_error(1.1)

    assert abs(calibrated) < 0.03
    assert over_driven - calibrated == pytest.approx(0.1, abs=0.02)


def _zigzag_scan_cfg(sweep: ZigZagScanSweepCfg) -> ZigZagScanCfg:
    return ZigZagScanCfg(
        reps=50,
        rounds=1,
        modules=ZigZagScanModuleCfg(
            X90_pulse=_pulse(_PI_GAIN / 2),
            X180_pulse=_pulse(_PI_GAIN),
            readout=_readout(),
        ),
        sweep=sweep,
        n_times=6,
        relax_delay=RESET_RELAX_DELAY,
    )


def test_zigzag_gain_scan_recovers_pi_gain() -> None:
    cfg = _zigzag_scan_cfg(
        ZigZagScanSweepCfg(gain=SweepCfg(start=0.7, stop=0.9, expts=21, step=0.01))
    )
    exp = ZigZagScanExp()
    with _context() as context:
        result = exp.run(cfg, context=context)
        analysis = exp.analyze(
            RunRecord(cfg, result), ZigZagScanAnalyzeOptions(), plots=context.plots
        )

    assert analysis.min_value == pytest.approx(_PI_GAIN, abs=0.011)


def test_zigzag_freq_scan_recovers_qubit_frequency() -> None:
    # The X90 pulse stays at q_f while the repeated pulse's frequency is swept.
    f_qubit = qubit_frequency_mhz()
    cfg = _zigzag_scan_cfg(
        ZigZagScanSweepCfg(
            freq=SweepCfg(start=f_qubit - 2.0, stop=f_qubit + 2.0, expts=41, step=0.1)
        )
    )
    exp = ZigZagScanExp()
    with _context() as context:
        result = exp.run(cfg, context=context)
        analysis = exp.analyze(
            RunRecord(cfg, result), ZigZagScanAnalyzeOptions(), plots=context.plots
        )

    assert analysis.min_value == pytest.approx(f_qubit, abs=0.11)
