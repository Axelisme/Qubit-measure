"""SimEngine-backed RO optimize gates for the first-stage readout model."""

from __future__ import annotations

import os

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from zcu_tools.experiment.v2.twotone.ro_optimize.freq_gain import (
    FreqGainCfg,
    FreqGainExp,
    FreqGainModuleCfg,
    FreqGainSweepCfg,
)
from zcu_tools.experiment.v2.twotone.ro_optimize.length import (
    LengthCfg,
    LengthExp,
    LengthModuleCfg,
    LengthSweepCfg,
)
from zcu_tools.experiment.v2.twotone.ro_optimize.power import (
    PowerCfg,
    PowerExp,
    PowerModuleCfg,
    PowerSweepCfg,
)
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.program.v2.sim.readout import resonator_freqs, s21
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

_OPERATING_FLUX = 1.0

_SIM = SimParams(
    EJ=8.5,
    EC=1.0,
    EL=0.5,
    flux_period=1.0,
    flux_half=0.0,
    flux_bias=0.2,
    T1=50.0,
    T2=30.0,
    T2_star=30.0,
    bare_rf=7.2,
    g=0.08,
    Ql=5000.0,
    Qi=50000.0,
    snr=10.0,
    pi_gain_len=0.4,
    seed=12345,
    poll_latency=0.0,
)


def _f_qubit_mhz() -> float:
    predictor = FluxoniumPredictor(
        params=(_SIM.EJ, _SIM.EC, _SIM.EL),
        flux_half=_SIM.flux_half,
        flux_period=_SIM.flux_period,
        flux_bias=_SIM.flux_bias,
    )
    return float(predictor.predict_freq(predictor.flux_to_value(_OPERATING_FLUX)))


def _rf_g_mhz() -> float:
    rf_g, _rf_e = resonator_freqs(_SIM, _OPERATING_FLUX)
    return rf_g * 1e3


def _probe_pulse() -> PulseCfg:
    return PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=_f_qubit_mhz(),
        phase=0.0,
        waveform=ConstWaveformCfg(length=_SIM.pi_gain_len),
    )


def _readout(*, pulse_length: float = 1.2, gain: float = 0.01) -> PulseReadoutCfg:
    ro = _rf_g_mhz()
    return PulseReadoutCfg(
        pulse_cfg=PulseCfg(
            ch=0,
            nqz=1,
            gain=gain,
            freq=ro,
            phase=0.0,
            waveform=ConstWaveformCfg(length=pulse_length),
        ),
        ro_cfg=DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro, trig_offset=0.0),
    )


def _deterministic_readout_contrast(freqs_mhz: np.ndarray) -> np.ndarray:
    rf_g, rf_e = resonator_freqs(_SIM, _OPERATING_FLUX)
    freqs_ghz = np.asarray(freqs_mhz, dtype=np.float64) / 1e3
    return np.abs(s21(_SIM, freqs_ghz, rf_e) - s21(_SIM, freqs_ghz, rf_g))


def test_power_exp_linear_gain_prefers_upper_bound() -> None:
    """With no saturation/backaction, linear readout gain prefers the high edge."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    cfg = PowerCfg(
        reps=1600,
        rounds=1,
        modules=PowerModuleCfg(
            reset=None,
            qub_pulse=_probe_pulse(),
            readout=_readout(),
        ),
        sweep=PowerSweepCfg(
            gain=SweepCfg(start=0.003, stop=0.018, expts=4, step=0.005)
        ),
    )

    exp = PowerExp()
    result = exp.run(soc, soccfg, cfg)
    best_gain, fig = exp.analyze(result, smooth=0.01, smooth_method="gaussian")
    plt.close(fig)

    assert result.signals[-1] > result.signals[0]
    assert best_gain == pytest.approx(float(result.gains[-1]))


def test_length_exp_prefers_longer_ro_length_when_pulse_covers_window() -> None:
    """Full-window readout integration improves SNR with longer ro_length."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    cfg = LengthCfg(
        reps=1600,
        rounds=1,
        modules=LengthModuleCfg(
            reset=None,
            qub_pulse=_probe_pulse(),
            readout=_readout(pulse_length=1.4),
        ),
        sweep=LengthSweepCfg(length=SweepCfg(start=0.3, stop=1.2, expts=4, step=0.3)),
    )

    exp = LengthExp()
    result = exp.run(soc, soccfg, cfg)
    best_length, fig = exp.analyze(result, smooth=0.01, smooth_method="gaussian")
    plt.close(fig)

    assert result.signals[-1] > result.signals[0]
    assert best_length == pytest.approx(float(result.lengths[-1]))


def test_freq_gain_exp_tracks_frequency_ridge_and_high_gain_edge() -> None:
    """Real FreqGainExp should follow deterministic S21 contrast and linear gain."""

    rf_g, rf_e = resonator_freqs(_SIM, _OPERATING_FLUX)
    high_rf_mhz = max(rf_g, rf_e) * 1e3
    freq_step = 2.0

    soc, soccfg = make_mock_soc(sim=_SIM)
    cfg = FreqGainCfg(
        reps=1600,
        rounds=1,
        modules=FreqGainModuleCfg(
            reset=None,
            qub_pulse=_probe_pulse(),
            readout=_readout(),
        ),
        sweep=FreqGainSweepCfg(
            freq=SweepCfg(
                start=high_rf_mhz - 8.0,
                stop=high_rf_mhz + 4.0,
                expts=7,
                step=freq_step,
            ),
            gain=SweepCfg(start=0.003, stop=0.018, expts=4, step=0.005),
        ),
    )

    exp = FreqGainExp()
    result = exp.run(soc, soccfg, cfg)
    best_freq, best_gain, fig = exp.analyze(
        result, smooth=0.01, smooth_method="gaussian"
    )
    plt.close(fig)

    contrast = _deterministic_readout_contrast(result.freqs)
    expected_freq = float(result.freqs[int(np.argmax(contrast))])
    actual_step = float(abs(result.freqs[1] - result.freqs[0]))

    assert abs(best_freq - expected_freq) <= 2.0 * actual_step
    assert best_gain == pytest.approx(float(result.gains[-1]))
