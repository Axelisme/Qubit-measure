"""Concrete callbacks at the native Run/Schedule/simulator seam."""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pytest
from qick import QickConfig
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.experiment.workflows import MissingCapability, Run
from zcu_tools.program.v2.mocksoc import MockQickSoc, make_mock_soc
from zcu_tools.program.v2.modules import DirectReadoutCfg, PulseCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg, GaussWaveformCfg
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.program.v2.sim.readout import resonator_freqs
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.simulate.fluxonium import FluxoniumPredictor

from zcu_lab.v2.twotone.freq.core import (
    FreqCfg,
    FreqModuleCfg,
    FreqResult,
    FreqSweepCfg,
)
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiCfg,
    LenRabiModuleCfg,
    LenRabiResult,
    LenRabiSweepCfg,
)
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import (
    FreqGainCfg,
    FreqGainModuleCfg,
    FreqGainResult,
    FreqGainSweepCfg,
)
from zcu_lab.v2.twotone.time_domain.t1.core import (
    T1Cfg,
    T1ModuleCfg,
    T1Result,
    T1SweepCfg,
)
from zcu_lab.workflows.acquire import (
    run_len_rabi,
    run_qubit_freq,
    run_ro_landscape,
    run_t1,
)


@dataclass
class Inputs:
    """Isolated board/config and the four validated native callback inputs.

    soc/soccfg form one physical simulator. freq/rabi/readout/t1 contain native
    cfg for their respective callbacks, calibrated to that simulator's qubit
    and ground-state resonator at reduced flux 1.0. They have no device setup.
    """

    soc: MockQickSoc
    soccfg: QickConfig
    freq: FreqCfg
    rabi: LenRabiCfg
    readout: FreqGainCfg
    t1: T1Cfg


@pytest.fixture
def inputs() -> Inputs:
    sim = SimParams(
        EJ=8.5,
        EC=1.0,
        EL=0.5,
        flux_period=1.0,
        flux_half=0.0,
        flux_bias=0.2,
        T1=20.0,
        T2=10.0,
        T2_star=10.0,
        bare_rf=7.2,
        g=0.08,
        Ql=5000.0,
        Qi=50000.0,
        snr=200.0,
        pi_gain_len=0.4,
        seed=12345,
        poll_latency=0.0,
    )
    soc, soccfg = make_mock_soc(sim=sim)
    predictor = FluxoniumPredictor(
        params=(sim.EJ, sim.EC, sim.EL),
        flux_half=sim.flux_half,
        flux_period=sim.flux_period,
        flux_bias=sim.flux_bias,
    )
    qubit_freq = float(predictor.predict_freq(predictor.flux_to_value(1.0)))
    ro_freq = resonator_freqs(sim, 1.0)[0] * 1000.0
    pi_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=qubit_freq,
        waveform=ConstWaveformCfg(length=sim.pi_gain_len),
    )
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_freq)
    weak_pulse = pi_pulse.model_copy(deep=True)
    weak_pulse.gain = 0.05
    weak_pulse.waveform = ConstWaveformCfg(length=2.0)
    pulse_readout = PulseReadoutCfg(
        pulse_cfg=PulseCfg(
            ch=1, nqz=2, gain=0.1, freq=ro_freq, waveform=ConstWaveformCfg(length=1.0)
        ),
        ro_cfg=DirectReadoutCfg(ro_ch=0, gen_ch=1, ro_length=1.0, ro_freq=ro_freq),
    )
    return Inputs(
        soc,
        soccfg,
        FreqCfg(
            reps=100,
            rounds=2,
            relax_delay=200.0,
            modules=FreqModuleCfg(qub_pulse=weak_pulse, readout=readout),
            sweep=FreqSweepCfg(
                freq=SweepCfg(
                    start=qubit_freq - 125.0,
                    stop=qubit_freq + 125.0,
                    expts=31,
                    step=250.0 / 30,
                )
            ),
        ),
        LenRabiCfg(
            reps=100,
            rounds=2,
            relax_delay=200.0,
            modules=LenRabiModuleCfg(qub_pulse=pi_pulse, readout=readout),
            sweep=LenRabiSweepCfg(
                length=SweepCfg(start=0.05, stop=1.2, expts=31, step=1.15 / 30)
            ),
        ),
        FreqGainCfg(
            reps=100,
            rounds=2,
            relax_delay=200.0,
            modules=FreqGainModuleCfg(qub_pulse=pi_pulse, readout=pulse_readout),
            sweep=FreqGainSweepCfg(
                freq=SweepCfg(
                    start=ro_freq - 2.0, stop=ro_freq + 2.0, expts=9, step=0.5
                ),
                gain=SweepCfg(start=0.05, stop=0.15, expts=3, step=0.05),
            ),
        ),
        T1Cfg(
            reps=100,
            rounds=2,
            relax_delay=200.0,
            modules=T1ModuleCfg(pi_pulse=pi_pulse, readout=readout),
            sweep=T1SweepCfg(length=SweepCfg(start=0.0, stop=80.0, expts=21, step=4.0)),
        ),
    )


def test_frequency_trace_uses_native_coordinates_and_isolated_cfg(
    inputs: Inputs,
) -> None:
    original = inputs.freq.model_copy(deep=True)
    run = Run(inputs.freq, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg)
    before = datetime.now(UTC)
    result = run_qubit_freq(run)
    assert (
        before
        <= result.evidence.started_at
        <= result.evidence.finished_at
        <= datetime.now(UTC)
    )
    assert result.evidence.completion == "complete"
    expected = sweep2array(
        inputs.freq.sweep.freq,
        "freq",
        {"soccfg": inputs.soccfg, "gen_ch": inputs.freq.modules.qub_pulse.ch},
    )
    assert isinstance(result, FreqResult)
    np.testing.assert_array_equal(result.freqs, expected)
    assert result.signals.shape == result.freqs.shape
    assert np.isfinite(result.signals).all()
    assert np.ptp(np.abs(result.signals)) > 0.1
    assert run.outcome.status == "completed"
    assert inputs.freq == original
    assert run.cfg == original


def test_rabi_trace_uses_native_time_coordinates(inputs: Inputs) -> None:
    run = Run(inputs.rabi, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg)
    before = datetime.now(UTC)
    result = run_len_rabi(run)
    assert (
        before
        <= result.evidence.started_at
        <= result.evidence.finished_at
        <= datetime.now(UTC)
    )
    assert result.evidence.completion == "complete"
    expected = sweep2array(
        inputs.rabi.sweep.length,
        "time",
        {"soccfg": inputs.soccfg, "gen_ch": inputs.rabi.modules.qub_pulse.ch},
    )
    assert isinstance(result, LenRabiResult)
    np.testing.assert_array_equal(result.lengths, expected)
    assert result.signals.shape == result.lengths.shape
    assert np.isfinite(result.signals).all()
    assert np.ptp(np.abs(result.signals)) > 0.1
    assert run.outcome.status == "completed"


def test_readout_landscape_keeps_native_axis_order_and_real_snr(inputs: Inputs) -> None:
    run = Run(inputs.readout, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg)
    before = datetime.now(UTC)
    result = run_ro_landscape(run)
    assert (
        before
        <= result.evidence.started_at
        <= result.evidence.finished_at
        <= datetime.now(UTC)
    )
    assert result.evidence.completion == "complete"
    env = {
        "soccfg": inputs.soccfg,
        "gen_ch": inputs.readout.modules.readout.pulse_cfg.ch,
    }
    assert isinstance(result, FreqGainResult)
    np.testing.assert_array_equal(
        result.freqs, sweep2array(inputs.readout.sweep.freq, "freq", env)
    )
    np.testing.assert_array_equal(
        result.gains, sweep2array(inputs.readout.sweep.gain, "gain", env)
    )
    assert result.signals.shape == (result.freqs.size, result.gains.size)
    assert result.signals.dtype == np.float64
    assert np.isfinite(result.signals).all()
    assert result.signals.max() > 1.0
    assert run.outcome.status == "completed"


def test_t1_trace_preserves_native_units_and_acquisition_history(
    inputs: Inputs,
) -> None:
    before = datetime.now(UTC)
    run = Run(inputs.t1, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg)
    result = run_t1(run)
    after = datetime.now(UTC)
    assert isinstance(result, T1Result)
    np.testing.assert_array_equal(
        result.times,
        sweep2array(inputs.t1.sweep.length, "time", {"soccfg": inputs.soccfg}),
    )
    assert result.signals.shape == result.times.shape
    assert np.isfinite(result.signals).all()
    assert (
        abs(np.abs(result.signals[:3]).mean() - np.abs(result.signals[-3:]).mean())
        > 0.1
    )
    assert run.outcome.status == "completed"
    assert before <= result.evidence.started_at <= result.evidence.finished_at <= after
    assert result.evidence.completion == "complete"


def test_preexisting_stop_keeps_native_partial_result_and_stopped_history(
    inputs: Inputs,
) -> None:
    stop = StopSignal()
    stop.set()
    run = Run(inputs.t1, stop, soc=inputs.soc, soccfg=inputs.soccfg)
    result = run_t1(run)
    assert run.outcome.status == "stopped"
    assert np.isnan(result.signals).all()
    assert result.evidence.completion == "stopped"


def test_rabi_rejects_waveforms_outside_flat_sweep_contract(inputs: Inputs) -> None:
    cfg = inputs.rabi.model_copy(deep=True)
    cfg.modules.qub_pulse.waveform = GaussWaveformCfg(length=0.4, sigma=0.1)
    with pytest.raises(ValueError, match="const|flat_top"):
        run_len_rabi(Run(cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg))


def test_t1_requires_native_soc_capability(inputs: Inputs) -> None:
    with pytest.raises(MissingCapability, match="soc"):
        run_t1(Run(inputs.t1, StopSignal()))


@pytest.mark.parametrize("uniform", [True, False])
def test_t1_rejects_modes_outside_uniform_sweep_contract(
    inputs: Inputs, uniform: bool
) -> None:
    cfg = inputs.t1.model_copy(deep=True)
    cfg.uniform = uniform
    if uniform:
        cfg.sweep.length = [0.0, 1.0]
    with pytest.raises(ValueError, match="uniform"):
        run_t1(Run(cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg))
