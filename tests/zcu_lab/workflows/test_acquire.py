"""Concrete callbacks at the native Run/Schedule/simulator seam."""

from datetime import UTC, datetime

import numpy as np
import pytest
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.experiment.workflows import MissingCapability, Run
from zcu_tools.program.v2.modules.waveform import GaussWaveformCfg
from zcu_tools.program.v2.sweep import SweepCfg

from zcu_lab.v2.twotone.freq.core import FreqResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiResult
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainResult
from zcu_lab.v2.twotone.time_domain.t1.core import T1Result
from zcu_lab.workflows.acquire import (
    run_len_rabi,
    run_qubit_freq,
    run_ro_landscape,
    run_t1,
)

from ._sim_inputs import Inputs


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


def test_invalid_rabi_pulse_keeps_native_schedule_failure(inputs: Inputs) -> None:
    cfg = inputs.rabi.model_copy(deep=True)
    # Coordinates are valid, but the first pulse is shorter than the generator minimum.
    cfg.sweep.length = SweepCfg(start=0.0, stop=1.2, expts=31, step=1.2 / 30)
    run = Run(cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg)
    result = run_len_rabi(run)
    outcome = run.outcome
    assert outcome.status == "failed"
    assert isinstance(outcome.exception, RuntimeError)
    assert "length" in str(outcome.exception)
    assert run.cancel_signal.error is not None
    assert run.cancel_signal.error.exception is outcome.exception
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
