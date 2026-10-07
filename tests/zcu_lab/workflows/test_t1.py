"""Native T1 cfg generation and initial scalar state at public seams."""

from datetime import UTC, datetime

import numpy as np
import pytest
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.experiment.workflows import InitEnv, Run
from zcu_tools.program.v2.modules import DirectReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.sweep import SweepCfg

from zcu_lab.workflows.acquire import run_t1
from zcu_lab.workflows.analysis import analyze_t1
from zcu_lab.workflows.calibration import Calibrated, CalibrationPlan
from zcu_lab.workflows.t1 import T1Plan, T1Tunables, init_t1, t1_cfg

from ._sim_inputs import Inputs


def make_plan(inputs: Inputs) -> T1Plan:
    """Build complete native inputs without replacing acquisition logic."""
    return T1Plan(
        flux_dev="flux",
        fluxes=(1.0, 1.1),
        points=21,
        calibration=CalibrationPlan(
            frequency=inputs.freq,
            rabi=inputs.rabi,
            readout=inputs.readout,
        ),
        t1=inputs.t1,
    )


def seed_from_inputs(inputs: Inputs, *, pi_length_us: float = 0.4) -> Calibrated:
    """Create accepted scalars matching this simulator's native drive/readout."""
    frequency = inputs.rabi.modules.qub_pulse.freq
    readout = inputs.readout.modules.readout
    assert isinstance(frequency, float)
    assert isinstance(readout.pulse_cfg.freq, float)
    return Calibrated(frequency, pi_length_us, readout.pulse_cfg.freq, 0.15)


def test_init_t1_detaches_remaining_setpoints_and_each_runs_history(
    inputs: Inputs,
) -> None:
    plan = make_plan(inputs)
    seed = seed_from_inputs(inputs)
    env = InitEnv(datetime.now(UTC), seed)
    first = init_t1(env, plan)
    first.remaining.pop(0)
    first.cal.f01_by_flux.append((1.0, seed.f01_mhz))
    first.cal.n_calibrated += 1
    first.t1_by_flux.append((1.0, 18.0))

    second = init_t1(env, plan)
    assert second.remaining == list(plan.fluxes)
    assert second.cal.seed == seed
    assert second.cal.last is None
    assert second.cal.f01_by_flux == []
    assert second.cal.n_calibrated == 0
    assert second.t1_by_flux == []
    assert first.remaining == [1.1]


@pytest.mark.parametrize("pulse_readout", [False, True])
def test_generated_cfg_acquires_t1_with_calibrated_drive_and_native_readout(
    inputs: Inputs, pulse_readout: bool
) -> None:
    inputs.rabi.modules.qub_pulse.set_param("gain", 0.5)
    # An unrelated T1 drive must not replace the drive used to calibrate pi.
    inputs.t1.modules.pi_pulse = inputs.t1.modules.pi_pulse.model_copy(deep=True)
    inputs.t1.modules.pi_pulse.set_param("gain", 0.1)
    inputs.t1.modules.pi_pulse.set_param("length", 0.1)
    if pulse_readout:
        inputs.t1.modules.readout = inputs.readout.modules.readout.model_copy(deep=True)
    plan = make_plan(inputs)
    original = plan.model_copy(deep=True)
    cal = seed_from_inputs(inputs, pi_length_us=0.8)
    tun = T1Tunables(reps=120, rounds=2, sweep_factor=4.0, relax_factor=7.0)
    cfg = t1_cfg(plan, tun, cal, 20.0)

    assert (cfg.reps, cfg.rounds, cfg.relax_delay) == (120, 2, 140.0)
    assert isinstance(cfg.sweep.length, SweepCfg)
    assert (cfg.sweep.length.start, cfg.sweep.length.stop) == (0.0, 80.0)
    assert cfg.sweep.length.expts == plan.points
    assert cfg.modules.pi_pulse.gain == 0.5
    assert cfg.modules.pi_pulse.freq == cal.f01_mhz
    assert cfg.modules.pi_pulse.waveform.length == cal.pi_length_us
    readout = cfg.modules.readout
    if pulse_readout:
        assert isinstance(readout, PulseReadoutCfg)
        assert readout.pulse_cfg.freq == cal.readout_freq_mhz
        assert readout.ro_cfg.ro_freq == cal.readout_freq_mhz
        assert readout.pulse_cfg.gain == cal.readout_gain
    else:
        assert isinstance(readout, DirectReadoutCfg)
        assert readout.ro_freq == cal.readout_freq_mhz

    run = Run(cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg)
    result = run_t1(run)
    assert run.outcome.status == "completed"
    np.testing.assert_array_equal(
        result.times, sweep2array(cfg.sweep.length, "time", {"soccfg": inputs.soccfg})
    )
    fit = analyze_t1(result)
    assert fit.lifetime_us == pytest.approx(20.0, rel=0.1)
    assert np.isfinite(fit.error_us)
    assert result.signals.shape == (plan.points,)
    assert plan == original

    cfg.modules.pi_pulse.set_param("freq", cal.f01_mhz + 10.0)
    cfg.modules.readout.set_param(
        "freq" if pulse_readout else "ro_freq", cal.readout_freq_mhz + 10.0
    )
    again = t1_cfg(plan, tun, cal, 20.0)
    assert again.modules.pi_pulse.freq == cal.f01_mhz
    assert plan == original


def test_new_generation_knobs_change_delay_and_averaging_without_mutating_prior_cfg(
    inputs: Inputs,
) -> None:
    plan = make_plan(inputs)
    cal = seed_from_inputs(inputs)
    first_tun = T1Tunables(reps=30, rounds=2, sweep_factor=4.0, relax_factor=7.0)
    first = t1_cfg(plan, first_tun, cal, 20.0)
    second = t1_cfg(
        plan,
        T1Tunables(reps=60, rounds=3, sweep_factor=6.0, relax_factor=8.0),
        cal,
        10.0,
    )
    assert isinstance(first.sweep.length, SweepCfg)
    assert isinstance(second.sweep.length, SweepCfg)
    assert (first.sweep.length.stop, first.relax_delay) == (80.0, 140.0)
    assert (second.sweep.length.stop, second.relax_delay) == (60.0, 80.0)
    assert (first.reps, first.rounds) == (30, 2)
    assert (second.reps, second.rounds) == (60, 3)
    assert first_tun.reps == 30


@pytest.mark.parametrize("lifetime_us", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_lifetime_guess_fails_before_native_cfg_generation(
    inputs: Inputs, lifetime_us: float
) -> None:
    plan = make_plan(inputs)
    original = plan.model_copy(deep=True)
    with pytest.raises(ValueError, match="positive and finite"):
        t1_cfg(plan, T1Tunables(), seed_from_inputs(inputs), lifetime_us)
    assert plan == original
