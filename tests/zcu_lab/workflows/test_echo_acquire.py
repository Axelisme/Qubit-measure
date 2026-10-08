"""Zero-detune echo crosses the real Run/Schedule/simulator seam."""

from datetime import UTC, datetime

import numpy as np
import pytest
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.experiment.workflows import MissingCapability, Run
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from zcu_lab.v2.twotone.time_domain.t2echo.core import (
    T2EchoCfg,
    T2EchoModuleCfg,
    T2EchoResult,
    T2EchoSweepCfg,
)
from zcu_lab.workflows.acquire import run_t2echo

from ._sim_inputs import Inputs


def echo_cfg(inputs: Inputs) -> T2EchoCfg:
    """Build a native echo cfg using the simulator fixture's calibrated pulses."""
    pi = inputs.t1.modules.pi_pulse
    return T2EchoCfg(
        reps=100,
        rounds=2,
        relax_delay=200.0,
        modules=T2EchoModuleCfg(
            pi_pulse=pi,
            pi2_pulse=pi.with_updates(gain=pi.gain / 2),
            readout=inputs.t1.modules.readout,
        ),
        sweep=T2EchoSweepCfg(length=SweepCfg(start=0.0, stop=40.0, expts=21, step=2.0)),
    )


def test_echo_uses_native_total_delay_and_acquisition_evidence(inputs: Inputs) -> None:
    cfg = echo_cfg(inputs)
    original_json = cfg.model_dump_json()
    run = Run(cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg)
    before = datetime.now(UTC)
    result = run_t2echo(run)
    assert isinstance(result, T2EchoResult)
    np.testing.assert_array_equal(
        result.times,
        sweep2array(cfg.sweep.length, "time", {"soccfg": inputs.soccfg, "scaler": 0.5}),
    )
    assert result.true_activate_detune == 0.0
    assert result.signals.dtype == np.complex128
    assert result.signals.shape == result.times.shape
    assert np.isfinite(result.signals).all()
    assert (
        abs(np.abs(result.signals[:3]).mean() - np.abs(result.signals[-3:]).mean())
        > 0.1
    )
    assert run.outcome.status == "completed"
    assert (
        before
        <= result.evidence.started_at
        <= result.evidence.finished_at
        <= datetime.now(UTC)
    )
    assert result.evidence.completion == "complete"
    assert cfg.model_dump_json() == original_json
    assert run.cfg.model_dump_json() == original_json


def test_echo_stop_preserves_native_partial_result(inputs: Inputs) -> None:
    stop = StopSignal()
    stop.set()
    run = Run(echo_cfg(inputs), stop, soc=inputs.soc, soccfg=inputs.soccfg)
    result = run_t2echo(run)
    assert run.outcome.status == "stopped"
    assert np.isnan(result.signals).all()
    assert result.evidence.completion == "stopped"


def test_echo_requires_soc_capability(inputs: Inputs) -> None:
    with pytest.raises(MissingCapability, match="soc"):
        run_t2echo(Run(echo_cfg(inputs), StopSignal()))


@pytest.mark.parametrize("detune", [-0.1, 0.1])
def test_echo_rejects_fringe_acquisition_before_schedule(
    inputs: Inputs, detune: float
) -> None:
    cfg = echo_cfg(inputs)
    cfg.detune = detune
    run = Run(cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg)
    with pytest.raises(ValueError, match="zero detune"):
        run_t2echo(run)
    assert not run.cancel_signal.is_set()


def test_echo_keeps_native_schedule_failure_and_partial_data(inputs: Inputs) -> None:
    cfg = echo_cfg(inputs)
    cfg.modules.pi2_pulse = cfg.modules.pi2_pulse.with_updates(
        waveform=ConstWaveformCfg(length=0.0)
    )
    run = Run(cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg)
    result = run_t2echo(run)
    assert run.outcome.status == "failed"
    assert isinstance(run.outcome.exception, RuntimeError)
    assert run.cancel_signal.error is not None
    assert run.cancel_signal.error.exception is run.outcome.exception
    assert np.isnan(result.signals).all()
    assert result.evidence.completion == "stopped"
