"""Ramsey callback reports the native phase/time-quantized activated detune."""

import numpy as np
import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import Run
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyModuleCfg,
    T2RamseySweepCfg,
)
from zcu_lab.workflows.acquire import run_t2ramsey

from ._sim_inputs import Inputs


def test_ramsey_reports_actual_native_activated_detune_not_nominal(
    inputs: Inputs,
) -> None:
    cfg = T2RamseyCfg(
        reps=100,
        rounds=2,
        relax_delay=200.0,
        detune=0.1234567,
        modules=T2RamseyModuleCfg(
            pi2_pulse=inputs.t1.modules.pi_pulse.with_updates(
                waveform=ConstWaveformCfg(length=0.2)
            ),
            readout=inputs.t1.modules.readout,
        ),
        sweep=T2RamseySweepCfg(
            length=SweepCfg(start=0.0, stop=2.74, expts=21, step=0.137)
        ),
    )
    plots = Plots(NonPresentingHost())
    try:
        native = T2RamseyExp().run(
            cfg, context=RunContext(inputs.soc, inputs.soccfg, plots, {}, StopSignal())
        )
    finally:
        plots.finish()
        plots.release()
    assert native.true_activate_detune is not None
    assert abs(native.true_activate_detune - cfg.detune) > 1e-8
    run = Run(
        cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg, device_snapshot={}
    )
    result = run_t2ramsey(run)
    assert run.outcome.status == "completed"
    np.testing.assert_array_equal(result.times, native.times)
    assert result.true_activate_detune == pytest.approx(
        native.true_activate_detune, rel=1e-12, abs=1e-12
    )
    assert cfg.detune == 0.1234567
