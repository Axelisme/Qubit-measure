"""mist_fluxdep acquire helpers for lab workflows."""

from copy import deepcopy

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    sweep2param,
)

from zcu_lab.v2.mist.power_dep.single_trace.core import PowerDepCfg, PowerDepResult
from zcu_lab.workflows.mist_fluxdep.analysis import mist_signal2real


def run_mist(cfg: PowerDepCfg, *, context: RunContext) -> PowerDepResult:
    """Acquire a disturbance gain trace to resolve MIST after calibrated pi preparation.

    Supply native PowerDepCfg: init_pulse is required here and represents the
    calibrated pi pulse; probe_pulse is the fixed MIST disturbance waveform.
    Native optional reset retains its meaning. gain SweepCfg is fixed by plan.
    Return actual rounded gains and complex IQ with partial data on stop.
    context supplies SoC, plots and the cancellation token. Missing preparation
    raises ValueError before acquisition. Schedule records acquisition failures
    on the token; other errors propagate. No fitting, device setup or saving
    occurs, and cfg remains unchanged.
    """
    cfg = deepcopy(cfg)
    if cfg.modules.init_pulse is None:
        raise ValueError("MIST acquisition requires a calibrated pi preparation")
    soc, soccfg = context.soc, context.soccfg
    gains = sweep2array(
        cfg.sweep.gain, "gain", {"soccfg": soccfg, "gen_ch": cfg.modules.probe_pulse.ch}
    )
    plot = context.plots.liveplot_1d("mist/gain", "MIST gain (a.u.)", "MIST")
    buffer = SignalBuffer(
        (len(gains),),
        on_update=lambda samples: plot.update(gains, mist_signal2real(samples)),
    )
    with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
        modules = sched.cfg.modules
        gain_sweep = sched.cfg.sweep.gain
        modules.probe_pulse.set_param("gain", sweep2param("gain", gain_sweep))
        (
            sched.prog_builder(soc, soccfg)
            .add(
                Reset("reset", modules.reset),
                Pulse("pi_pulse", modules.init_pulse),
                Pulse("mist_pulse", modules.probe_pulse),
                Readout("readout", modules.readout),
            )
            .declare_sweep("gain", gain_sweep)
            .build_and_acquire()
        )
    return PowerDepResult(gains, buffer.array)
