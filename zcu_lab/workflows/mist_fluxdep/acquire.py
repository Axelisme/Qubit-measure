"""mist_fluxdep acquire helpers for lab workflows."""

from copy import deepcopy

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    TimelineBuilder,
)

from zcu_lab.v2.mist.power_dep.single_trace.core import PowerDepCfg, PowerDepResult
from zcu_lab.v2.support import live_line
from zcu_lab.workflows.mist_fluxdep.analysis import mist_signal2real


def run_mist(cfg: PowerDepCfg, *, context: RunContext, retry: int) -> PowerDepResult:
    """Acquire a disturbance gain trace after optional pi preparation.

    Supply native PowerDepCfg: init_pulse is the calibrated pi pulse for excited
    preparation, or None for ground. probe_pulse is the fixed disturbance.
    Native optional reset retains its meaning. gain SweepCfg is fixed by plan.
    Return actual rounded gains and complex IQ with partial data on stop.
    context supplies SoC, plots and the cancellation token. retry is the number
    of retries after a transient acquisition failure, from workflow settings.
    Schedule records acquisition failures
    on the token; other errors propagate. No fitting, device setup or saving
    occurs, and cfg remains unchanged.
    """
    cfg = deepcopy(cfg)
    soc, soccfg = context.soc, context.soccfg
    gains = sweep2array(
        cfg.gain, "gain", {"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch}
    )
    fig, ax = context.plots.subplots(num="mist/gain")
    ax.set(xlabel="MIST gain (a.u.)", ylabel="MIST")
    plot = live_line(ax)

    @context.plots.on_plot_thread
    def update(samples):
        return plot.set(gains, mist_signal2real(samples))

    fig.show()
    buffer = SignalBuffer(
        (len(gains),),
        on_update=update,
    )
    with Schedule(buffer, scope=context.scope) as sched:
        builder = TimelineBuilder(soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay)
        gain = builder.add_sweep("gain", cfg.gain)
        reset = Reset("reset", cfg.reset)
        if reset is not None:
            builder.emit(reset)
        builder.emit(Pulse("init_pulse", cfg.init_pulse))
        builder.emit(Pulse("probe_pulse", cfg.probe_pulse, gain=gain))
        builder.emit(Readout("readout", cfg.readout))
        sched.run_acquire(
            builder.build(),
            soc,
            rounds=cfg.rounds,
            options=RunOptions(retry=retry),
        )
    return PowerDepResult(gains, buffer.array)
