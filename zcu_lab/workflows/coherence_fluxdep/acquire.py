"""coherence_fluxdep acquire helpers for lab workflows."""

from copy import deepcopy

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    TimelineBuilder,
)

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyCfg,
    T2RamseyResult,
    t2ramsey_signal2real,
)


def run_t2ramsey(cfg: T2RamseyCfg, *, context: RunContext) -> T2RamseyResult:
    """Measure Ramsey fringes to observe dephasing between two pi/2 pulses.

    Supply complete native cfg with scalar detune in MHz and delay sweep in
    microseconds. context supplies SoC, plots and the cancellation token.
    The second pi/2 phase advances by 360*detune*delay. Return rounded delays,
    complex IQ and actual activated detune in MHz, retaining partial data on
    stop. Schedule records acquisition failures on the token; other errors
    propagate. No setup, fit or save occurs, and cfg remains unchanged.
    """
    cfg = deepcopy(cfg)
    soc, soccfg = context.soc, context.soccfg
    times = sweep2array(cfg.delay, "time", {"soccfg": soccfg})
    fig, ax = context.plots.subplots(num="measurement")
    ax.set(xlabel="Time (us)", ylabel="Signal")
    plot = live_line(ax)

    @context.plots.on_plot_thread
    def update(samples):
        return plot.set(times, np.real(samples))

    fig.show()
    buffer = SignalBuffer((len(times),), on_update=update)
    # Execute the pulse sequence using locally constructed modules and sweeps.
    with Schedule( buffer, scope=context.scope) as sched:
        builder = TimelineBuilder(soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay)
        delay = builder.add_sweep("delay", cfg.delay)
        reset = Reset("reset", cfg.reset)
        if reset is not None:
            builder.emit(reset)
        builder.emit(Pulse("pi2_pulse", cfg.pi2_pulse))
        builder.timeline_inc(delay)
        builder.emit(
            Pulse(
                "pi2_pulse",
                cfg.pi2_pulse,
                phase=cfg.pi2_pulse.phase + 360 * cfg.detune * delay,
            )
        )
        builder.emit(Readout("readout", cfg.readout))
        sched.run_acquire(
            builder.build(),
            soc,
            rounds=cfg.rounds,
            options=RunOptions(
                retry=cfg.acquire_retry,
                stop_condition=snr_checker(
                    buffer.at(), cfg.earlystop_snr, t2ramsey_signal2real
                ),
            ),
        )
    return T2RamseyResult(
        times=times,
        signals=buffer.array,
        true_activate_detune=actual_detune(
            cfg.delay,
            detune=cfg.detune,
            gen_ch=cfg.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=1.0,
        ),
    )
