"""coherence_fluxdep acquire helpers for lab workflows."""

from copy import deepcopy

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import (
    Delay,
    Pulse,
    Readout,
    Reset,
    sweep2param,
)

from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyCfg, T2RamseyResult


def run_t2ramsey(cfg: T2RamseyCfg, *, context: RunContext) -> T2RamseyResult:
    """Measure Ramsey fringes to observe dephasing between two pi/2 pulses.

    Supply complete native cfg with scalar detune in MHz and length sweep in
    microseconds. context supplies SoC, plots and the cancellation token.
    The second pi/2 phase advances by 360*detune*delay. Return rounded delays,
    complex IQ and actual activated detune in MHz, retaining partial data on
    stop. Schedule records acquisition failures on the token; other errors
    propagate. No setup, fit or save occurs, and cfg remains unchanged.
    """
    cfg = deepcopy(cfg)
    soc, soccfg = context.soc, context.soccfg
    times = sweep2array(cfg.sweep.length, "time", {"soccfg": soccfg})
    plot = context.plots.liveplot_1d("measurement", "Time (us)", "Signal")
    buffer = SignalBuffer(
        (len(times),), on_update=lambda samples: plot.update(times, np.real(samples))
    )
    # Execute the pulse sequence using schedule-local modules and sweeps.
    with Schedule(cfg, buffer, scope=context.scope) as sched:
        modules = sched.cfg.modules
        length_sweep = cfg.sweep.length
        length_param = sweep2param("length", length_sweep)
        (
            sched.prog_builder(soc, soccfg)
            .add(
                Reset("reset", modules.reset),
                Pulse("pi2_pulse1", modules.pi2_pulse),
                Delay("t2r_delay", delay=length_param),
                Pulse(
                    "pi2_pulse2",
                    modules.pi2_pulse.with_updates(
                        phase=modules.pi2_pulse.phase + 360 * cfg.detune * length_param
                    ),
                ),
                Readout("readout", modules.readout),
            )
            .declare_sweep("length", length_sweep)
            .build_and_acquire()
        )
    return T2RamseyResult(
        times=times,
        signals=buffer.array,
        true_activate_detune=actual_detune(
            cfg.sweep.length,
            detune=cfg.detune,
            gen_ch=cfg.modules.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=1.0,
        ),
    )
