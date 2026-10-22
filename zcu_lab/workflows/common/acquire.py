"""Custom spectroscopy sequence; shared sequences use bound core run methods."""

from copy import deepcopy

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import sweep2param

from zcu_lab.v2.twotone.freq.core import FreqCfg, FreqResult


def run_qubit_freq(cfg: FreqCfg, *, context: RunContext) -> FreqResult:
    """Sweep drive frequency after the lab's optional initialization pulse.

    Reset and optional init_pulse precede the frequency-swept qubit pulse and
    readout. Unlike FreqExp.run, this sequence retains init_pulse. Return rounded
    MHz coordinates and complex IQ of matching shape, including partial data
    on stop. context supplies SoC, plots and the cooperative cancellation token.
    Schedule records acquisition failures on that token; other errors propagate.
    No device setup, analysis or saving occurs, and cfg remains unchanged.
    """
    cfg = deepcopy(cfg)
    soc, soccfg = context.soc, context.soccfg
    freqs = sweep2array(
        cfg.sweep.freq,
        "freq",
        {"soccfg": soccfg, "gen_ch": cfg.modules.qub_pulse.ch},
    )
    plot = context.plots.liveplot_1d(
        "measurement",
        "Frequency (MHz)",
        "Signal",
        num_lines=2,
    )
    buffer = SignalBuffer(
        (len(freqs),),
        on_update=lambda samples: plot.update(
            freqs, np.stack((np.abs(samples), np.full_like(np.real(samples), np.nan)))
        ),
    )
    # Keep initialization: the standard core frequency sequence omits it.
    with Schedule(cfg, buffer, scope=context.scope) as sched:
        modules = sched.cfg.modules
        freq_sweep = sched.cfg.sweep.freq
        modules.qub_pulse.set_param("freq", sweep2param("freq", freq_sweep))
        (
            sched.prog_builder(soc, soccfg)
            .add_reset("reset", modules.reset)
            .add_pulse("init_pulse", modules.init_pulse)
            .add_pulse("qub_pulse", modules.qub_pulse)
            .add_readout("readout", modules.readout)
            .declare_sweep("freq", freq_sweep)
            .build_and_acquire()
        )

    return FreqResult(freqs=freqs, signals=buffer.array)
