import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.schedule.tools import format_sweep1D, sweep2array, sweep2param
from zcu_tools.schedule.v2.template import sweep_template


def measure_res_freq(soc, soccfg, cfg, progress=True, instant_show=False):
    cfg = make_cfg(cfg)  # prevent in-place modification

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    cfg["dac"]["res_pulse"]["freq"] = sweep2param("freq", sweep_cfg)

    fpts = sweep2array(sweep_cfg, False)  # predicted frequency points

    prog, signals = sweep_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        init_signals=np.full(len(fpts), np.nan, dtype=complex),
        ticks=(fpts,),
        progress=progress,
        instant_show=instant_show,
        xlabel="Frequency (MHz)",
        ylabel="Amplitude",
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return fpts, signals
