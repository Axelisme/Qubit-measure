import numpy as np

from zcu_tools import make_cfg
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.schedule.tools import (
    format_sweep1D,
    map2adcfreq,
    sweep2array,
    sweep2param,
)
from zcu_tools.schedule.v2.template import sweep_hard_template


def measure_res_freq(soc, soccfg, cfg, progress=True):
    cfg = make_cfg(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

    sweep_cfg = cfg["sweep"]["freq"]
    res_pulse["freq"] = sweep2param("freq", sweep_cfg)

    fpts = sweep2array(sweep_cfg)  # predicted frequency points
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    prog, signals = sweep_hard_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        init_signals=np.full(len(fpts), np.nan, dtype=complex),
        ticks=(fpts,),
        progress=progress,
        xlabel="Frequency (MHz)",
        ylabel="Amplitude",
        signals2real=np.abs,
    )

    # get the actual frequency points
    fpts = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    return fpts, signals
