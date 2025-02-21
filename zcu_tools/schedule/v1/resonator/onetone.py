from copy import deepcopy

import numpy as np

from zcu_tools.program.v1 import OneToneProgram
from zcu_tools.schedule.tools import map2adcfreq, sweep2array
from zcu_tools.schedule.v1.template import sweep1D_soft_template


def measure_res_freq(soc, soccfg, cfg, instant_show=False):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]

    fpts = sweep2array(cfg["sweep"])
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    def update_cfg(cfg, _, f):
        cfg["dac"]["res_pulse"]["freq"] = f

    fpts, signals = sweep1D_soft_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        xs=fpts,
        init_signals=np.full(len(fpts), np.nan, dtype=np.complex128),
        progress=True,
        instant_show=instant_show,
        signal2amp=np.abs,
        updateCfg=update_cfg,
        xlabel="Frequency (MHz)",
        ylabel="Amplitude",
    )

    return fpts, signals
