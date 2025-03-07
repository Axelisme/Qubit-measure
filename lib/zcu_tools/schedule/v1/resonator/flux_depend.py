from copy import deepcopy

import numpy as np

from zcu_tools.analysis import minus_background, rescale
from zcu_tools.program.v1 import OneToneProgram
from zcu_tools.schedule.tools import map2adcfreq, sweep2array
from zcu_tools.schedule.v1.template import sweep2D_soft_soft_template


def signal2real(signals):
    return rescale(minus_background(np.abs(signals), axis=1), axis=1)


def measure_res_flux_dep(soc, soccfg, cfg):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if cfg["dev"]["flux_dev"] == "none":
        raise NotImplementedError("Flux sweep but get flux_dev == 'none'")

    res_pulse = cfg["dac"]["res_pulse"]

    fpt_sweep = cfg["sweep"]["freq"]
    fpts = sweep2array(fpt_sweep, allow_array=True)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

    flx_sweep = cfg["sweep"]["flux"]
    flxs = sweep2array(flx_sweep, allow_array=True)

    del cfg["sweep"]  # remove sweep for program use

    def x_updateCfg(cfg, _, flx):
        cfg["dev"]["flux"] = flx

    def y_updateCfg(cfg, _, fpt):
        cfg["dac"]["res_pulse"]["freq"] = fpt

    flxs, fpts, signals2D = sweep2D_soft_soft_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        xs=flxs,
        ys=fpts,
        x_updateCfg=x_updateCfg,
        y_updateCfg=y_updateCfg,
        xlabel="Flux (a.u.)",
        ylabel="Frequency (MHz)",
        signal2real=signal2real,
    )

    return flxs, fpts, signals2D
