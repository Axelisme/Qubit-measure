from copy import deepcopy

import numpy as np

from zcu_tools.analysis import minus_background, rescale
from zcu_tools.program.v1 import OneToneProgram
from zcu_tools.schedule.tools import map2adcfreq, sweep2array
from zcu_tools.schedule.v1.template import sweep2D_soft_soft_template


def signal2real(signals):
    return rescale(minus_background(np.abs(signals), axis=1), axis=1)


def measure_res_pdr_dep(soc, soccfg, cfg, dynamic_reps=False, gain_ref=1000):
    cfg = deepcopy(cfg)  # prevent in-place modification

    res_pulse = cfg["dac"]["res_pulse"]
    reps_ref = cfg["reps"]

    fpts = sweep2array(cfg["sweep"]["freq"], allow_array=True)
    fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])
    pdrs = sweep2array(cfg["sweep"]["gain"], allow_array=True)

    del cfg["sweep"]  # remove sweep for program use

    def x_updateCfg(cfg, _, pdr):
        cfg["dac"]["res_pulse"]["gain"] = pdr

        if dynamic_reps:
            cfg["reps"] = int(reps_ref * gain_ref / pdr)
            if cfg["reps"] < 0.1 * reps_ref:
                cfg["reps"] = int(0.1 * reps_ref + 0.99)
            elif cfg["reps"] > 10 * reps_ref:
                cfg["reps"] = int(10 * reps_ref)

    def y_updateCfg(cfg, _, fpt):
        cfg["dac"]["res_pulse"]["freq"] = fpt

    pdrs, fpts, signals2D = sweep2D_soft_soft_template(
        soc,
        soccfg,
        cfg,
        OneToneProgram,
        xs=pdrs,
        ys=fpts,
        x_updateCfg=x_updateCfg,
        y_updateCfg=y_updateCfg,
        xlabel="Power (a.u.)",
        ylabel="Frequency (MHz)",
        signal2real=signal2real,
    )

    return pdrs, fpts, signals2D
