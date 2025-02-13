import warnings

import numpy as np


def check_time_sweep(soccfg, ts, gen_ch=None, ro_ch=None):
    cycles = [soccfg.us2cycles(t, gen_ch=gen_ch, ro_ch=ro_ch) for t in ts]
    if len(set(cycles)) != len(ts):
        warnings.warn(
            "Some time points are duplicated, you sweep step may be too small"
        )


def map2adcfreq(soccfg, fpts, gen_ch, ro_ch):
    fpts = soccfg.adcfreq(fpts, gen_ch=gen_ch, ro_ch=ro_ch)
    if len(set(fpts)) != len(fpts):
        warnings.warn(
            "Some frequencies are duplicated, you sweep step may be too small"
        )
    return fpts


def sweep2array(sweep, soft_loop=True, err_str=None):
    if isinstance(sweep, dict):
        return sweep["start"] + np.arange(sweep["expts"]) * sweep["step"]

    assert soft_loop, err_str
    return np.array(sweep)
