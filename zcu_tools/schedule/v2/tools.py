import warnings
from typing import Any, Dict, Optional

import numpy as np


def format_sweep(sweep: Dict[str, Any], default_name: str) -> Dict[str, Any]:
    # convert abbreviated single sweep to regular format
    if "start" in sweep and "stop" in sweep:
        # conclude by key "start" and "stop"
        return {default_name: sweep}

    return sweep


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


def sweep2param(sweep: Dict[str, Any], name: Optional[str] = None):
    from qick.asm_v2 import QickSweep1D

    # convert formatted sweep to qick v2 sweep param
    assert sweep, "Sweep should not be empty"
    if name is None:
        # use the first key as the name
        name = list(sweep.keys())[0]

    return QickSweep1D(name, sweep[name]["start"], sweep[name]["stop"])
