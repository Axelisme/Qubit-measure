import warnings
from typing import Any, Dict
from qick.asm_v2 import QickSweep1D, QickParam

import numpy as np


def format_sweep1D(sweeps: Dict[str, Any], name: str) -> Dict[str, Any]:
    # convert abbreviated single sweep to regular format
    # if already in regular format, check it key is correct

    if "start" in sweeps and "stop" in sweeps:
        # conclude by key "start" and "stop"
        # use default name if not provided
        return {name: sweeps}

    # check if only one sweep is provided
    assert len(sweeps) == 1, "Only one sweep is allowed"
    assert sweeps.get(name) is not None, f"Key {name} is not found in the sweep"

    return sweeps


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


def sweep2param(name: str, sweep: Dict[str, Any]) -> QickParam:
    # convert formatted sweep to qick v2 sweep param
    return QickSweep1D(name, sweep["start"], sweep["stop"])
