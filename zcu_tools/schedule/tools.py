import warnings
from typing import Any, Dict

import numpy as np

from qick.asm_v2 import QickParam, QickSweep1D


def format_sweep1D(sweeps: Dict[str, Any], name: str) -> Dict[str, Any]:
    """
    Convert abbreviated single sweep to regular format

    Args:
        sweeps: dict, single sweep dict, abbreviated or not
        name: str, expected key name

    Returns:
        dict, regular format of the sweep
    """

    if "start" in sweeps and "stop" in sweeps:
        # conclude by key "start" and "stop"
        # use default name if not provided
        return {name: sweeps}

    # check if only one sweep is provided
    assert len(sweeps) == 1, "Only one sweep is allowed"
    assert sweeps.get(name) is not None, f"Key {name} is not found in the sweep"

    return sweeps


def check_time_sweep(soccfg, ts, gen_ch=None, ro_ch=None):
    """
    Check if time points are duplicated

    Args:
        soccfg: SocCfg, soc configuration
        ts: list, time points in us
        gen_ch: int, generator channel
        ro_ch: int, readout channel

    Returns:
        None
    """
    cycles = [soccfg.us2cycles(t, gen_ch=gen_ch, ro_ch=ro_ch) for t in ts]
    if len(set(cycles)) != len(ts):
        warnings.warn(
            "Some time points are duplicated, you sweep step may be too small"
        )


def map2adcfreq(soccfg, fpts, gen_ch, ro_ch):
    """
    Map frequencies to adc frequencies

    Args:
        soccfg: SocCfg, soc configuration
        fpts: array, frequencies in MHz
        gen_ch: int, generator channel
        ro_ch: array, readout channel

    Returns:
        mapped_fpts: list, adc frequencies in Hz
    """
    fpts = soccfg.adcfreq(fpts, gen_ch=gen_ch, ro_ch=ro_ch)
    if len(set(fpts)) != len(fpts):
        warnings.warn(
            "Some frequencies are duplicated, you sweep step may be too small"
        )
    return fpts


def sweep2array(sweep, allow_array=False):
    """
    Convert sweep to array

    Args:
        sweep: dict or array, sweep
        soft_loop: bool, whether to allow array sweep
        err_str: str, error message

    Returns:
        array, sweep array
    """
    if isinstance(sweep, dict):
        return sweep["start"] + np.arange(sweep["expts"]) * sweep["step"]
    elif isinstance(sweep, list) or isinstance(sweep, np.ndarray):
        if not allow_array:
            raise ValueError("Custom sweep is not allowed")
        return np.array(sweep)
    else:
        raise ValueError("Invalid sweep format")


def sweep2param(name: str, sweep: Dict[str, Any]) -> QickParam:
    """
    Convert formatted sweep to qick v2 sweep param

    Args:
        name: str, name of the sweep
        sweep: dict, formatted sweep

    Returns:
        QickParam, qick v2 sweep param
    """
    # convert formatted sweep to qick v2 sweep param
    return QickSweep1D(name, sweep["start"], sweep["stop"])
