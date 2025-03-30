import warnings
from typing import Any, Dict, Union

import numpy as np
from myqick.asm_v2 import QickParam, QickSweep1D


def format_sweep1D(
    sweep: Union[Dict[str, Any], np.ndarray], name: str
) -> Dict[str, Any]:
    """
    Convert abbreviated single sweep to regular format.

    This function takes a sweep parameter in different formats and converts it
    to a standardized dictionary format with a specified key name.

    Args:
        sweep: A dictionary containing sweep parameters (with 'start' and 'stop' keys)
               or a numpy array of values to sweep through
        name: Expected key name for the sweep in the returned dictionary

    Returns:
        A dictionary in regular format with 'name' as the key
    """

    if isinstance(sweep, np.ndarray):
        return {name: sweep}

    elif isinstance(sweep, dict):
        # conclude by key "start" and "stop"
        if "start" in sweep and "stop" in sweep:
            # it is in abbreviated format
            return {name: sweep}

        # check if only one sweep is provided
        assert len(sweep) == 1, "Only one sweep is allowed"
        assert sweep.get(name) is not None, f"Key {name} is not found in the sweep"

        # it is already in regular format
        return sweep


def check_time_sweep(soccfg, ts, gen_ch=None, ro_ch=None):
    """
    Check if time points are duplicated when converted to machine cycles.

    This function converts the given time points to cycles using the soccfg
    and checks if any cycles are duplicated, which would indicate that the
    sweep step size is too small.

    Args:
        soccfg: SocCfg object containing the system configuration
        ts: List of time points in microseconds (us)
        gen_ch: Generator channel number (optional)
        ro_ch: Readout channel number (optional)

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
    Map frequencies to ADC frequencies.

    This function converts the input frequencies to ADC frequencies using the
    system configuration and checks for any duplicated frequencies after conversion.

    Args:
        soccfg: SocCfg object containing the system configuration
        fpts: Array of frequencies in MHz
        gen_ch: Generator channel number
        ro_ch: Readout channel number or array of readout channels

    Returns:
        Array of mapped ADC frequencies in Hz
    """
    fpts = soccfg.adcfreq(fpts, gen_ch=gen_ch, ro_ch=ro_ch)
    if len(set(fpts)) != len(fpts):
        warnings.warn(
            "Some frequencies are duplicated, you sweep step may be too small"
        )
    return fpts


def sweep2array(sweep, allow_array=False) -> np.ndarray:
    """
    Convert sweep parameter to a numpy array.

    This function converts different sweep parameter formats into a numpy array
    of values to sweep through.

    Args:
        sweep: Dictionary with 'start', 'step', and 'expts' keys defining the sweep,
               or a list/array of explicit sweep values
        allow_array: Whether to allow direct array input for custom sweeps (default: False)

    Returns:
        numpy.ndarray: Array of sweep values
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
    Convert formatted sweep dictionary to a QickSweep1D parameter.

    This function creates a QickSweep1D parameter from a formatted sweep dictionary,
    which is used in Qick v2 assembly programming.

    Args:
        name: Name of the sweep parameter
        sweep: Dictionary containing 'start' and 'stop' values for the sweep

    Returns:
        QickSweep1D: Qick v2 sweep parameter object
    """
    # convert formatted sweep to qick v2 sweep param
    return QickSweep1D(name, sweep["start"], sweep["stop"])
