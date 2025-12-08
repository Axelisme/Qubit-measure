import warnings
from typing import Mapping, MutableMapping, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeVar

from zcu_tools.program import SweepCfg

T = TypeVar("T", bound=Union[SweepCfg, NDArray])


def format_sweep1D(
    sweep: Union[Mapping[str, T], T], name: str
) -> MutableMapping[str, T]:
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

    if isinstance(sweep, np.ndarray) or isinstance(sweep, list):
        return {name: cast(T, np.asarray(sweep))}

    elif isinstance(sweep, dict):
        # conclude by key "start" and "stop"
        if "start" in sweep and "stop" in sweep:
            # it is in abbreviated format
            return {name: cast(T, sweep)}

        # check if only one sweep is provided
        assert len(sweep) == 1, "Only one sweep is allowed"
        assert sweep.get(name) is not None, f"Key {name} is not found in the sweep"

        # it is already in regular format
        return sweep
    else:
        raise ValueError(sweep)


def check_time_sweep(
    soccfg, ts, gen_ch: Optional[int] = None, ro_ch: Optional[int] = None
) -> None:
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


def sweep2array(
    sweep: Union[SweepCfg, NDArray[np.float64]], allow_array: bool = False
) -> NDArray[np.float64]:
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
        return np.asarray(sweep)
    else:
        raise ValueError("Invalid sweep format")


def make_ge_sweep() -> SweepCfg:
    return {"start": 0.0, "stop": 1.0, "expts": 2, "step": 0.5}
