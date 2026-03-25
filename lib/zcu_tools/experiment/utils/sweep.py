from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Mapping, TypeVar, Union, cast

from zcu_tools.program import SweepCfg

T = TypeVar("T", bound=Union[SweepCfg, NDArray])


def format_sweep1D(sweep: Union[Mapping[str, T], T], name: str) -> dict[str, T]:
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
        return dict(sweep)
    else:
        raise ValueError(sweep)
