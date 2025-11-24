import warnings
from typing import Optional, Union, cast, overload

import numpy as np
from numpy.typing import NDArray

from zcu_tools.library import ModuleLibrary
from zcu_tools.program.v2 import PulseCfg


@overload
def check_gains(gains: NDArray[np.float64], name: str) -> NDArray[np.float64]: ...


@overload
def check_gains(gains: float, name: str) -> float: ...


def check_gains(
    gains: Union[float, NDArray[np.float64]], name: str
) -> Union[float, NDArray[np.float64]]:
    if np.any(gains > 1.0):
        warnings.warn(
            f"Some {name} gains are larger than 1.0, force clip to 1.0, which may cause distortion."
        )
        gains = np.clip(gains, 0.0, 1.0)
    return gains


def make_pulse(
    ml: ModuleLibrary,
    pulse_name: str,
    freq: Optional[float] = None,
    gain: Optional[float] = None,
    length: Optional[float] = None,
) -> PulseCfg:
    pulse_cfg = ml.get_module(pulse_name)
    if freq is not None:
        pulse_cfg["freq"] = freq
    if gain is not None:
        pulse_cfg["gain"] = gain
    if length is not None:
        pulse_cfg["waveform"]["length"] = length
    return cast(PulseCfg, pulse_cfg)
