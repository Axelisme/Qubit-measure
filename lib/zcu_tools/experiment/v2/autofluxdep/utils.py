import warnings
from typing import Union, overload

import numpy as np
from numpy.typing import NDArray


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
