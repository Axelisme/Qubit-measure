from __future__ import annotations

from typing import Union, overload

import numpy as np
from numpy.typing import NDArray


@overload
def mA2flx(mA: float, mA_c: float, period: float, fold: bool = False) -> float: ...


@overload
def mA2flx(
    mA: NDArray[np.float64], mA_c: float, period: float, fold: bool = False
) -> NDArray[np.float64]: ...


def mA2flx(
    mA: Union[float, NDArray[np.float64]],
    mA_c: float,
    period: float,
    fold: bool = False,
) -> Union[float, NDArray[np.float64]]:
    flxs = (mA - mA_c) / period + 0.5
    if fold:
        flxs = np.mod(flxs, 1)
        flxs = np.where(flxs > 0.5, 1 - flxs, flxs)

    return flxs


def flx2mA(
    flxs: NDArray[np.float64], mA_c: float, period: float
) -> NDArray[np.float64]:
    return (flxs - 0.5) * period + mA_c


__all__ = ["mA2flx", "flx2mA"]
