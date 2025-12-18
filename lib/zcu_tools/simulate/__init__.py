import numpy as np
from numpy.typing import NDArray


def mA2flx(
    mAs: NDArray[np.float64], mA_c: float, period: float, fold: bool = False
) -> NDArray[np.float64]:
    flxs = (mAs - mA_c) / period + 0.5
    if fold:
        flxs = np.mod(flxs, 1)
        flxs = np.where(flxs > 0.5, 1 - flxs, flxs)

    return flxs


def flx2mA(
    flxs: NDArray[np.float64], mA_c: float, period: float
) -> NDArray[np.float64]:
    return (flxs - 0.5) * period + mA_c


__all__ = ["mA2flx", "flx2mA"]
