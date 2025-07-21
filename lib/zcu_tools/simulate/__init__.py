import numpy as np


def mA2flx(mAs: np.ndarray, mA_c: float, period: float) -> np.ndarray:
    return (mAs - mA_c) / period + 0.5


def flx2mA(flxs: np.ndarray, mA_c: float, period: float) -> np.ndarray:
    return (flxs - 0.5) * period + mA_c


__all__ = ["mA2flx", "flx2mA"]
