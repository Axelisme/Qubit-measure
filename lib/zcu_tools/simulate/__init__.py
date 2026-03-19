from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing_extensions import overload


@overload
def value2flx(
    value: float, flx_half: float, flx_period: float, fold: bool = False
) -> float: ...


@overload
def value2flx(
    value: NDArray[np.float64], flx_half: float, flx_period: float, fold: bool = False
) -> NDArray[np.float64]: ...


def value2flx(value, flx_half: float, flx_period: float, fold: bool = False):
    flxs = (value - flx_half) / flx_period + 0.5
    if fold:
        flxs = np.mod(flxs, 1)
        flxs = np.where(flxs > 0.5, 1 - flxs, flxs)

    return flxs


@overload
def flx2value(flxs: float, flx_half: float, flx_period: float) -> float: ...


@overload
def flx2value(
    flxs: NDArray[np.float64], flx_half: float, flx_period: float
) -> NDArray[np.float64]: ...


def flx2value(flxs, flx_half: float, flx_period: float):
    return (flxs - 0.5) * flx_period + flx_half


__all__ = ["value2flx", "flx2value"]
