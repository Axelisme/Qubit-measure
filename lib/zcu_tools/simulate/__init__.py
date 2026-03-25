from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing_extensions import overload


@overload
def value2flux(
    value: float, flux_half: float, flux_period: float, fold: bool = False
) -> float: ...


@overload
def value2flux(
    value: NDArray[np.float64], flux_half: float, flux_period: float, fold: bool = False
) -> NDArray[np.float64]: ...


def value2flux(value, flux_half: float, flux_period: float, fold: bool = False):
    fluxs = (value - flux_half) / flux_period + 0.5
    if fold:
        fluxs = np.mod(fluxs, 1)
        fluxs = np.where(fluxs > 0.5, 1 - fluxs, fluxs)

    return fluxs


@overload
def flux2value(fluxs: float, flux_half: float, flux_period: float) -> float: ...


@overload
def flux2value(
    fluxs: NDArray[np.float64], flux_half: float, flux_period: float
) -> NDArray[np.float64]: ...


def flux2value(fluxs, flux_half: float, flux_period: float):
    return (fluxs - 0.5) * flux_period + flux_half


__all__ = ["value2flux", "flux2value"]
