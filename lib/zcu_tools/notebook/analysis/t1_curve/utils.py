from __future__ import annotations

from typing import Union, overload

import numpy as np
import scipy.constants as sp
from numpy.typing import NDArray


def format_exponent(n: float) -> str:
    base, exp = "{:.2e}".format(n).split("e")
    clean_exp = int(exp)
    return rf"${base} \times 10^{{{clean_exp}}}$"


@overload
def freq2omega(freqs: float) -> float: ...


@overload
def freq2omega(freqs: NDArray[np.float64]) -> NDArray[np.float64]: ...


def freq2omega(
    freqs: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """GHz -> rad/ns"""
    return 2 * np.pi * freqs  # type: ignore

@overload
def convert_eV_to_Hz(val: float) -> float: ...


@overload
def convert_eV_to_Hz(val: NDArray[np.float64]) -> NDArray[np.float64]: ...


def convert_eV_to_Hz(
    val: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Convert a value in electron volts to Hz."""
    return val * sp.e / sp.h

@overload
def calc_therm_ratio(omega: float, T: float) -> float: ...


@overload
def calc_therm_ratio(omega: NDArray[np.float64], T: float) -> NDArray[np.float64]: ...


def calc_therm_ratio(
    omega: Union[float, NDArray[np.float64]],
    T: float,
) -> Union[float, NDArray[np.float64]]:
    """omega: rad/ns, T: K"""
    return (sp.hbar * omega * 1e9) / (sp.k * T)  # type: ignore
