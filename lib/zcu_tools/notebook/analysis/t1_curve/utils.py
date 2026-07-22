from __future__ import annotations

from typing import overload

import numpy as np
import scipy.constants as sp
from numpy.typing import NDArray

from zcu_tools.notebook.analysis.fit_tools import (
    F01FluxCorrectionResult as F01FluxCorrectionResult,
)
from zcu_tools.notebook.analysis.fit_tools import (
    correct_flux_from_f01 as correct_flux_from_f01,
)


def format_exponent(n: float) -> str:
    if not np.isfinite(n):
        return "NaN"
    base, exp = "{:.2e}".format(n).split("e")
    clean_exp = int(exp)
    return rf"${base} \times 10^{{{clean_exp}}}$"


@overload
def freq2omega(freqs: float) -> float: ...


@overload
def freq2omega(freqs: NDArray[np.float64]) -> NDArray[np.float64]: ...


def freq2omega(
    freqs: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """GHz -> rad/ns"""
    return 2 * np.pi * freqs  # type: ignore


@overload
def convert_eV_to_Hz(val: float) -> float: ...


@overload
def convert_eV_to_Hz(val: NDArray[np.float64]) -> NDArray[np.float64]: ...


def convert_eV_to_Hz(
    val: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Convert a value in electron volts to Hz."""
    return val * sp.e / sp.h


@overload
def calc_therm_ratio(omega: float, T: float) -> float: ...


@overload
def calc_therm_ratio(omega: NDArray[np.float64], T: float) -> NDArray[np.float64]: ...


def calc_therm_ratio(
    omega: float | NDArray[np.float64],
    T: float,
) -> float | NDArray[np.float64]:
    """omega: rad/ns, T: K"""
    return (sp.hbar * omega * 1e9) / (sp.k * T)  # type: ignore
