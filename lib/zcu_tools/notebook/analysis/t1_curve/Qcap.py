from __future__ import annotations

from typing import Optional, Union, overload

import numpy as np
from numpy.typing import NDArray

from zcu_tools.notebook.analysis.t1_curve.utils import calc_therm_ratio


@overload
def charge_spectral_density(omega: float, Temp: float, EC: float) -> float: ...


@overload
def charge_spectral_density(
    omega: NDArray[np.float64], Temp: float, EC: float
) -> NDArray[np.float64]: ...


def charge_spectral_density(
    omega: Union[float, NDArray[np.float64]], Temp: float, EC: float
) -> Union[float, NDArray[np.float64]]:
    """omega: rad/ns, EC: GHz, T: K -> GHz"""
    therm_ratio = calc_therm_ratio(omega, Temp)
    return (
        2
        * 8
        * EC
        * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
        / (1 + np.exp(-therm_ratio))
    )

def calc_cap_dipole(
    params: tuple[float, float, float],
    n_elements: NDArray[np.complex128],
    omegas: NDArray[np.float64],
    Temp: float,
) -> NDArray[np.float64]:
    """Calculate the dipole of the charge operator.

    params: GHz, omega: rad/ns, T: K, EC: GHz -> GHz
    """
    EJ, EC, EL = params

    spectral_densities = charge_spectral_density(
        omegas, Temp, EC
    ) + charge_spectral_density(-omegas, Temp, EC)
    return np.abs(n_elements[..., 0, 1]) ** 2 * spectral_densities


@overload
def calc_Qcap_vs_omega(
    params: tuple[float, float, float],
    omegas: NDArray[np.float64],
    T1s: NDArray[np.float64],
    n_elements: NDArray[np.complex128],
    T1errs: None,
    Temp: float = 20e-3,
) -> NDArray[np.float64]: ...


@overload
def calc_Qcap_vs_omega(
    params: tuple[float, float, float],
    omegas: NDArray[np.float64],
    T1s: NDArray[np.float64],
    n_elements: NDArray[np.complex128],
    T1errs: NDArray[np.float64],
    Temp: float = 20e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def calc_Qcap_vs_omega(
    params: tuple[float, float, float],
    omegas: NDArray[np.float64],
    T1s: NDArray[np.float64],
    n_elements: NDArray[np.complex128],
    T1errs: Optional[NDArray[np.float64]] = None,
    Temp: float = 20e-3,
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """fpts: GHz, T1s: ns, guess_Temp: K -> 1"""
    EJ, EC, EL = params

    dipoles = calc_cap_dipole(params, n_elements, omegas, Temp)
    Qcap_vs_omega = T1s * dipoles

    if T1errs is not None:
        Qcap_vs_omega_err = T1errs * dipoles

        return Qcap_vs_omega, Qcap_vs_omega_err
    else:
        return Qcap_vs_omega
