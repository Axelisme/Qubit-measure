from __future__ import annotations

from typing import Optional, Union, overload

import numpy as np
from numpy.typing import NDArray

from zcu_tools.notebook.analysis.t1_curve.utils import calc_therm_ratio


@overload
def inductive_spectral_density(omega: float, Temp: float, EL: float) -> float: ...


@overload
def inductive_spectral_density(
    omega: NDArray[np.float64], Temp: float, EL: float
) -> NDArray[np.float64]: ...


def inductive_spectral_density(
    omega: Union[float, NDArray[np.float64]], Temp: float, EL: float
) -> Union[float, NDArray[np.float64]]:
    """omega: rad/ns, EL: GHz, T: K -> GHz"""
    therm_ratio = calc_therm_ratio(omega, Temp)
    return (
        2 * EL * (1 / np.tanh(0.5 * np.abs(therm_ratio))) / (1 + np.exp(-therm_ratio))
    )


def calc_ind_dipole(
    params: tuple[float, float, float],
    phi_elements: NDArray[np.complex128],
    omegas: NDArray[np.float64],
    Temp: float,
) -> NDArray[np.float64]:
    """Calculate the dipole of the inductive operator.

    params: GHz, omega: rad/ns, T: K, EL: GHz -> GHz
    """
    EJ, EC, EL = params

    spectral_densities = inductive_spectral_density(
        omegas, Temp, EL
    ) + inductive_spectral_density(-omegas, Temp, EL)
    return np.abs(phi_elements[..., 0, 1]) ** 2 * spectral_densities


@overload
def calc_Qind_vs_omega(
    params: tuple[float, float, float],
    omegas: NDArray[np.float64],
    T1s: NDArray[np.float64],
    phi_elements: NDArray[np.complex128],
    T1errs: None,
    Temp: float = 20e-3,
) -> NDArray[np.float64]: ...


@overload
def calc_Qind_vs_omega(
    params: tuple[float, float, float],
    omegas: NDArray[np.float64],
    T1s: NDArray[np.float64],
    phi_elements: NDArray[np.complex128],
    T1errs: NDArray[np.float64],
    Temp: float = 20e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def calc_Qind_vs_omega(
    params: tuple[float, float, float],
    omegas: NDArray[np.float64],
    T1s: NDArray[np.float64],
    phi_elements: NDArray[np.complex128],
    T1errs: Optional[NDArray[np.float64]] = None,
    Temp: float = 20e-3,
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """params: GHz, omegas: rad/ns, T1s: ns, guess_Temp: K -> 1"""
    EJ, EC, EL = params

    # calculate Qind vs omega
    dipoles = calc_ind_dipole(params, phi_elements, omegas, Temp)
    Qind_vs_omega = T1s * dipoles

    if T1errs is not None:
        Qind_vs_omega_err = T1errs * dipoles

        return Qind_vs_omega, Qind_vs_omega_err
    else:
        return Qind_vs_omega
