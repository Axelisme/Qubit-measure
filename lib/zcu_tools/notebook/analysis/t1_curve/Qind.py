from __future__ import annotations

from typing import overload

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
    omega: float | NDArray[np.float64], Temp: float, EL: float
) -> float | NDArray[np.float64]:
    """Unit-Q inductive spectral rate in ns⁻¹ for omega (rad/ns), Temp (K), EL (GHz)."""
    therm_ratio = calc_therm_ratio(omega, Temp)
    return (
        2
        * EL
        * (2 * np.pi)  # EL/h in GHz -> EL/ħ in ns⁻¹ (ADR-0071).
        * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
        / (1 + np.exp(-therm_ratio))
    )


def calc_ind_dipole(
    params: tuple[float, float, float],
    phi_elements: NDArray[np.complex128],
    omegas: NDArray[np.float64],
    Temp: float,
) -> NDArray[np.float64]:
    """Calculate the dipole of the inductive operator.

    params: (EJ, EC, EL) in GHz, omegas: rad/ns, Temp: K -> unit-Q rate in ns⁻¹.
    """
    _EJ, _EC, EL = params

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
    T1errs: NDArray[np.float64] | None = None,
    Temp: float = 20e-3,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """params: GHz, omegas: rad/ns, T1s: ns, guess_Temp: K -> 1"""
    _EJ, _EC, _EL = params

    # calculate Qind vs omega
    dipoles = calc_ind_dipole(params, phi_elements, omegas, Temp)
    Qind_vs_omega = T1s * dipoles

    if T1errs is not None:
        Qind_vs_omega_err = T1errs * dipoles

        return Qind_vs_omega, Qind_vs_omega_err
    return Qind_vs_omega
