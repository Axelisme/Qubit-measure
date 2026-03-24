from __future__ import annotations

from typing import Optional, Union, overload

import numpy as np
import scipy.constants as sp
import scipy.special as sp_special
from numpy.typing import NDArray

from zcu_tools.simulate.fluxonium.matrix_element import calculate_sin_phi_oper

from .utils import calc_therm_ratio, convert_eV_to_Hz

# Default superconducting gap for aluminum: 3.4e-4 eV ≈ 82.2 GHz
DELTA_ALUMINUM: float = 3.4e-4


@overload
def qp_spectral_density(
    omega: float, Temp: float, EJ: float, Delta_eV: float = DELTA_ALUMINUM
) -> float: ...


@overload
def qp_spectral_density(
    omega: NDArray[np.float64], Temp: float, EJ: float, Delta_eV: float = DELTA_ALUMINUM
) -> NDArray[np.float64]: ...


def qp_spectral_density(
    omega: Union[float, NDArray[np.float64]],
    Temp: float,
    EJ: float,
    Delta_eV: float = DELTA_ALUMINUM,
) -> Union[float, NDArray[np.float64]]:
    """Quasiparticle tunneling spectral density factor (without x_qp).

    Based on Smith et al (2020), Eq. S23 and Eq. 19.

    omega: rad/ns, EJ: GHz, Delta: GHz, Temp: K -> GHz
    """
    therm_ratio = calc_therm_ratio(omega, Temp)
    abs_therm = calc_therm_ratio(abs(omega), Temp)

    omega_Hz = np.abs(1e9 * omega) / (2 * np.pi)
    EJ_Hz = EJ * 1e9
    Delta_Hz = convert_eV_to_Hz(Delta_eV)

    R_k = sp.h / sp.e**2

    # Re(Y_qp) / x_qp [Siemens], Eq. S23 of Smith et al (2020)
    Y_qp_norm = (
        np.sqrt(2 / np.pi)
        * (8 / R_k)
        * (EJ_Hz / Delta_Hz)
        * (2 * Delta_Hz / abs(omega_Hz)) ** (3 / 2)
        * np.sqrt(1 / 2 * abs_therm)
        * sp_special.kv(0, 1 / 2 * abs(abs_therm))
        * np.sinh(1 / 2 * abs_therm)
    )

    return (
        2e-9
        * sp.hbar
        * omega_Hz
        / sp.e**2
        * Y_qp_norm
        * (1 / np.tanh(0.5 * therm_ratio))
        / (1 + np.exp(-therm_ratio))
    )


def calc_qp_oper(
    params: tuple[float, float, float],
    flux: float,
    return_dim: int = 4,
    esys: Optional[tuple[NDArray[np.float64], NDArray[np.complex128]]] = None,
) -> NDArray[np.complex128]:
    """Calculate the sin(phi/2) operator matrix elements."""
    # In some literature the operator sin(phi/2) is used, which assumes
    # that the flux is grouped with the inductive term in the Hamiltonian.
    # Here we assume a grouping with the cosine term, which requires us to
    # transform the operator using phi -> phi + 2*pi*flux
    return calculate_sin_phi_oper(
        params, flux, return_dim, alpha=0.5, beta=0.5 * (2 * np.pi * flux), esys=esys
    )


def calc_qp_dipole(
    params: tuple[float, float, float],
    sin2_elements: NDArray[np.complex128],
    omegas: NDArray[np.float64],
    Temp: float,
    Delta_eV: float = DELTA_ALUMINUM,
) -> NDArray[np.float64]:
    """Calculate the dipole of the quasiparticle operator.

    params: GHz, omega: rad/ns, T: K, Delta: GHz -> GHz
    """
    EJ, EC, EL = params

    Xqp_factors = qp_spectral_density(omegas, Temp, EJ, Delta_eV) + qp_spectral_density(
        -omegas, Temp, EJ, Delta_eV
    )

    return np.abs(sin2_elements[..., 0, 1]) ** 2 * Xqp_factors


@overload
def calc_Qqp_vs_omega(
    params: tuple[float, float, float],
    omegas: NDArray[np.float64],
    T1s: NDArray[np.float64],
    sin2_elements: NDArray[np.complex128],
    T1errs: None,
    Temp: float = 20e-3,
    Delta_eV: float = DELTA_ALUMINUM,
) -> NDArray[np.float64]: ...


@overload
def calc_Qqp_vs_omega(
    params: tuple[float, float, float],
    omegas: NDArray[np.float64],
    T1s: NDArray[np.float64],
    sin2_elements: NDArray[np.complex128],
    T1errs: NDArray[np.float64],
    Temp: float = 20e-3,
    Delta_eV: float = DELTA_ALUMINUM,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def calc_Qqp_vs_omega(
    params: tuple[float, float, float],
    omegas: NDArray[np.float64],
    T1s: NDArray[np.float64],
    sin2_elements: NDArray[np.complex128],
    T1errs: Optional[NDArray[np.float64]] = None,
    Temp: float = 20e-3,
    Delta_eV: float = DELTA_ALUMINUM,
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Extract Q_qp = 1/x_qp from T1 data via quasiparticle spectral density.

    omegas: rad/ns, T1s: ns, Temp: K, Delta: GHz (superconducting gap)
    sin2_elements: <0|sin(phi/2)|1> matrix elements
    """
    EJ, EC, EL = params

    dipoles = calc_qp_dipole(params, sin2_elements, omegas, Temp, Delta_eV)
    Qqp_vs_omega = T1s * dipoles

    if T1errs is not None:
        Qqp_vs_omega_err = T1errs * dipoles

        return Qqp_vs_omega, Qqp_vs_omega_err
    else:
        return Qqp_vs_omega
