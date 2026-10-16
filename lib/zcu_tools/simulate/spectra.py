"""Shared directional loss spectra in angular GHz, ns and kelvin.

These spectra include the coupling prefactors for dimensionless noise operators.
Positive omega denotes downward energy transfer. Negative omega gives the upward
rate by detailed balance; callers add both directions only for equilibrium T1.
No matrix elements, eigenstate selection or channel admission is performed here.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.constants import e, h, hbar, k
from scipy.special import kve

RealValue = float | NDArray[np.float64]
ComplexValue = complex | RealValue | NDArray[np.complex128]
QualityFactor = RealValue | Callable[[RealValue, float], RealValue]
Impedance = ComplexValue | Callable[[RealValue], ComplexValue]
Admittance = ComplexValue | Callable[[RealValue, float], ComplexValue]

DEFAULT_IMPEDANCE_OHM = 50.0
DEFAULT_MUTUAL_PHI0_PER_A = 400.0
DEFAULT_QP_DENSITY = 3e-6
DEFAULT_GAP_EV = 3.4e-4


def _result(value: RealValue, omega: RealValue) -> RealValue:
    array = np.asarray(value, dtype=np.float64)
    return float(array) if np.ndim(omega) == 0 else array


def _thermal(omega: RealValue, temperature_k: float) -> NDArray[np.float64]:
    magnitude = np.abs(omega)
    if temperature_k == 0:
        return np.asarray(np.where(np.asarray(omega) > 0, 1.0, 0.0), dtype=np.float64)
    ratio = hbar * magnitude * 1e9 / (k * temperature_k)
    downward = -1 / np.expm1(-ratio)
    return np.asarray(
        np.where(np.asarray(omega) > 0, downward, downward * np.exp(-ratio)),
        dtype=np.float64,
    )


def _scaled_bessel_weight(x: RealValue) -> NDArray[np.float64]:
    # sqrt(x)*exp(x)*K0(x) tends to sqrt(pi/2), also beyond scipy's range.
    values = np.asarray(x, dtype=np.float64)
    result = np.empty_like(values)
    large = values > 1e6
    inverse = 1 / values[large]
    result[large] = np.sqrt(np.pi / 2) * (1 - inverse / 8 + 9 * inverse**2 / 128)
    result[~large] = np.sqrt(values[~large]) * kve(0, values[~large])
    return result


def capacitive_spectrum(
    omega: RealValue,
    temperature_k: float,
    ec_ghz: float,
    quality_factor: QualityFactor | None = None,
) -> RealValue:
    """Return directional n-coupled capacitive spectrum in 1/ns.

    omega: nonzero finite signed angular frequency, rad/ns (2*pi*GHz), scalar or
    array. temperature_k: nonnegative finite K; ec_ghz: positive charging energy.
    quality_factor: positive scalar/array Q or callable (omega, temperature_k),
    default 1e6*(2*pi*6/abs(omega))**0.7. Output mirrors omega's scalar/array shape.
    Callers validate model scales/options before evaluating this numeric kernel.
    """
    q = (
        1e6 * (2 * np.pi * 6 / np.abs(omega)) ** 0.7
        if quality_factor is None
        else quality_factor(omega, temperature_k)
        if callable(quality_factor)
        else quality_factor
    )
    return _result(32 * np.pi * ec_ghz / q * _thermal(omega, temperature_k), omega)


def inductive_spectrum(
    omega: RealValue,
    temperature_k: float,
    el_ghz: float,
    quality_factor: QualityFactor | None = None,
) -> RealValue:
    """Return directional phi-coupled fluxonium inductive spectrum in 1/ns.

    omega: nonzero finite signed rad/ns, scalar/array; temperature_k: nonnegative
    finite K; el_ghz: positive inductive energy GHz. quality_factor: positive Q
    scalar/array or callable (omega, temperature_k). None uses the scqubits
    500 MHz reference-Q model, evaluated with scaled Bessel functions and its
    zero-temperature limit. Return shape follows omega. Callers validate scales.
    """
    if quality_factor is None:
        ratio = np.abs(omega) / (2 * np.pi * 0.5)
        if temperature_k == 0:
            q = 500e6 * np.sqrt(ratio)
        else:
            x = hbar * np.abs(omega) * 1e9 / (2 * k * temperature_k)
            reference = hbar * 2 * np.pi * 0.5e9 / (2 * k * temperature_k)
            q = (
                500e6
                * np.sqrt(ratio)
                * _scaled_bessel_weight(reference)
                / _scaled_bessel_weight(x)
                * (-np.expm1(-2 * reference))
                / (-np.expm1(-2 * x))
            )
    else:
        q = (
            quality_factor(omega, temperature_k)
            if callable(quality_factor)
            else quality_factor
        )
    return _result(4 * np.pi * el_ghz / q * _thermal(omega, temperature_k), omega)


def charge_impedance_spectrum(
    omega: RealValue,
    temperature_k: float,
    impedance: Impedance = DEFAULT_IMPEDANCE_OHM,
) -> RealValue:
    """Return directional n-coupled ohmic charge spectrum in 1/ns.

    omega is nonzero finite signed rad/ns, scalar/array; temperature_k is finite
    nonnegative K. impedance is complex ohms or callable (omega), scalar/array;
    only its positive real part enters. Default is 50 ohms. Return shape mirrors
    omega. Caller owns scale/option validation; no matrix element is included.
    """
    z = impedance(omega) if callable(impedance) else impedance
    return _result(
        16
        * np.pi
        * np.abs(omega)
        * np.real(z)
        / (h / e**2)
        * _thermal(omega, temperature_k),
        omega,
    )


def flux_line_spectrum(
    omega: RealValue,
    temperature_k: float,
    mutual_phi0_per_a: float = DEFAULT_MUTUAL_PHI0_PER_A,
    impedance: Impedance = DEFAULT_IMPEDANCE_OHM,
) -> RealValue:
    """Return directional d(H/h)/df-coupled flux-line spectrum in 1/ns.

    omega is nonzero finite signed rad/ns, scalar/array; temperature_k is finite
    nonnegative K. mutual_phi0_per_a is finite signed Phi0/A, default 400.
    impedance is complex ohms or callable (omega), with positive real part,
    default 50. The noise operator is in GHz/reduced-flux, so this spectrum
    includes the GHz-to-Hz squared conversion. Return shape follows omega.
    Caller validates scales/options; this is not a physical applicability check.
    """
    z = impedance(omega) if callable(impedance) else impedance
    return _result(
        2
        * (2 * np.pi) ** 2
        * mutual_phi0_per_a**2
        * np.abs(omega)
        * hbar
        / np.real(z)
        * 1e18
        * _thermal(omega, temperature_k),
        omega,
    )


def quasiparticle_spectrum(
    omega: RealValue,
    temperature_k: float,
    ej_ghz: float,
    density: float = DEFAULT_QP_DENSITY,
    gap_ev: float = DEFAULT_GAP_EV,
    admittance: Admittance | None = None,
) -> RealValue:
    """Return directional fluxonium sin(phi/2)-coupled qp spectrum in 1/ns.

    omega: nonzero finite signed rad/ns, scalar/array. temperature_k: finite
    nonnegative K; ej_ghz: positive Josephson GHz. density: nonnegative x_qp,
    default 3e-6; gap_ev: positive superconducting gap eV, default 3.4e-4.
    admittance: complex value or callable (omega, temperature_k), with positive
    real part; None uses the BCS quasiparticle model. The default model uses a
    scaled Bessel expression and its T=0 limit, avoiding overflow/underflow.
    Output mirrors omega's shape. Caller validates scales/options.
    """
    magnitude = np.abs(omega)
    if admittance is not None:
        y = admittance(omega, temperature_k) if callable(admittance) else admittance
        return _result(
            2 * hbar * magnitude / e**2 * np.real(y) * _thermal(omega, temperature_k),
            omega,
        )
    delta_ghz = gap_ev * e / h / 1e9
    gap_ghz = magnitude / (2 * np.pi)
    factor = (
        np.sqrt(np.pi / 2)
        if temperature_k == 0
        else _scaled_bessel_weight(hbar * magnitude * 1e9 / (2 * k * temperature_k))
    )
    # sinh(x)*K0(x)/(1-exp(-2*x)) = exp(x)*K0(x)/2.
    y_down = (
        np.sqrt(2 / np.pi)
        * 8
        / (h / e**2)
        * ej_ghz
        / delta_ghz
        * (2 * delta_ghz / gap_ghz) ** 1.5
        * density
        * factor
        / 2
    )
    balance = (
        np.where(np.asarray(omega) > 0, 1.0, 0.0)
        if temperature_k == 0
        else np.exp(np.minimum(np.asarray(omega), 0) * hbar * 1e9 / (k * temperature_k))
    )
    return _result(2 * hbar * magnitude / e**2 * y_down * balance, omega)
