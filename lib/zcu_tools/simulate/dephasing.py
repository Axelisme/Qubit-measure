"""Effective exponential Markov dephasing; no relaxation T1/2 is added.

Flux uses the echo characteristic rate, not a full 1/f envelope. Photon noise
uses the full state pull difference and low-occupation residual thermal photons,
not driven coherent occupation. All returned rates are 1/ns.
"""

import numpy as np
from numpy.typing import NDArray

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _channels(values: RealArray, scales: RealArray) -> tuple[RealArray, RealArray]:
    values, scales = np.asarray(values, dtype=float), np.asarray(scales, dtype=float)
    if values.ndim != 2 or scales.shape != (len(values),):
        raise ValueError("dephasing values must be (channel,state), scales (channel,)")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(scales).all()
        or np.any(scales < 0)
    ):
        raise ValueError("dephasing values must be finite and amplitudes nonnegative")
    return values, scales


def flux_dephasing(
    slopes_rad_per_us: RealArray, amplitudes_phi0: RealArray
) -> RealArray:
    """Return (state,state) pure rates from independent loop slope differences.

    slopes_rad_per_us: (loop,state) expectation of d(2*pi*H/h)/df in rad/us.
    amplitudes_phi0: (loop,) nonnegative Aphi, Phi0/sqrt(Hz) at 1 Hz convention.
    Gamma_ij=sum sqrt(ln2)*Aphi*abs(si-sj)/1000, in 1/ns. Zero diagonal.
    Invalid shape/nonfinite values raise ValueError. Inputs remain unmodified.
    """
    slopes, amplitudes = _channels(slopes_rad_per_us, amplitudes_phi0)
    difference = np.abs(slopes[:, :, None] - slopes[:, None, :])
    return (
        np.sum(
            flux_noise_gamma_phi_per_us(amplitudes[:, None, None], difference), axis=0
        )
        / 1000
    )


def photon_dephasing(
    pulls_rad_per_us: RealArray,
    linewidths_rad_per_us: RealArray,
    occupations: RealArray,
) -> RealArray:
    """Return (state,state) residual-photon pure rates in 1/ns.

    pulls_rad_per_us: (resonator,state) full angular pulls, linewidths_rad_per_us:
    positive total angular linewidth (resonator,), occupations: nonnegative nth
    per resonator. Gamma=nth*kappa*delta²/(kappa²+delta²)/1000. No half-chi or
    T1 contribution. Bad dimensions/scales/nonfinite values raise ValueError.
    """
    pulls, nth = _channels(pulls_rad_per_us, occupations)
    kappa = np.asarray(linewidths_rad_per_us, dtype=float)
    if kappa.shape != nth.shape or not np.isfinite(kappa).all() or np.any(kappa <= 0):
        raise ValueError("photon linewidths must be positive finite per resonator")
    delta2 = (pulls[:, :, None] - pulls[:, None, :]) ** 2
    return (
        np.sum(
            thermal_photon_gamma_phi_per_us(
                nth[:, None, None],
                kappa_over_2pi_mhz=kappa[:, None, None] / (2 * np.pi),
                chi_over_2pi_mhz=np.sqrt(delta2) / (2 * np.pi),
            ),
            axis=0,
        )
        / 1000
    )


def dephasing_operators(rates_per_ns: RealArray) -> tuple[ComplexArray, ...]:
    """Construct diagonal Lindblad operators for a CP pure-dephasing matrix.

    rates_per_ns is symmetric (N,N), nonnegative, zero diagonal. The centered
    Gram -J*Gamma*J must be PSD (roundoff tolerance 1e-12 times max rate).
    Eigenvectors give real diagonal operators in the supplied energy-state
    order; units ns**(-1/2). Flux absolute-distance and Lorentzian photon kernels
    satisfy this condition. Invalid kernels raise, rather than repair a density.
    """
    gamma = np.asarray(rates_per_ns, dtype=float)
    n = len(gamma)
    if (
        gamma.shape != (n, n)
        or not np.isfinite(gamma).all()
        or np.any(gamma < 0)
        or np.any(np.diag(gamma) != 0)
    ):
        raise ValueError("pure rates must be nonnegative square with diagonal zero")
    if not np.allclose(gamma, gamma.T, rtol=0, atol=1e-15):
        raise ValueError("pure rates must be symmetric")
    center = np.eye(n) - np.ones((n, n)) / n
    values, vectors = np.linalg.eigh(-center @ gamma @ center)
    tolerance = 1e-12 * max(float(np.max(gamma)), np.finfo(float).tiny)
    if values.min() < -tolerance:
        raise ValueError("pure rates do not define a completely positive generator")
    return tuple(
        np.diag(np.sqrt(value) * vector).astype(complex)
        for value, vector in zip(values, vectors.T, strict=True)
        if value > tolerance
    )


def flux_noise_gamma_phi_per_us(
    A_phi: float | RealArray,
    domega_dflux: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return first-order echo characteristic pure rate in 1/us.

    A_phi is Phi0/sqrt(Hz) at 1 Hz; domega_dflux is rad/us/Phi0. Finite numeric
    inputs broadcast with NumPy rules and return that array shape. Pairwise FTF
    APIs validate nonnegative amplitudes; this scalar formula also serves fitting.
    Does not add relaxation, convert to ns or implement a full 1/f envelope.
    """
    return (
        np.sqrt(np.log(2.0))
        * np.asarray(A_phi, dtype=np.float64)
        * np.abs(np.asarray(domega_dflux, dtype=np.float64))
    )


def thermal_photon_gamma_phi_per_us(
    n_th: float | NDArray[np.float64],
    *,
    kappa_over_2pi_mhz: float | RealArray,
    chi_over_2pi_mhz: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Readout thermal-photon shot-noise dephasing rate in 1/us.

    Inputs use the common ``kappa/2pi`` and ``chi/2pi`` MHz convention. Since
    1 MHz = 1 cycle/us, the angular rates are obtained by multiplying by 2*pi.
    n_th is residual Bose occupation, chi is the full two-state pull difference.
    Numeric inputs broadcast; a scalar returns float, arrays preserve broadcast
    shape. Callers validate nonnegative finite nth and positive linewidth. No
    driven-photon occupation or T1/2 is added.
    """
    kappa = 2.0 * np.pi * np.asarray(kappa_over_2pi_mhz, dtype=np.float64)
    chi = 2.0 * np.pi * np.asarray(chi_over_2pi_mhz, dtype=np.float64)
    gamma_per_photon = kappa * chi**2 / (kappa**2 + chi**2)
    gamma = np.asarray(n_th, dtype=np.float64) * gamma_per_photon
    if np.ndim(gamma) == 0:
        return float(gamma)
    return np.asarray(gamma, dtype=np.float64)
