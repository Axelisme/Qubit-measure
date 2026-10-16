"""Joint-energy directional weak-coupling secular baths, in GHz/ns/K.

Rates are indexed [target, source]. Operators are full physical operators in the
joint energy basis, not normalized 01 matrix elements. Exact degenerate active
transitions are rejected: this secular model does not select a preferred vector
inside a degenerate subspace. No near-degeneracy applicability threshold is used.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.constants import h, k

from .spectra import (
    capacitive_spectrum,
    charge_impedance_spectrum,
    flux_line_spectrum,
    inductive_spectrum,
    quasiparticle_spectrum,
)

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
H_OVER_K = h * 1e9 / k


def _positive(*values: float) -> None:
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("bath scales must be positive finite")


@dataclass(frozen=True)
class CapacitiveLoss:
    """ec_ghz: positive charging energy GHz; quality_factor: positive cap Q.

    Couples to dimensionless n. Constant Q uses the physical LC energy lifetime
    Q/(2*pi*nu), not an equilibrium 01 T1 as a directional transition rate.
    """

    ec_ghz: float
    quality_factor: float

    def __post_init__(self) -> None:
        _positive(self.ec_ghz, self.quality_factor)


@dataclass(frozen=True)
class InductiveLoss:
    """el_ghz: positive inductive energy GHz; quality_factor: positive ind Q.

    Couples to dimensionless fluxonium phase phi. Not a SQUID coupler channel.
    """

    el_ghz: float
    quality_factor: float

    def __post_init__(self) -> None:
        _positive(self.el_ghz, self.quality_factor)


@dataclass(frozen=True)
class QuasiparticleLoss:
    """Fluxonium sin(phi/2+pi*f) bath with stable low-temperature Bessel factor.

    ej_ghz: positive Josephson energy GHz. density: nonnegative quasiparticle
    fraction xqp. gap_ev: positive superconducting gap eV, default 3.4e-4.
    No Fluxonium qp formula is applied to the SQUID coupler.
    """

    ej_ghz: float
    density: float
    gap_ev: float = 3.4e-4

    def __post_init__(self) -> None:
        _positive(self.ej_ghz, self.gap_ev)
        if not np.isfinite(self.density) or self.density < 0:
            raise ValueError("quasiparticle density must be nonnegative finite")


@dataclass(frozen=True)
class ChargeImpedance:
    """resistance_ohm: positive effective real charge impedance; bath operator n.

    Constant ohmic frequency dependence. No arbitrary YAML expression is used.
    """

    resistance_ohm: float

    def __post_init__(self) -> None:
        _positive(self.resistance_ohm)


@dataclass(frozen=True)
class FluxLineLoss:
    """resistance_ohm: positive line impedance; mutual_phi0_per_a: finite mutual.

    Couples to d(H/h)/df in GHz/reduced-flux, including the actual SQUID slope.
    mutual_phi0_per_a is in Phi0/A, may be zero or signed. Ohmic line spectrum.
    """

    resistance_ohm: float
    mutual_phi0_per_a: float

    def __post_init__(self) -> None:
        _positive(self.resistance_ohm)
        if not np.isfinite(self.mutual_phi0_per_a):
            raise ValueError("mutual inductance must be finite")


LossChannel = (
    CapacitiveLoss | InductiveLoss | QuasiparticleLoss | ChargeImpedance | FluxLineLoss
)


def gibbs_weights(energies_ghz: RealArray, *, temperature_k: float) -> RealArray:
    """Return normalized Gibbs weights for finite ordered joint H/h eigenvalues.

    temperature_k must be finite and nonnegative. T=0 mixes all exactly equal
    lowest eigenvalues uniformly; no eigensolver-vector choice selects a state.
    Nonfinite/empty/unordered inputs raise ValueError. Input is not modified.
    """
    energy = np.asarray(energies_ghz, dtype=float)
    if energy.ndim != 1 or not len(energy) or not np.isfinite(energy).all():
        raise ValueError("joint energies must be a finite nonempty vector")
    if (
        np.any(np.diff(energy) < 0)
        or not np.isfinite(temperature_k)
        or temperature_k < 0
    ):
        raise ValueError("ordered energies and nonnegative finite temperature required")
    delta = energy - energy[0]
    weights = (
        (delta == 0).astype(float)
        if temperature_k == 0
        else np.exp(-H_OVER_K * delta / temperature_k)
    )
    return weights / weights.sum()


def _down_spectrum(gap: float, temperature: float, channel: LossChannel) -> float:
    omega = 2 * np.pi * gap
    if isinstance(channel, CapacitiveLoss):
        spectrum = capacitive_spectrum(
            omega, temperature, channel.ec_ghz, channel.quality_factor
        )
    elif isinstance(channel, InductiveLoss):
        spectrum = inductive_spectrum(
            omega, temperature, channel.el_ghz, channel.quality_factor
        )
    elif isinstance(channel, ChargeImpedance):
        spectrum = charge_impedance_spectrum(omega, temperature, channel.resistance_ohm)
    elif isinstance(channel, FluxLineLoss):
        spectrum = flux_line_spectrum(
            omega, temperature, channel.mutual_phi0_per_a, channel.resistance_ohm
        )
    else:
        spectrum = quasiparticle_spectrum(
            omega, temperature, channel.ej_ghz, channel.density, channel.gap_ev
        )
    return float(spectrum)


def directional_rates(
    energies_ghz: RealArray,
    matrix_elements: ComplexArray,
    channel: LossChannel,
    *,
    temperature_k: float,
) -> RealArray:
    """Compute all joint multilevel downward/upward rates in 1/ns.

    energies_ghz is ordered (N,), matrix_elements is a finite Hermitian (N,N)
    physical bath operator in that eigensystem. channel names its spectrum and
    units. At finite common temperature up/down=exp(-h*gap/kBT); at T=0 upward
    rates vanish. Diagonal is zero. Active exact degeneracies raise ValueError;
    nonfinite rates fail instead of NaN-to-zero repair. No 01-only reduction.
    """
    gibbs_weights(energies_ghz, temperature_k=temperature_k)
    n = len(energies_ghz)
    op = np.asarray(matrix_elements, dtype=complex)
    if (
        op.shape != (n, n)
        or not np.isfinite(op).all()
        or not np.allclose(op, op.conj().T, atol=1e-12, rtol=0)
    ):
        raise ValueError("bath matrix elements must be finite Hermitian shape (N,N)")
    rates = np.zeros((n, n))
    for source in range(n):
        for target in range(source):
            strength = float(abs(op[target, source]) ** 2)
            if strength == 0:
                continue
            gap = float(energies_ghz[source] - energies_ghz[target])
            if gap == 0:
                raise ValueError("active exactly degenerate secular transition")
            down = strength * _down_spectrum(gap, temperature_k, channel)
            rates[target, source] = down
            rates[source, target] = (
                0
                if temperature_k == 0
                else down * np.exp(-H_OVER_K * gap / temperature_k)
            )
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("bath produced invalid directional rates")
    return rates


def jump_operators(
    rates_per_ns: RealArray, eigenvectors: ComplexArray
) -> tuple[ComplexArray, ...]:
    """Return sqrt(rate)|target><source| in the caller's fixed coordinates.

    rates_per_ns is nonnegative (N,N) with zero diagonal in 1/ns; eigenvectors
    is a unitary (N,N) matrix whose columns are energy states in fixed coordinates.
    Output operators have units ns**(-1/2). Invalid dimensions/values raise.
    """
    rates = np.asarray(rates_per_ns, dtype=float)
    v = np.asarray(eigenvectors, dtype=complex)
    n = len(rates)
    if (
        rates.shape != (n, n)
        or v.shape != (n, n)
        or not np.isfinite(rates).all()
        or np.any(rates < 0)
        or np.any(np.diag(rates) != 0)
    ):
        raise ValueError(
            "directional jump rates must be nonnegative square, diagonal zero"
        )
    if not np.allclose(v.conj().T @ v, np.eye(n), atol=1e-10, rtol=0):
        raise ValueError("jump eigenvectors must be unitary")
    return tuple(
        np.sqrt(rates[i, j]) * np.outer(v[:, i], v[:, j].conj())
        for i, j in zip(*np.nonzero(rates), strict=True)
    )
