"""Numeric FTF parameters, independent of file schemas and instrument units."""

from dataclasses import dataclass

import numpy as np


def _positive(*values: float) -> None:
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("circuit scales must be positive finite")


def _noise(
    q_cap: float | None, q_ind: float | None, density: float, amplitude: float
) -> None:
    for q in (q_cap, q_ind):
        if q is not None:
            _positive(q)
    if not np.isfinite([density, amplitude]).all() or min(density, amplitude) < 0:
        raise ValueError("noise amplitudes must be nonnegative finite")


@dataclass(frozen=True)
class FluxoniumCircuit:
    """One fluxonium's numeric circuit, fixed oscillator native coordinates.

    ej_ghz/ec_ghz/el_ghz: positive energies H/h in GHz. native_cutoff: oscillator
    dimension >=2. levels: retained local levels in [1,native_cutoff]. q_cap,
    q_ind: optional positive constant loss Qs, None disables that channel.
    xqp: nonnegative quasiparticle fraction (gap 3.4e-4 eV). aphi: nonnegative
    Phi0/sqrt(Hz) at 1 Hz, used for effective echo Markov dephasing.
    Invalid values raise ValueError. Truncations are not convergence guarantees.
    """

    ej_ghz: float
    ec_ghz: float
    el_ghz: float
    native_cutoff: int
    levels: int
    q_cap: float | None = None
    q_ind: float | None = None
    xqp: float = 0.0
    aphi: float = 0.0

    def __post_init__(self) -> None:
        _positive(self.ej_ghz, self.ec_ghz, self.el_ghz)
        _noise(self.q_cap, self.q_ind, self.xqp, self.aphi)
        if (
            type(self.native_cutoff) is not int
            or type(self.levels) is not int
            or not 1 <= self.levels <= self.native_cutoff
            or self.native_cutoff < 2
        ):
            raise ValueError("fluxonium levels must fit oscillator native_cutoff >=2")


@dataclass(frozen=True)
class SquidCircuit:
    """Fixed-theta asymmetric SQUID, not flux-shifted instantaneous coordinates.

    ejmax_ghz/ec_ghz: positive H/h energies GHz; asymmetry: d in [0,1].
    charge_cutoff: positive integer ncut, native n=-ncut..ncut. levels: number
    retained in that native space. ng: Cooper-pair charge offset, default zero.
    q_cap: optional positive cap Q. resistance_ohm: optional positive charge-line
    impedance. flux_resistance_ohm and mutual_phi0_per_a must be both present
    to enable a flux line. aphi: effective independent-loop flux noise amplitude.
    No fluxonium inductive or quasiparticle channel is accepted for this circuit.
    """

    ejmax_ghz: float
    ec_ghz: float
    asymmetry: float
    charge_cutoff: int
    levels: int
    ng: float = 0.0
    q_cap: float | None = None
    resistance_ohm: float | None = None
    flux_resistance_ohm: float | None = None
    mutual_phi0_per_a: float | None = None
    aphi: float = 0.0

    def __post_init__(self) -> None:
        _positive(self.ejmax_ghz, self.ec_ghz)
        _noise(self.q_cap, None, 0.0, self.aphi)
        if (
            not np.isfinite([self.asymmetry, self.ng]).all()
            or not 0 <= self.asymmetry <= 1
        ):
            raise ValueError("SQUID d must be in [0,1] and ng finite")
        if (
            type(self.charge_cutoff) is not int
            or type(self.levels) is not int
            or self.charge_cutoff < 1
            or not 1 <= self.levels <= 2 * self.charge_cutoff + 1
        ):
            raise ValueError("SQUID levels must fit charge native space")
        for resistance in (self.resistance_ohm, self.flux_resistance_ohm):
            if resistance is not None:
                _positive(resistance)
        if (self.flux_resistance_ohm is None) != (self.mutual_phi0_per_a is None):
            raise ValueError("flux line requires both impedance and mutual inductance")
        if self.mutual_phi0_per_a is not None and not np.isfinite(
            self.mutual_phi0_per_a
        ):
            raise ValueError("mutual inductance must be finite")


@dataclass(frozen=True)
class JointParameters:
    """F1-C-F2 product and fixed lowest-M coupled evolution truncation.

    f1/coupler/f2: numeric local circuits. g1c_ghz/g2c_ghz/g12_ghz: signed GHz
    coefficients multiplying full dimensionless n_i*n_j. retained_levels: M in
    [1,product(local levels)]. Native/local/global truncations remain user-owned.
    """

    f1: FluxoniumCircuit
    coupler: SquidCircuit
    f2: FluxoniumCircuit
    g1c_ghz: float
    g2c_ghz: float
    g12_ghz: float
    retained_levels: int

    def __post_init__(self) -> None:
        if not np.isfinite([self.g1c_ghz, self.g2c_ghz, self.g12_ghz]).all():
            raise ValueError("joint charge couplings must be finite")
        if (
            type(self.retained_levels) is not int
            or not 1
            <= self.retained_levels
            <= self.f1.levels * self.coupler.levels * self.f2.levels
        ):
            raise ValueError("retained_levels must fit local product space")


@dataclass(frozen=True)
class Resonator:
    """One resonator's weak dispersive numeric parameters.

    frequency_ghz: positive bare GHz. charge_coefficients_ghz: three finite signed
    GHz coefficients multiplying full F1/C/F2 charge (no n01 normalization).
    kappa_external_per_ns/kappa_internal_per_ns: nonnegative photon linewidths
    1/ns, total >0. thermal_occupation: nonnegative residual nth, or None for
    Bose occupation at the common model temperature. Driven photons are separate.
    """

    frequency_ghz: float
    charge_coefficients_ghz: tuple[float, float, float]
    kappa_external_per_ns: float
    kappa_internal_per_ns: float
    thermal_occupation: float | None = None

    def __post_init__(self) -> None:
        _positive(self.frequency_ghz)
        if (
            len(self.charge_coefficients_ghz) != 3
            or not np.isfinite(self.charge_coefficients_ghz).all()
        ):
            raise ValueError("resonator requires three finite full-charge coefficients")
        rates = [self.kappa_external_per_ns, self.kappa_internal_per_ns]
        if not np.isfinite(rates).all() or min(rates) < 0 or sum(rates) <= 0:
            raise ValueError("resonator linewidths must be nonnegative, total positive")
        if self.thermal_occupation is not None and (
            not np.isfinite(self.thermal_occupation) or self.thermal_occupation < 0
        ):
            raise ValueError("residual occupation must be nonnegative finite")
