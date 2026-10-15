"""Pure-numeric shared transmission-line and receiver model."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

RealArray = NDArray[np.float64]


def readonly_array(value: RealArray, name: str) -> RealArray:
    """Own a finite float64 array, preserving shape, and make it read-only."""
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Receiver:
    """Single receiver calibration, applied after physical line response.

    gain: finite real amplitude gain, converting outgoing sqrt(photons/ns) to
        ADC sample units. Zero/signed gain is supported; it does not change drive.
    phase_rad: finite additive receiver phase in radians after demodulation.
    offset: owned read-only (2,) [I,Q] ADC sample units, added once per sample.
    noise_covariance: owned read-only symmetric positive-semidefinite (2,2) I/Q
        covariance in ADC sample units squared, per sample. Degenerate and zero
        matrices are valid. Samples/shots are independent; noise is added after
        gain/phase/offset, not scaled by gain. PSD means positive semidefinite,
        not power spectral density. Invalid input raises ValueError.
    """

    gain: float = 1.0
    phase_rad: float = 0.0
    offset: RealArray = field(default_factory=lambda: np.zeros(2))
    noise_covariance: RealArray = field(default_factory=lambda: np.zeros((2, 2)))

    def __post_init__(self) -> None:
        offset = readonly_array(self.offset, "receiver offset")
        covariance = readonly_array(self.noise_covariance, "noise covariance")
        if not np.isfinite([self.gain, self.phase_rad]).all() or offset.shape != (2,):
            raise ValueError("receiver gain/phase must be finite and offset shape (2,)")
        if covariance.shape != (2, 2) or not np.allclose(
            covariance, covariance.T, rtol=0, atol=1e-12
        ):
            raise ValueError("noise covariance must be symmetric shape (2,2)")
        if np.linalg.eigvalsh(covariance).min() < -1e-12:
            raise ValueError("noise covariance must be positive semidefinite")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "noise_covariance", covariance)


@dataclass(frozen=True)
class ReadoutModel:
    """Two reciprocal co-located resonators on one two-port transmission line.

    identity: nonempty reproducible numeric model/version identifier.
    state_labels: unique joint-state names in row order of frequencies/rates;
        include retained coupler/leakage states, not only a two-state reduction.
    frequencies_ghz: owned finite (state,2) absolute resonant frequencies in GHz.
    reference_frequency_ghz: finite rotating-frame frequency in GHz; all drive,
        resonator and demod frequencies are compared explicitly against it.
    kappa_internal_per_ns/kappa_external_per_ns: owned nonnegative (2,) photon
        decay rates in 1/ns, not angular-frequency GHz. v=sqrt(kappa_external/2).
        Field generator A_s=diag(kappa_internal/2+2πi*(f_s-f_ref))+v v.T.
        Off-diagonal radiative damping is essential to the shared line model.
    rates_per_ns: owned nonnegative (state,state) directional transition rates
        in 1/ns, indexed [target,source], with zero diagonal. The tool constructs
        probability generator Q=R-diag(sum(R,axis=0)); up/down/direct/cascade rates
        are independent, not equilibrium rates interpreted as one-way decay.
    receiver: single output calibration/noise model. No QICK buffer conversion.
    Invalid dimensions, values or rates raise ValueError. Weak dispersive QND
    mixture/Markov first-moment assumptions are not runtime applicability gates.
    """

    identity: str
    state_labels: tuple[str, ...]
    frequencies_ghz: RealArray
    reference_frequency_ghz: float
    kappa_internal_per_ns: RealArray
    kappa_external_per_ns: RealArray
    rates_per_ns: RealArray
    receiver: Receiver = field(default_factory=Receiver)

    def __post_init__(self) -> None:
        n = len(self.state_labels)
        if not self.identity or n == 0 or len(set(self.state_labels)) != n:
            raise ValueError("readout identity and unique state labels required")
        if not np.isfinite(self.reference_frequency_ghz):
            raise ValueError("reference frequency must be finite")
        specs = (
            ("frequencies_ghz", (n, 2), False),
            ("kappa_internal_per_ns", (2,), True),
            ("kappa_external_per_ns", (2,), True),
            ("rates_per_ns", (n, n), True),
        )
        for name, shape, nonnegative in specs:
            array = readonly_array(getattr(self, name), name)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if nonnegative and np.any(array < 0):
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, array)
        if np.any(np.diag(self.rates_per_ns) != 0):
            raise ValueError("directional rates diagonal must be zero")
