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
    phase_rad: finite additive receiver phase in radians after demodulation
        and conversion to microwave I+iQ by conjugating the physical field.
    time_of_flight_ns: finite nonnegative total line delay in ns, default zero.
        Delays the physical output before demodulation and calibration, including
        ringup/ringdown, and gives microwave exp(-2πi*f*delay) at absolute carrier f.
        Offset and noise remain present before the delayed signal arrives.
    ripple_depth: finite reflected amplitude in [0,1), default zero.
    ripple_period_ghz: finite positive standing-wave period in GHz, default 1.
    ripple_phase_rad: finite standing-wave phase in radians, default zero.
    slope_db_per_ghz: finite amplitude slope in dB/GHz, default zero.
    pivot_ghz: finite absolute GHz where the slope factor is unity, default zero.
        background(f) multiplies microwave IQ per incident carrier before
        gain/phase, without changing cavity dynamics, offset or noise.
    offset: owned read-only (2,) [I,Q] ADC sample units, added once per sample.
    noise_covariance: owned read-only symmetric positive-semidefinite (2,2) I/Q
        covariance in ADC sample units squared, per sample. Degenerate and zero
        matrices are valid. Samples/shots are independent; noise is added after
        gain/phase/offset, not scaled by gain. PSD means positive semidefinite,
        not power spectral density. Symmetry/PSD allow only float64 roundoff
        relative to the largest absolute matrix entry. Invalid input raises
        ValueError, including non-PSD covariances at arbitrarily small scales.
    """

    gain: float = 1.0
    phase_rad: float = 0.0
    offset: RealArray = field(default_factory=lambda: np.zeros(2))
    noise_covariance: RealArray = field(default_factory=lambda: np.zeros((2, 2)))
    time_of_flight_ns: float = 0.0
    ripple_depth: float = 0.0
    ripple_period_ghz: float = 1.0
    ripple_phase_rad: float = 0.0
    slope_db_per_ghz: float = 0.0
    pivot_ghz: float = 0.0

    def background(self, frequency_ghz: float) -> complex:
        """Return the complex microwave-IQ background at an absolute carrier GHz.

        B(f) = 10**(slope_db_per_ghz*(f-pivot_ghz)/20)
               * (1 + ripple_depth*exp(i*(2*pi*f/ripple_period_ghz
                                         + ripple_phase_rad))).
        The slope term, not the complete ripple, is unity at the pivot.
        Defaults give B(f)=1. Nonfinite frequency raises ValueError.
        """
        if not np.isfinite(frequency_ghz):
            raise ValueError("background frequency_ghz must be finite")
        slope = 10 ** (self.slope_db_per_ghz * (frequency_ghz - self.pivot_ghz) / 20)
        ripple = 1 + self.ripple_depth * np.exp(
            1j
            * (
                2 * np.pi * frequency_ghz / self.ripple_period_ghz
                + self.ripple_phase_rad
            )
        )
        return complex(slope * ripple)

    def __post_init__(self) -> None:
        for name in (
            "ripple_depth",
            "ripple_period_ghz",
            "ripple_phase_rad",
            "slope_db_per_ghz",
            "pivot_ghz",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.ripple_depth < 1:
            raise ValueError("ripple_depth must be in [0, 1)")
        if self.ripple_period_ghz <= 0:
            raise ValueError("ripple_period_ghz must be positive")
        if not np.isfinite(self.time_of_flight_ns) or self.time_of_flight_ns < 0:
            raise ValueError("time_of_flight_ns must be finite and nonnegative")
        offset = readonly_array(self.offset, "receiver offset")
        covariance = readonly_array(self.noise_covariance, "noise covariance")
        if not np.isfinite([self.gain, self.phase_rad]).all() or offset.shape != (2,):
            raise ValueError("receiver gain/phase must be finite and offset shape (2,)")
        if covariance.shape != (2, 2):
            raise ValueError("noise covariance must be symmetric shape (2,2)")
        scale = float(np.max(np.abs(covariance)))
        if scale:
            # Normalize first so roundoff tolerance follows even tiny covariances.
            normalized = covariance / scale
            tolerance = 8 * np.finfo(np.float64).eps
            if not np.allclose(normalized, normalized.T, rtol=0, atol=tolerance):
                raise ValueError("noise covariance must be symmetric shape (2,2)")
            if np.linalg.eigvalsh(normalized).min() < -tolerance:
                raise ValueError("noise covariance must be positive semidefinite")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "noise_covariance", covariance)


@dataclass(frozen=True)
class ReadoutModel:
    """One or two reciprocal co-located resonators on one two-port line.

    identity: nonempty reproducible numeric model/version identifier.
    state_labels: unique joint-state names in row order of frequencies/rates;
        include retained coupler/leakage states, not only a two-state reduction.
    frequencies_ghz: owned finite (state,cavity) absolute resonant frequencies
        in GHz, with cavity count 1 or 2. Other cavity counts raise ValueError.
    reference_frequency_ghz: finite rotating-frame frequency in GHz; all drive,
        resonator and demod frequencies are compared explicitly against it.
    kappa_internal_per_ns/kappa_external_per_ns: owned nonnegative (cavity,) photon
        decay rates in 1/ns, not angular-frequency GHz. v=sqrt(kappa_external/2).
        Field generator A_s=diag(kappa_internal/2+2πi*(f_s-f_ref))+v v.T.
        Off-diagonal radiative damping is essential to the shared line model.
    rates_per_ns: owned nonnegative (state,state) directional transition rates
        in 1/ns, indexed [target,source], with zero diagonal. The tool constructs
        probability generator Q=R-diag(sum(R,axis=0)); up/down/direct/cascade rates
        are independent, not equilibrium rates interpreted as one-way decay.
    receiver: single output delay/calibration/noise model. No QICK conversion.
    mismatch_phase_rad: owned finite (cavity,) scattering phases in radians.
        The default zeros(2) preserves two-cavity callers. One-cavity callers
        must explicitly supply a one-element array, including zeros(1).
        Output is b_in-sum(v_r*exp(-i*phi_r)*alpha_r) before conjugation.
        Each phase rotates only its resonator's scattered field, with +phi in
        microwave IQ as in the hanger model. It does not change A_s or rates.
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
    mismatch_phase_rad: RealArray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self) -> None:
        n = len(self.state_labels)
        if not self.identity or n == 0 or len(set(self.state_labels)) != n:
            raise ValueError("readout identity and unique state labels required")
        if not np.isfinite(self.reference_frequency_ghz):
            raise ValueError("reference frequency must be finite")
        frequencies = readonly_array(self.frequencies_ghz, "frequencies_ghz")
        if frequencies.ndim != 2 or frequencies.shape[1] not in (1, 2):
            raise ValueError(
                "frequencies_ghz must have shape (state, cavity), with 1 or 2 cavities"
            )
        cavities = frequencies.shape[1]
        specs = (
            ("frequencies_ghz", (n, cavities), False),
            ("kappa_internal_per_ns", (cavities,), True),
            ("kappa_external_per_ns", (cavities,), True),
            ("rates_per_ns", (n, n), True),
            ("mismatch_phase_rad", (cavities,), False),
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

    @property
    def cavity_count(self) -> int:
        """Number of physical resonators (1 or 2), independent of receiver I/Q."""
        return self.frequencies_ghz.shape[1]


def readout_model_key(model: ReadoutModel) -> tuple[object, ...]:
    """Return a lossless hashable key of validated model and receiver content.

    Includes identity, state order, all physical arrays, calibration and noise.
    The key owns bytes rather than array views; no model reference is retained.
    Model identity alone may name many DC points and is not sufficient.
    """
    receiver = model.receiver
    return (
        model.identity,
        model.state_labels,
        model.reference_frequency_ghz,
        model.frequencies_ghz.tobytes(),
        model.kappa_internal_per_ns.tobytes(),
        model.kappa_external_per_ns.tobytes(),
        model.rates_per_ns.tobytes(),
        model.mismatch_phase_rad.tobytes(),
        receiver.time_of_flight_ns,
        receiver.gain,
        receiver.phase_rad,
        receiver.ripple_depth,
        receiver.ripple_period_ghz,
        receiver.ripple_phase_rad,
        receiver.slope_db_per_ghz,
        receiver.pivot_ghz,
        receiver.offset.tobytes(),
        receiver.noise_covariance.tobytes(),
    )
