"""Known-model numeric inputs for independent experiment analysis.

All inputs are required. Coordinates and parameters use the same units as the
corresponding fitting model. Seeds initialize a local NumPy generator and never
reseed process-wide state. Returned arrays do not alias caller inputs.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fitting import cosfunc, decaycos, expfunc, lorfunc
from zcu_tools.analysis.fitting.anticross import get_predict_ys
from zcu_tools.analysis.fitting.multi_decay import TransitionRates, model_func
from zcu_tools.analysis.fitting.resonance import HangerModel, TransmissionModel

from .shot_generators import GaussianCloudParameters as GaussianCloudParameters
from .shot_generators import generate_singleshot_ge as generate_singleshot_ge
from .shot_generators import generate_singleshot_rabi as generate_singleshot_rabi


@dataclass(frozen=True)
class DecayParameters:
    """Exponential truth: offset, amplitude in signal units; time in x units."""

    offset: float
    amplitude: float
    decay_time: float


@dataclass(frozen=True)
class OscillationParameters:
    """Cosine truth in fitting coordinates.

    offset/amplitude are in signal units; frequency is cycles per x unit and
    phase_degrees is the phase at x=0. decay_time is positive in x units, or
    None for constant envelope (supported by generate_rabi only).
    """

    offset: float
    amplitude: float
    frequency: float
    phase_degrees: float
    decay_time: float | None


@dataclass(frozen=True)
class LorentzianParameters:
    """Lorentzian truth: offset/amplitude in signal units, slope in signal/x.

    center and positive half_width use x units; offset is the baseline at center.
    """

    offset: float
    slope: float
    amplitude: float
    center: float
    half_width: float


@dataclass(frozen=True)
class ResonatorParameters:
    """Complex resonance truth in fitting's MHz frequency convention.

    frequency is resonance MHz; loaded_quality is positive Ql. coupling_quality
    selects hanger when positive and transmission when None. asymmetry_phase
    is hanger phi in radians (unused for transmission). amplitude is complex
    a0 at resonance, electrical_delay is in inverse MHz. amplitude_slope is
    real log-amplitude slope per MHz; phase_curvature is radians per MHz squared.
    """

    frequency: float
    loaded_quality: float
    coupling_quality: float | None
    asymmetry_phase: float
    amplitude: complex
    electrical_delay: float
    amplitude_slope: float
    phase_curvature: float


@dataclass(frozen=True)
class ConicParameters:
    """Coefficients of a*x² + b*x + c*y² + d*y + e*x*y + f = 0.

    x/y use the supplied coordinate/frequency units. c must be nonzero and
    discriminants must be nonnegative on the requested x grid. Overall scale
    is unidentifiable; fitted physics, not individual coefficients, is truth.
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


@dataclass(frozen=True)
class MultiDecayParameters:
    """Three-state Markov truth at elapsed time zero.

    rates holds nonnegative ge/eg/eo/oe/go/og transition rates; each two-letter
    name means source to destination, in inverse time units. initial_g and
    initial_e are populations in [0,1] whose sum is at most one. The remaining
    population is state o. No readout/confusion matrix is applied.
    """

    rates: TransitionRates
    initial_g: float
    initial_e: float


def _coordinates(xs: NDArray[np.float64]) -> NDArray[np.float64]:
    if xs.ndim != 1 or xs.size == 0 or not np.all(np.isfinite(xs)):
        raise ValueError("coordinates must be a nonempty finite 1D array")
    return xs


def _noise(
    shape: tuple[int, ...], noise_std: float, seed: int
) -> NDArray[np.float64]:
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    return np.random.default_rng(seed).normal(0, noise_std, shape)


def generate_decay(
    xs: NDArray[np.float64], truth: DecayParameters, noise_std: float, seed: int
) -> NDArray[np.float64]:
    """Return float64 exponential samples matching the 1D finite xs array.

    truth defines offset/amplitude and positive decay time in xs units.
    noise_std is independent additive Gaussian signal noise per sample; zero
    disables it. seed initializes only this call's RNG. Invalid coordinates,
    nonpositive time or invalid noise raise ValueError. Inputs are unchanged.
    """
    _coordinates(xs)
    if not np.isfinite(truth.decay_time) or truth.decay_time <= 0:
        raise ValueError("decay_time must be finite and positive")
    return expfunc(xs, truth.offset, truth.amplitude, truth.decay_time) + _noise(
        xs.shape, noise_std, seed
    )


def generate_decay_fringe(
    xs: NDArray[np.float64], truth: OscillationParameters, noise_std: float, seed: int
) -> NDArray[np.float64]:
    """Return a float64 damped cosine matching finite 1D xs.

    truth uses the fitting decaycos convention; decay_time must be positive,
    not None. noise_std is additive independent Gaussian signal noise and seed
    initializes a local RNG. Invalid coordinates, time or noise raise ValueError.
    No inputs are modified.
    """
    _coordinates(xs)
    if (
        truth.decay_time is None
        or not np.isfinite(truth.decay_time)
        or truth.decay_time <= 0
    ):
        raise ValueError("decay_time must be finite and positive")
    return decaycos(
        xs,
        truth.offset,
        truth.amplitude,
        truth.frequency,
        truth.phase_degrees,
        truth.decay_time,
    ) + _noise(xs.shape, noise_std, seed)


def generate_rabi(
    xs: NDArray[np.float64], truth: OscillationParameters, noise_std: float, seed: int
) -> NDArray[np.float64]:
    """Return float64 Rabi samples matching finite 1D xs without changing inputs.

    None decay_time selects fitting.cosfunc, otherwise fitting.decaycos with
    positive time. Frequency is cycles per x unit, phase is in degrees.
    noise_std is independent additive Gaussian signal noise per sample; seed
    initializes a local RNG. Invalid coordinates, time or noise raise ValueError.
    """
    if truth.decay_time is not None:
        return generate_decay_fringe(xs, truth, noise_std, seed)
    _coordinates(xs)
    return cosfunc(
        xs, truth.offset, truth.amplitude, truth.frequency, truth.phase_degrees
    ) + _noise(xs.shape, noise_std, seed)


def generate_lorentzian(
    xs: NDArray[np.float64], truth: LorentzianParameters, noise_std: float, seed: int
) -> NDArray[np.float64]:
    """Return fitting.lorfunc samples on finite 1D xs, without changing inputs.

    truth supplies a peak (positive amplitude) or dip (negative amplitude) with
    linear background and positive half_width in xs units. noise_std is the
    independent additive Gaussian signal noise; seed initializes a local RNG.
    Invalid coordinates, width or noise raise ValueError.
    """
    _coordinates(xs)
    if not np.isfinite(truth.half_width) or truth.half_width <= 0:
        raise ValueError("half_width must be finite and positive")
    return lorfunc(
        xs, truth.offset, truth.slope, truth.amplitude, truth.center, truth.half_width
    ) + _noise(xs.shape, noise_std, seed)


def generate_resonator(
    freqs: NDArray[np.float64], truth: ResonatorParameters, noise_std: float, seed: int
) -> NDArray[np.complex128]:
    """Return complex128 resonator samples matching finite 1D freqs in MHz.

    Select the public HangerModel or TransmissionModel through truth's coupling
    quality. noise_std is independent Gaussian noise in each IQ quadrature;
    seed initializes a local RNG. Nonpositive frequency/quality or invalid
    coordinates/noise raise ValueError. Inputs are unchanged.
    """
    _coordinates(freqs)
    if truth.frequency <= 0 or truth.loaded_quality <= 0:
        raise ValueError("frequency and loaded_quality must be positive")
    if truth.coupling_quality is None:
        signals = TransmissionModel.calc_signals(
            freqs, truth.frequency, truth.loaded_quality, truth.amplitude,
            truth.electrical_delay, truth.amplitude_slope, truth.phase_curvature
        )
    else:
        if truth.coupling_quality <= 0:
            raise ValueError("coupling_quality must be positive or None")
        signals = HangerModel.calc_signals(
            freqs, truth.frequency, truth.loaded_quality, truth.coupling_quality,
            truth.asymmetry_phase, truth.amplitude, truth.electrical_delay,
            truth.amplitude_slope, truth.phase_curvature
        )
    noise = _noise((2, freqs.size), noise_std, seed)
    return signals + noise[0] + 1j * noise[1]


def generate_qubit_frequency(
    freqs: NDArray[np.float64], truth: LorentzianParameters, noise_std: float, seed: int
) -> NDArray[np.float64]:
    """Return float64 Lorentzian qubit spectroscopy samples matching freqs.

    freqs is a finite 1D frequency array. truth.center/half_width use those
    frequency units; full width is twice half_width. truth.slope must be zero
    to match fit_qubit_freq's Lorentzian contract. Either amplitude sign is
    allowed. noise_std is additive independent Gaussian signal noise, and seed
    initializes a local RNG. Invalid slope, width, coordinates or noise raise
    ValueError. Inputs are unchanged.
    """
    if truth.slope != 0:
        raise ValueError("qubit frequency Lorentzian requires zero slope")
    return generate_lorentzian(freqs, truth, noise_std, seed)


def generate_anticrossing(
    xs: NDArray[np.float64], truth: ConicParameters, noise_std: float, seed: int
) -> NDArray[np.float64]:
    """Return float64 branches shaped (2, len(xs)), upper then lower noiseless.

    xs is a finite 1D coordinate array. truth gives the implicit conic in
    coordinate/frequency units. Reuse fitting.anticross.get_predict_ys; c=0 or
    nonreal/nonfinite branches raise ValueError. noise_std is independent
    additive Gaussian frequency noise on each branch, so noisy points may
    cross. seed initializes a local RNG. Inputs are unchanged.
    """
    _coordinates(xs)
    if truth.c == 0:
        raise ValueError("conic c must be nonzero")
    branches = np.stack(get_predict_ys(
        xs, truth.a, truth.b, truth.c, truth.d, truth.e, truth.f
    ))
    if not np.all(np.isfinite(branches)):
        raise ValueError("conic must have finite real branches on xs")
    return branches + _noise(branches.shape, noise_std, seed)


def generate_multi_decay(
    times: NDArray[np.float64], truth: MultiDecayParameters, noise_std: float, seed: int
) -> NDArray[np.float64]:
    """Return (len(times), 3) float64 g/e/o populations from model_func.

    times contains finite nonnegative elapsed times since preparation, not
    absolute timestamps. truth gives six rates in inverse time units and
    initial g/e populations; o is their complement. noise_std is independent
    additive Gaussian population measurement noise; no clipping or
    renormalization is applied, so noisy rows need not sum to one. seed
    initializes a local RNG. Invalid coordinates, rates, populations or noise
    raise ValueError. Inputs are unchanged.
    """
    _coordinates(times)
    rates = truth.rates.as_array()
    if np.any(times < 0) or not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("times and rates must be finite and nonnegative")
    if not (0 <= truth.initial_g <= 1 and 0 <= truth.initial_e <= 1
            and truth.initial_g + truth.initial_e <= 1):
        raise ValueError("initial populations must lie in the probability simplex")
    populations = model_func(
        times, *truth.rates.as_tuple(), truth.initial_g, truth.initial_e
    )
    return populations + _noise(populations.shape, noise_std, seed)
