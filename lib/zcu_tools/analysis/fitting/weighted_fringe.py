"""Weighted damped cosines with one frequency and independent envelopes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..failure import AnalysisFailed
from .base import decaycos
from .decay import fit_decay_fringe
from .quality import FitQuality, compute_fit_quality, decay_time_bounds
from .shared import FitTrace, ParameterSpec, SharedFitResult, fit_shared


@dataclass(frozen=True)
class WeightedFringeTrace:
    """One trace's model and diagnostics projected from the same shared fit.

    offset and amplitude have observation units. Frequency is cycles per delay
    unit, phase is in degrees and decay_time uses delay units. curve contains
    the prediction at the input delays; quality uses this trace's observations
    and covariance submatrix from the shared fit.
    """

    offset: float
    amplitude: float
    frequency: float
    phase: float
    decay_time: float
    curve: NDArray[np.float64]
    quality: FitQuality

    def evaluate(self, delays: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the fitted model at caller-selected delays in input units."""
        return decaycos(
            delays,
            self.offset,
            self.amplitude,
            self.frequency,
            self.phase,
            self.decay_time,
        )


@dataclass(frozen=True)
class WeightedFringeFit:
    """One frequency and its covariance SE, both in cycles per input delay unit.

    frequency_se is NaN if its variance is not available. shared holds named
    values, full covariance in parameter_names order, and optimizer diagnostics
    (including validity). traces are the projected models in input trace order.
    """

    frequency: float
    frequency_se: float
    shared: SharedFitResult
    traces: tuple[WeightedFringeTrace, ...]


def fit_weighted_decay_fringe(
    delays: NDArray[np.float64],
    observations: Sequence[NDArray[np.float64]],
    sigmas: Sequence[NDArray[np.float64]],
) -> WeightedFringeFit:
    """Fit damped cosines sharing only positive frequency by weighted least squares.

    delays is a common 1-D, finite, nonnegative, regularly spaced float64 grid.
    Provide at least one finite float64 observation vector and one positive
    finite pointwise sigma vector per trace, each matching delays. Observations
    and sigma have the same units. The first trace supplies the frequency seed.

    Return the shared frequency, its covariance SE, optimizer diagnostics, and
    each trace's independent offset, amplitude, phase and decay model. Invalid
    inputs raise ValueError; insufficient data (fewer than six samples), no
    resolved contrast or a failed seed fit raise AnalysisFailed. When the shared
    optimizer returns an invalid solution, retain its models and mark
    shared.diagnostics.valid false; failure to return a minimum raises RuntimeError.
    """
    _validate_inputs(delays, observations, sigmas)
    seeds = [fit_decay_fringe(delays, trace) for trace in observations]
    decay_bounds = decay_time_bounds(delays)
    frequencies = np.fft.fftfreq(len(delays), d=abs(delays[1] - delays[0]))
    frequencies = frequencies[frequencies >= 0.125 / float(np.ptp(delays))]
    frequency_bounds = (float(frequencies.min()), float(frequencies.max()))
    parameters = [
        ParameterSpec("frequency", float(seeds[0].freq), limits=frequency_bounds)
    ]
    traces = []
    for index, (obs, sigma, seed) in enumerate(
        zip(observations, sigmas, seeds, strict=True)
    ):
        offset, amplitude, _, phase, decay = seed.params
        # Retain the original singleton names and parameter order.
        time_name = "decay_time" if index == 0 else f"decay_time_{index}"
        parameters.append(ParameterSpec(time_name, float(decay), limits=decay_bounds))
        names = tuple(f"{name}_{index}" for name in ("offset", "amplitude", "phase"))
        parameters.extend(
            ParameterSpec(name, float(value))
            for name, value in zip(names, (offset, amplitude, phase), strict=True)
        )
        traces.append(
            FitTrace(
                delays,
                obs,
                decaycos,
                (names[0], names[1], "frequency", names[2], time_name),
                sigma,
            )
        )
    shared = fit_shared(traces, parameters)
    variance = shared.variance("frequency")
    return WeightedFringeFit(
        shared.values["frequency"],
        float(np.sqrt(variance)) if variance >= 0 else np.nan,
        shared,
        tuple(_project_trace(trace, shared, decay_bounds) for trace in traces),
    )


def _validate_inputs(
    delays: NDArray[np.float64],
    observations: Sequence[NDArray[np.float64]],
    sigmas: Sequence[NDArray[np.float64]],
) -> None:
    if (
        delays.ndim != 1
        or delays.dtype != np.float64
        or np.any(~np.isfinite(delays) | (delays < 0))
    ):
        raise ValueError("delays must be a finite nonnegative float64 vector")
    if len(observations) == 0 or len(observations) != len(sigmas):
        raise ValueError("provide one or more observations with matching sigma vectors")
    for obs, sigma in zip(observations, sigmas, strict=True):
        if (
            obs.shape != delays.shape
            or obs.dtype != np.float64
            or np.any(~np.isfinite(obs))
        ):
            raise ValueError(
                "observations must be finite float64 vectors matching delays"
            )
        if sigma.shape != delays.shape or np.any(~np.isfinite(sigma) | (sigma <= 0)):
            raise ValueError("sigmas must be positive finite vectors matching delays")
    if len(delays) < 6:
        raise AnalysisFailed(
            "At least six delays are required for a weighted fringe fit"
        )
    steps = np.diff(delays)
    if steps[0] == 0 or not np.allclose(steps, steps[0]):
        raise ValueError("delays must have regularly spaced distinct coordinates")
    if any(np.ptp(obs) <= np.finfo(float).eps for obs in observations):
        raise AnalysisFailed("No identifiable fringe contrast")


def _project_trace(
    trace: FitTrace, shared: SharedFitResult, decay_bounds: tuple[float, float]
) -> WeightedFringeTrace:
    values = tuple(shared.values[name] for name in trace.parameter_names)
    curve = decaycos(trace.x, *values)
    indices = [shared.parameter_names.index(name) for name in trace.parameter_names]
    # FitQuality checks decay bounds under this local parameter name.
    local_names = (*trace.parameter_names[:-1], "decay_time")
    quality = compute_fit_quality(
        trace.y,
        curve,
        dict(zip(local_names, values, strict=True)),
        shared.covariance[np.ix_(indices, indices)],
        decay_bounds=decay_bounds,
    )
    offset, amplitude, frequency, phase, decay = values
    return WeightedFringeTrace(
        offset, amplitude, frequency, phase, decay, curve, quality
    )
