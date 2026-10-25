"""Prepare IQ traces and fit decay/fringe curves with explicit numeric settings."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import (
    FitQuality,
    FringeSamplingQuality,
    assess_fringe_sampling,
    compute_fit_quality,
    decay_time_bounds,
    fit_decay,
    fit_decay_fringe,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support.lifetime_models import LifetimeComparison


@dataclass(frozen=True)
class MuxCoherenceFit:
    """Original tone index and fitted coherence time and fringe rate.

    index is the original full-readout tone position. t2/t2_err are coherence
    time and one-sigma error in us. detune/detune_err are fitted fringe rate/error
    in MHz, zero for decay. quality describes the fitted observations, parameters
    and covariance. fit_fringe is True for a damped cosine, False for a decay.
    sampling_quality describes delay-grid density relative to the fitted fringe
    and caller-supplied known frequency lower bound; it is None for decay.
    Adequate sampling does not exclude unknown higher-frequency aliases.
    comparison is optional diagnostic model records/curves supplied by the
    Echo caller; it never changes the primary scalars or quality. Actual detune
    and source frequency remain in the measurement Result.
    """

    index: int
    t2: float
    t2_err: float
    detune: float
    detune_err: float
    quality: FitQuality
    fit_fringe: bool
    sampling_quality: FringeSamplingQuality | None
    comparison: LifetimeComparison | None = None


@dataclass(frozen=True)
class MuxCoherenceTrace:
    """Finite IQ observations for one original tone after the requested omission.

    index is the original full-readout position. times is the requested delay
    in us, shape (N,). signal is matching complex mean IQ, shape (N,). Both arrays
    contain only acquired points; N may be zero for an incomplete measurement.
    """

    index: int
    times: NDArray[np.float64]
    signal: NDArray[np.complex128]


def prepare_coherence_trace(
    index: int,
    times: NDArray[np.float64],
    signal: NDArray[np.complex128],
    *,
    skip: int,
) -> MuxCoherenceTrace:
    """Omit skip leading positions, then retain finite IQ with its requested delay.

    index is the original tone position. times (us) and signal (mean complex IQ)
    are matching one-dimensional arrays from a validated MuxCoherenceResult.
    skip is a nonnegative point count. No rotation, fitting or mutation occurs.
    """
    # Omit by position before filtering so missing IQ cannot shift the omission.
    signal = signal[skip:]
    finite = np.isfinite(signal)
    return MuxCoherenceTrace(index, times[skip:][finite], signal[finite])


def project_coherence_trace(
    trace: MuxCoherenceTrace, *, minimum_points: int
) -> NDArray[np.float64]:
    """Rotate acquired IQ to its real principal axis after checking point count.

    minimum_points is the fit's required observation count. Return one real
    value per trace time. Raise ValueError for too few acquired observations;
    rotation errors propagate so the caller can report a per-tone fit failure.
    """
    if len(trace.times) < minimum_points:
        raise ValueError("Insufficient acquired points for coherence fit")
    return rotate2real(trace.signal).real


def fit_coherence_trace(
    trace: MuxCoherenceTrace,
    real: NDArray[np.float64],
    *,
    fit_fringe: bool,
    phase: float | None,
    expected_frequency_mhz: float | None,
) -> tuple[MuxCoherenceFit, NDArray[np.float64]]:
    """Fit projected IQ to a damped cosine or exponential and return its curve.

    trace supplies requested us delays and original tone index; real contains
    the matching projected IQ. fit_fringe selects the damped cosine. phase fixes
    its phase in degrees; None fits phase freely. Fixed phase is excluded from
    uncertainty invalid diagnostics, not from reported parameters. Decay ignores
    phase and has no fixed fringe parameters. Return
    T2/error in us, fringe rate/error in MHz and fit quality, plus the predicted
    real IQ at each trace time. expected_frequency_mhz is a finite known alias-risk
    lower bound in MHz, or None when unknown; decay ignores it. Fringe sampling
    quality uses the fitted trace's masked delays, not the full raw delay grid.
    Numeric fit errors propagate; nonfinite estimates
    raise ValueError. Inputs are not changed and no plotting occurs.
    """
    # Fit with the caller's numeric phase choice, never an experiment name.
    if fit_fringe:
        t2, error, detune, detune_error, fitted, (params, covariance) = (
            fit_decay_fringe(
                trace.times,
                real,
                fixedparams=None if phase is None else [None, None, None, phase, None],
            )
        )
    else:
        t2, error, fitted, (params, covariance) = fit_decay(trace.times, real)
        detune, detune_error = 0.0, 0.0
    if not np.all(np.isfinite((t2, error, detune, detune_error))):
        raise ValueError("Coherence fit returned nonfinite estimates")

    # Describe the chosen fit without repeating optimization or changing its data.
    names = (
        ("y0", "yscale", "freq", "phase", "decay_time")
        if fit_fringe
        else ("y0", "yscale", "decay_time")
    )
    quality = compute_fit_quality(
        real,
        fitted,
        {name: float(value) for name, value in zip(names, params, strict=True)},
        covariance,
        decay_bounds=decay_time_bounds(trace.times),
        fixed_parameters=("phase",) if fit_fringe and phase is not None else (),
    )
    return (
        MuxCoherenceFit(
            trace.index,
            float(t2),
            float(error),
            float(detune),
            float(detune_error),
            quality,
            fit_fringe,
            assess_fringe_sampling(trace.times, float(detune), expected_frequency_mhz)
            if fit_fringe
            else None,
        ),
        fitted,
    )
