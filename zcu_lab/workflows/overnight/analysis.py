"""overnight analysis helpers for lab workflows."""

import numpy as np
from zcu_tools.analysis.fitting import compute_fit_quality, decay_time_bounds, fit_decay
from zcu_tools.experiment.workflows import Failed

from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoResult, t2echo_signal2real
from zcu_lab.workflows.common.analysis import (
    DecayFit,
    fit_or_failed,
    validate_fit_trace,
)


def analyze_t2echo(result: T2EchoResult) -> DecayFit | Failed:
    """Fit zero-detune Echo decay to summarize lifetime drift with one exponential.

    Supply matching 1-D float64 times in microseconds and complex128 IQ.
    true_activate_detune must explicitly be 0.0, not None or a fringe value.
    Exclude the first sample before projection/fitting, as native analysis
    does. The remaining trace needs at least four samples with strictly
    increasing finite times; violations raise ValueError.
    Return lifetime, marginal uncertainty and residual curves aligned with
    result.times[1:]. Do not mutate result or acquire, plot, save, or apply
    an acceptance threshold. Nonfinite retained IQ and fit_or_failed's expected
    failures return Failed(reason). Grid/shape/detune errors raise ValueError;
    unexpected errors propagate. Native nonfinite estimates remain visible.
    """
    if result.true_activate_detune != 0.0:
        raise ValueError("T2echo decay analysis requires explicit zero detune")
    if (
        result.times.ndim != 1
        or result.signals.ndim != 1
        or result.times.shape != result.signals.shape
    ):
        raise ValueError("T2echo times and signals must have matching 1-D shape")
    times, signals = result.times[1:], result.signals[1:]
    if len(times) < 4:
        raise ValueError("T2echo needs at least four samples after the first")
    failure = validate_fit_trace(times, signals)
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("T2echo retained times must be strictly increasing")
    if failure is not None:
        return failure
    observed = t2echo_signal2real(signals)
    fit = fit_or_failed(lambda: fit_decay(times, observed))
    if isinstance(fit, Failed):
        return fit
    lifetime, error, fitted, (parameters, covariance) = fit
    quality = compute_fit_quality(
        observed,
        fitted,
        dict(zip(("y0", "yscale", "decay_time"), map(float, parameters), strict=True)),
        covariance,
        decay_bounds=decay_time_bounds(times),
    )
    return DecayFit(float(lifetime), float(error), observed, fitted, quality, times)
