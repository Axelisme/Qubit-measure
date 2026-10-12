"""overnight analysis helpers for lab workflows."""

import numpy as np
from zcu_tools.analysis.fitting import compute_fit_quality, fit_decay

from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoResult, t2echo_signal2real
from zcu_lab.workflows.common.analysis import DecayFit


def analyze_t2echo(result: T2EchoResult) -> DecayFit:
    """Fit zero-detune Echo decay to summarize lifetime drift with one exponential.

    Supply matching 1-D float64 times in microseconds and complex128 IQ.
    true_activate_detune must explicitly be 0.0, not None or a fringe value.
    Exclude the first sample before projection/fitting, as native analysis
    does. The remaining trace needs at least four finite samples with
    strictly increasing times; invalid inputs raise ValueError.
    Return lifetime, marginal uncertainty and residual curves aligned with
    result.times[1:]. Do not mutate result or acquire, plot, save, or apply
    an acceptance threshold. Native errors and optimizer warnings propagate;
    native nonfinite estimates and unavailable diagnostics remain visible.
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
    if not np.isfinite(times).all() or not np.isfinite(signals).all():
        raise ValueError("T2echo retained samples must be finite")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("T2echo retained times must be strictly increasing")
    observed = t2echo_signal2real(signals)
    lifetime, error, fitted, _optimizer = fit_decay(times, observed)
    quality = compute_fit_quality(observed, fitted, {}, None)
    return DecayFit(float(lifetime), float(error), observed, fitted, quality)
