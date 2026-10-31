"""Count-weighted single and shared fringe fits for idle conditional Ramsey."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_decay_fringe
from zcu_tools.analysis.fitting.shared import FitTrace, ParameterSpec, fit_shared


@dataclass(frozen=True)
class IdleBranchFit:
    """One spectator state at one DC coordinate.

    dc_index/branch select DC and state 0/1. Ordinary state means nominal I/Xπ;
    joint fits use actual spectator g/e. acquired_shots
    counts accepted shots over all delays; complete requires all expected rounds
    before herald selection. trace is the T-point target (N_e-N_g)/N_total,
    retaining NaN at unmeasured or zero-acceptance points. fit_curve is the decay-cosine
    model or None when unavailable. magnitude_mhz is nonnegative; signed_mhz
    assumes both branches stay on the side selected by the operator's detune.
    error_mhz is the fitted frequency's covariance 1σ error. resolution_mhz is
    max(1/actual delay span in µs, error_mhz). reason explains why signed_mhz is
    None. fit_quality reports diagnostics even for a sign-ambiguous fitted curve.
    No automatic lack-of-fit/statistical threshold rejects finite fits. Operators
    judge single-frequency model applicability from residuals and uncertainty;
    mixed conditional frequencies can pass. None never means a measured zero.
    Thermal branches are not pure states.
    """

    dc_index: int
    branch: int
    acquired_shots: int
    complete: bool
    trace: NDArray[np.float64]
    fit_curve: NDArray[np.float64] | None = None
    magnitude_mhz: float | None = None
    signed_mhz: float | None = None
    error_mhz: float | None = None
    resolution_mhz: float | None = None
    reason: str | None = None
    fit_quality: FitQuality | None = None


@dataclass(frozen=True)
class IdleZZAnalysis:
    """Ordered (DC-major, I then Xπ) fits and one conditional difference per DC.

    zeta_mhz is f_Xπ−f_I using detune-signed fringe frequencies, never a bare
    magnitude difference. zeta_error_mhz propagates independent branch covariance
    errors in quadrature. Missing/unidentifiable branches leave both None and a
    zeta_reasons explanation. This is a nominal-branch single-model estimate,
    not a purified Hamiltonian ZZ or a writeback/calibration proposal.
    """

    branches: tuple[IdleBranchFit, ...]
    zeta_mhz: tuple[float | None, ...]
    zeta_error_mhz: tuple[float | None, ...]
    zeta_reasons: tuple[str | None, ...]


def complete_points(sums: NDArray[np.float64], expected: int) -> NDArray[np.bool_]:
    """Validate class counts and return full-round points before herald selection.

    sums has DC, spectator, delay then two or four 3-class axes. All-NaN points
    are unmeasured. Finite nonnegative counts may be partial, but cannot exceed
    expected shots. Return the DC × spectator × delay completeness mask.
    """
    nclass = sums.ndim - 3
    if nclass not in (2, 4) or sums.shape[3:] != (3,) * nclass:
        raise ValueError("Idle ZZ requires DC × spectator × delay × 9/81 classes")
    axes = tuple(range(3, sums.ndim))
    finite = np.all(np.isfinite(sums) & (sums >= 0), axis=axes)
    missing = np.all(np.isnan(sums), axis=axes)
    total = sums.sum(axis=axes)
    if not np.all(finite | missing) or np.any(total > expected):
        raise ValueError(
            "Idle ZZ counts must be nonnegative, within requested shots, or all NaN"
        )
    return finite & (total == expected)


def fringe(
    x: NDArray[np.float64],
    frequency: float,
    decay: float,
    offset: float,
    amplitude: float,
    phase: float,
) -> NDArray[np.float64]:
    """Return a decay cosine, with µs, MHz and phase in degrees."""
    return offset + amplitude * np.exp(-x / decay) * np.cos(
        2 * np.pi * frequency * x + np.deg2rad(phase)
    )


def sigma_errors(counts: NDArray[np.float64], target: str) -> NDArray[np.float64]:
    """Return shot-noise errors for (Ne-Ng)/N, including other in N.

    counts is delay × terminal F1 × terminal F2. The multinomial variance of
    the {-1,+1,0} observation is (Pg+Pe-sigma²)/N. A one-shot error floor
    1/N keeps saturated finite-count bins from receiving infinite weight.
    Zero counts give NaN, not an invented observation.
    """
    marginal = counts.sum(axis=-1 if target == "f1" else -2)
    total = marginal.sum(axis=-1)
    value = np.full(total.shape, np.nan)
    second = np.full(total.shape, np.nan)
    np.divide(marginal[:, 1] - marginal[:, 0], total, out=value, where=total > 0)
    np.divide(marginal[:, 1] + marginal[:, 0], total, out=second, where=total > 0)
    variance = np.full(total.shape, np.nan)
    np.divide(np.maximum(second - value**2, 0), total, out=variance, where=total > 0)
    floor = np.full(total.shape, np.nan)
    np.divide(1.0, total, out=floor, where=total > 0)
    return np.maximum(np.sqrt(variance), floor)


def fit_traces(
    fits: tuple[IdleBranchFit, ...],
    errors: tuple[NDArray[np.float64], ...],
    *,
    delays: NDArray[np.float64],
    detune: float,
) -> tuple[IdleBranchFit, ...]:
    """Fit traces of one actual spectator state with shared frequency/decay.

    Each trace keeps its own offset, signed amplitude and phase, so an ee
    fringe can be inverted without changing its physical frequency. fits and
    errors have matching order and contain only one actual spectator state.
    Require complete full delay axes and nonzero accepted counts in each trace.
    Failed or sign-ambiguous fits retain traces and an unavailable reason.
    """
    span = float(np.ptp(delays))
    reason = None
    if not all(f.complete for f in fits):
        reason = "Required branch is incomplete; all delays and rounds are needed"
    elif not all(np.all(np.isfinite(f.trace)) for f in fits):
        reason = "No accepted shots at one or more required delays"
    elif (
        len(delays) < 6
        or span <= 0
        or not np.allclose(np.diff(delays), np.diff(delays)[0])
    ):
        reason = "At least six distinct regularly spaced delays with positive span are needed"
    elif any(np.ptp(f.trace) <= np.finfo(float).eps for f in fits):
        reason = "No identifiable Ramsey contrast"
    if reason:
        return tuple(replace(f, reason=reason) for f in fits)
    try:
        return _fit_complete_traces(fits, errors, delays, detune)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return tuple(replace(f, reason=f"Fringe fit failed: {exc}") for f in fits)


def _fit_complete_traces(
    fits: tuple[IdleBranchFit, ...],
    errors: tuple[NDArray[np.float64], ...],
    delays: NDArray[np.float64],
    detune: float,
) -> tuple[IdleBranchFit, ...]:
    seeds = [fit_decay_fringe(delays, f.trace) for f in fits]
    parameters = [
        ParameterSpec("frequency", abs(float(seeds[0].freq)), limits=(0, None)),
        ParameterSpec("decay", float(seeds[0].decay_time), limits=(1e-9, None)),
    ]
    traces = []
    for index, (fit, error, seed) in enumerate(zip(fits, errors, seeds, strict=True)):
        offset, amplitude, _, phase, _ = seed.params
        names = tuple(f"{name}_{index}" for name in ("offset", "amplitude", "phase"))
        parameters.extend(
            ParameterSpec(name, float(value))
            for name, value in zip(names, (offset, amplitude, phase), strict=True)
        )
        traces.append(
            FitTrace(delays, fit.trace, fringe, ("frequency", "decay", *names), error)
        )
    shared = fit_shared(traces, parameters)
    frequency = shared.values["frequency"]
    variance = shared.variance("frequency")
    error = float(np.sqrt(variance)) if variance >= 0 else np.nan
    resolution = max(1 / float(np.ptp(delays)), error)
    reason = None
    if not shared.diagnostics.valid:
        reason = "Shared fringe optimizer did not converge"
    elif not np.isfinite(error):
        reason = "Frequency uncertainty is not finite; sign is unidentifiable"
    elif detune == 0 or min(frequency, abs(frequency - 2 * abs(detune))) <= resolution:
        reason = "Sign unidentifiable near 0 or 2*abs(detune); possible zero-frequency folding"
    answers = []
    for fit, trace in zip(fits, traces, strict=True):
        values = {name: shared.values[name] for name in trace.parameter_names}
        curve = fringe(delays, *values.values())
        indices = [shared.parameter_names.index(name) for name in trace.parameter_names]
        quality = compute_fit_quality(
            fit.trace, curve, values, shared.covariance[np.ix_(indices, indices)]
        )
        answers.append(
            replace(
                fit,
                fit_curve=curve,
                magnitude_mhz=frequency,
                signed_mhz=None if reason else float(np.sign(detune) * frequency),
                error_mhz=error if np.isfinite(error) else None,
                resolution_mhz=resolution,
                reason=reason,
                fit_quality=quality,
            )
        )
    return tuple(answers)


def summarize(fits: tuple[IdleBranchFit, ...]) -> IdleZZAnalysis:
    """Return signed e-minus-g frequency differences from DC-major g/e pairs.

    Each pair was fitted separately by actual spectator state, so independent
    frequency variances add. Unavailable frequencies retain their reasons.
    """
    zetas: list[float | None] = []
    errors: list[float | None] = []
    reasons: list[str | None] = []
    for a, b in zip(fits[::2], fits[1::2], strict=True):
        if (
            a.signed_mhz is None
            or b.signed_mhz is None
            or a.error_mhz is None
            or b.error_mhz is None
        ):
            zetas.append(None)
            errors.append(None)
            reasons.append("; ".join(f.reason for f in (a, b) if f.reason))
        else:
            zetas.append(b.signed_mhz - a.signed_mhz)
            errors.append(float(np.hypot(a.error_mhz, b.error_mhz)))
            reasons.append(None)
    return IdleZZAnalysis(fits, tuple(zetas), tuple(errors), tuple(reasons))


def validate_axes(
    dc_values: NDArray[np.float64],
    spectator_states: NDArray[np.int64],
    delays: NDArray[np.float64],
    sums: NDArray[np.float64],
    expected_dc: NDArray[np.float64],
    ndelay: int,
) -> None:
    """Reject mismatched saved coordinates or class-count dimensions.

    Require saved DC coordinates, integer spectator indices [0,1] and ndelay
    finite distinct nonnegative µs coordinates. Class axes are all length 3.
    """
    if (
        not np.array_equal(dc_values, expected_dc)
        or not np.array_equal(spectator_states, np.arange(2))
        or delays.shape != (ndelay,)
        or not np.all(np.isfinite(delays))
        or np.any(delays < 0)
        or len(np.unique(delays)) != ndelay
        or sums.shape[:3] != (len(expected_dc), 2, ndelay)
    ):
        raise ValueError(
            "Idle ZZ requires saved DC × spectator_states [0,1] × distinct delay axes"
        )
