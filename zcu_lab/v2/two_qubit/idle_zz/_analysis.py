"""Count-weighted fringe analysis for nominal idle Ramsey branches."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import FitQuality, fit_weighted_decay_fringe


@dataclass(frozen=True)
class IdleBranchFit:
    """One spectator state at one DC coordinate.

    dc_index/branch selects DC and nominal spectator I/Xπ. acquired_shots
    counts accepted shots across all delays. complete marks branches with
    acquired counts at every delay. trace is the T-point target (N_e-N_g)/N_total,
    retaining NaN at unmeasured or zero-acceptance points. fit_curve evaluates
    the decay-cosine at the measured delays; dense_curve holds the model delay
    grid in µs and its model values for display. Both are None when unavailable. magnitude_mhz is nonnegative; signed_mhz
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
    dense_curve: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None


@dataclass(frozen=True)
class IdleZZAnalysis:
    """DC-major state-0/state-1 fits and one conditional difference per DC.

    States mean nominal spectator I/Xπ. zeta_mhz is f_Xπ−f_I using signed
    frequencies, never a bare magnitude difference. zeta_error_mhz propagates independent branch covariance
    errors in quadrature. Missing/unidentifiable branches leave both None and a
    zeta_reasons explanation. This is a nominal-branch single-model estimate,
    not a purified Hamiltonian ZZ or a writeback/calibration proposal.
    """

    branches: tuple[IdleBranchFit, ...]
    zeta_mhz: tuple[float | None, ...]
    zeta_error_mhz: tuple[float | None, ...]
    zeta_reasons: tuple[str | None, ...]


def complete_points(sums: NDArray[np.float64], expected: int) -> NDArray[np.bool_]:
    """Validate counts and mark points with at least one completed round.

    sums has DC, spectator, delay and two terminal class axes. All-NaN points
    are unmeasured. Finite nonnegative counts cannot exceed expected shots.
    A measured delay with zero accepted shots is complete but has no contrast.
    """
    if sums.shape[3:] != (3, 3):
        raise ValueError("Idle ZZ requires DC × spectator × delay × 9 classes")
    axes = (-2, -1)
    finite = np.all(np.isfinite(sums) & (sums >= 0), axis=axes)
    missing = np.all(np.isnan(sums), axis=axes)
    total = sums.sum(axis=axes)
    if not np.all(finite | missing) or np.any(total > expected):
        raise ValueError(
            "Idle ZZ counts must be nonnegative, within requested shots, or all NaN"
        )
    return finite


def fit_trace(
    fit: IdleBranchFit,
    errors: NDArray[np.float64],
    *,
    delays: NDArray[np.float64],
    detune: float,
) -> IdleBranchFit:
    """Fit one nominal spectator branch; retain observations on failure."""
    span = float(np.ptp(delays))
    reason = None
    if not fit.complete:
        reason = "No completed acquisition at one or more required delays"
    elif not np.all(np.isfinite(fit.trace)):
        reason = "No accepted shots at one or more required delays"
    elif (
        len(delays) < 6
        or span <= 0
        or not np.allclose(np.diff(delays), np.diff(delays)[0])
    ):
        reason = "At least six distinct regularly spaced delays with positive span are needed"
    elif np.ptp(fit.trace) <= np.finfo(float).eps:
        reason = "No identifiable Ramsey contrast"
    if reason:
        return replace(fit, reason=reason)
    try:
        return _fit_complete_trace(fit, errors, delays, detune)
    except (
        AnalysisFailed,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as exc:
        return replace(fit, reason=f"Fringe fit failed: {exc}")


def _fit_complete_trace(
    fit: IdleBranchFit,
    errors: NDArray[np.float64],
    delays: NDArray[np.float64],
    detune: float,
) -> IdleBranchFit:
    weighted = fit_weighted_decay_fringe(delays, (fit.trace,), (errors,))
    (model,) = weighted.traces
    frequency = weighted.frequency
    error = weighted.frequency_se
    resolution = max(1 / float(np.ptp(delays)), error)
    reason = None
    if not weighted.shared.diagnostics.valid:
        reason = "Shared fringe optimizer did not converge"
    elif not np.isfinite(error):
        reason = "Frequency uncertainty is not finite; sign is unidentifiable"
    elif detune == 0 or min(frequency, abs(frequency - 2 * abs(detune))) <= resolution:
        reason = "Sign unidentifiable near 0 or 2*abs(detune); possible zero-frequency folding"
    dense_delays = np.linspace(delays.min(), delays.max(), max(201, len(delays)))
    return replace(
        fit,
        fit_curve=model.curve,
        dense_curve=(dense_delays, model.evaluate(dense_delays)),
        magnitude_mhz=frequency,
        signed_mhz=None if reason else float(np.sign(detune) * frequency),
        error_mhz=error if np.isfinite(error) else None,
        resolution_mhz=resolution,
        reason=reason,
        fit_quality=model.quality,
    )


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
    delay_class_shape: tuple[int, ...],
) -> None:
    """Reject mismatched saved coordinates or class-count dimensions.

    Require saved DC coordinates, integer spectator indices [0,1] and finite
    distinct nonnegative µs delays. delay_class_shape specifies the required
    delay length followed by terminal or herald/terminal 3-class axes.
    """
    if (
        not np.array_equal(dc_values, expected_dc)
        or not np.array_equal(spectator_states, np.arange(2))
        or delays.shape != (delay_class_shape[0],)
        or not np.all(np.isfinite(delays))
        or np.any(delays < 0)
        or len(np.unique(delays)) != delay_class_shape[0]
        or sums.shape != (len(expected_dc), 2, *delay_class_shape)
    ):
        raise ValueError(
            "Idle ZZ requires saved DC × spectator_states [0,1] × distinct delay axes"
        )
