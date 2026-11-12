"""Paired-round binomial statistics and two-group shared RB decays."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .shared import FitTrace, ParameterSpec, SharedFitResult, fit_shared


@dataclass(frozen=True)
class RBSeedStatistics:
    """Equal-round seed statistics and their equal-seed aggregate.

    All arrays are float64 and unitless; unavailable seed curves are NaN.
    Means preserve seed multiplicity and never use inverse-count weights.
    """

    seed_means: NDArray[np.float64]
    """Round-averaged survival, shape (seed,2,depth), arms reference/interleaved."""
    seed_variances: NDArray[np.float64]
    """Propagated binomial variance of seed_means, same shape, probability²."""
    means: NDArray[np.float64]
    """Equal-seed average survival, shape (2,depth), probability units."""
    errors: NDArray[np.float64]
    """Standard error from summed seed variances / seed_count², shape (2,depth)."""


def seed_statistics(
    survival: NDArray[np.float64],
    totals: NDArray[np.float64],
) -> RBSeedStatistics:
    """Reduce paired rounds to equal-weight seeds, then aggregate the seeds.

    survival is (depth,seed,arm,round), already masked to whole paired rounds.
    Binomial variance propagates through the arithmetic round average. The
    finite-shot variance floor 1/(4*N**2) keeps all-success/all-failure points
    usable without assigning zero error. Counts affect errors, never seed means.
    Arrays are float64 probabilities/counts, shape (depth,seed,2,round).
    A non-two-arm survival or mismatched totals shape raises ValueError.
    A round/seed contributes only when both arms at every depth are finite.
    Unavailable seeds remain NaN in the seed arrays. At least two complete
    seeds are required; otherwise raise ValueError. totals must match survival
    and be finite and positive for contributing entries, or raise ValueError.
    Totals in unavailable paired rounds are ignored. Inputs are not modified.
    """
    if survival.shape[2] != 2 or totals.shape != survival.shape:
        raise ValueError(
            f"Expected survival shape (depth,seed,2,round) and matching totals; "
            f"got survival shape {survival.shape}, totals shape {totals.shape}"
        )
    complete = np.asarray(np.isfinite(survival).all(axis=(0, 2)), dtype=np.bool_)
    nrounds = complete.sum(axis=1)
    valid = complete[None, :, None, :]
    if np.any(valid & (~np.isfinite(totals) | (totals <= 0))):
        raise ValueError("Contributing totals must be finite and strictly positive")
    means = np.full(survival.shape[:-1], np.nan)
    variances = np.full_like(means, np.nan)
    variance = np.full_like(survival, np.nan)
    np.divide(
        np.maximum(survival * (1 - survival), 0.25 / np.maximum(totals, 1)),
        totals,
        out=variance,
        where=totals > 0,
    )
    np.divide(
        np.where(valid, survival, 0).sum(axis=-1),
        nrounds[None, :, None],
        out=means,
        where=nrounds[None, :, None] > 0,
    )
    np.divide(
        np.where(valid, variance, 0).sum(axis=-1),
        nrounds[None, :, None] ** 2,
        out=variances,
        where=nrounds[None, :, None] > 0,
    )
    seed_means = means.transpose(1, 2, 0)
    seed_variances = variances.transpose(1, 2, 0)
    aggregate, errors = _aggregate_seeds(seed_means, seed_variances)
    return RBSeedStatistics(seed_means, seed_variances, aggregate, errors)


def _aggregate_seeds(
    means: NDArray[np.float64],
    variances: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Equal-seed mean and propagated binomial standard error, shape (2,depth).

    Missing seeds have wholly NaN curves. Repeated bootstrap identities retain
    their multiplicity. No seed receives an inverse-shot-count weight.
    """
    kept = np.isfinite(means).all(axis=(1, 2))
    count = int(kept.sum())
    if count < 2:
        raise ValueError(
            "Need at least two seeds with complete reference/interleaved pairs per herald"
        )
    return means[kept].mean(axis=0), np.sqrt(variances[kept].sum(axis=0)) / count


@dataclass(frozen=True)
class RBJointDecayFit:
    """Shared arm decays with group-local amplitudes/offsets and model curves."""

    fit: SharedFitResult
    """Global p_0/p_1 and A_<group>_<arm>/B_<group>_<arm>, all unitless.

    covariance follows fit.parameter_names, including cross-group correlations.
    Its least-squares scale is reduced chi-square, not absolute supplied errors.
    """
    curves: NDArray[np.float64]
    """A*p**depth+B, float64 (2,2,depth), group/arm/depth probability axes."""


def _decay_model(
    depth: NDArray[np.float64], p: float, amplitude: float, offset: float
) -> NDArray[np.float64]:
    """Return A*p**depth+B in probability coordinates."""
    return amplitude * np.power(p, depth) + offset


def fit_joint_decay(
    depths: NDArray[np.int64],
    means: NDArray[np.float64],
    errors: NDArray[np.float64],
) -> RBJointDecayFit:
    """Fit two groups sharing one decay per arm, with local amplitude/offset.

    depths is int64 (depth,), in random Clifford counts. means and errors are
    float64 (2,2,depth), axes group/reference-interleaved/depth, in probability
    and one-sigma probability units. Errors must be finite and strictly positive.
    Groups have independent A/B and share p by arm in A*p**depth+B; callers
    supply equal-seed means. Inputs are not modified.
    Measurable contrast in each arm, valid decaying minima and finite covariance
    are required; otherwise raise ValueError. Non-two-group/two-arm means or
    mismatched errors shape raise ValueError. Shared-fit errors propagate.
    """
    if means.shape[:2] != (2, 2) or errors.shape != means.shape:
        raise ValueError(
            f"Expected means shape (2,2,depth) and matching errors; "
            f"got means shape {means.shape}, errors shape {errors.shape}"
        )
    parameters = []
    traces = []
    xs = depths.astype(np.float64)
    for arm in range(2):
        if np.max(np.ptp(means[:, arm], axis=-1)) <= 1e-12:
            raise ValueError("IRB decay has no measurable contrast")
        parameters.append(ParameterSpec(f"p_{arm}", 0.95, limits=(1e-6, 1 - 1e-6)))
        for group in range(2):
            curve = means[group, arm]
            amplitude = float(curve[0] - curve[-1])
            names = (f"p_{arm}", f"A_{group}_{arm}", f"B_{group}_{arm}")
            parameters.extend(
                (
                    ParameterSpec(names[1], amplitude, limits=(-1.0, 1.0)),
                    ParameterSpec(names[2], float(curve[-1]), limits=(0.0, 1.0)),
                )
            )
            traces.append(FitTrace(xs, curve, _decay_model, names, errors[group, arm]))
    fit = fit_shared(traces, parameters)
    if (
        not fit.diagnostics.valid
        or not np.isfinite(fit.covariance).all()
        or fit.diagnostics.hesse_failed
    ):
        raise ValueError("IRB joint decay fit is unidentifiable or non-decaying")
    curves = np.empty_like(means)
    for group in range(2):
        for arm in range(2):
            curves[group, arm] = _decay_model(
                xs,
                fit.values[f"p_{arm}"],
                fit.values[f"A_{group}_{arm}"],
                fit.values[f"B_{group}_{arm}"],
            )
    return RBJointDecayFit(fit, curves)
