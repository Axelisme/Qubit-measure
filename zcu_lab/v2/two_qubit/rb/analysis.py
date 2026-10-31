"""Joint00 count ratios and d=4 interpretation of paired RB fits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.fitting.shared import (
    FitTrace,
    ParameterSpec,
    SharedFitResult,
    fit_shared,
)
from zcu_tools.cfg_model import ConfigBase

from ...twotone.irb.analysis import IRBAnalysis


class RBOptions(ConfigBase):
    """Analysis-only reproducible paired-seed bootstrap settings.

    bootstrap_samples is at least 100 whole-seed resamples, not independent
    shots. bootstrap_seed is the nonnegative RNG seed for this uncertainty
    calculation, separate from the acquired sequence identities.
    """

    bootstrap_samples: int = Field(default=1000, ge=100)
    bootstrap_seed: int = Field(default=42, ge=0)


@dataclass(frozen=True)
class RBEstimate:
    """Unitless d=4 RB estimates, conditional on the exponential decay model.

    p_reference/p_interleaved are per random Clifford (plus CZ for interleaved).
    reference_epc=3/4*(1-p_reference). gate_error=3/4*(1-p_i/p_r), not clipped.
    error_ci_low/error_ci_high is the 95% paired-seed percentile interval.
    n_paired_seeds/n_paired_rounds count complete curves/round-seed pairs.
    bootstrap_successes counts valid resampled fits; warning describes ratio
    and model limitations. No field asserts calibrated physical fidelity.
    """

    p_reference: float
    p_interleaved: float
    reference_epc: float
    gate_error: float
    error_ci_low: float
    error_ci_high: float
    n_paired_seeds: int
    n_paired_rounds: int
    bootstrap_successes: int
    warning: str


@dataclass(frozen=True)
class RBAnalysis:
    """Survival and fit diagnostics derived only from saved cfg/counts.

    survival has shape (depth,seed,2,round); NaN means unavailable paired round.
    acquired_rounds has that shape and records finite whole rounds in each arm
    before pair selection. mean_survival/fit_survival are (2,depth), with equal
    seed weighting; fit_survival remains NaN when estimate is None. estimate is
    absent on missing cfg/data or an unidentifiable fit. fit_quality is
    reference/interleaved residual diagnostics, or None without a fit. Parameter
    covariance is not exposed by the shared paired owner, so these diagnostics
    have no parameter-error entries. reason explains
    why (None only on success). No missing sample is replaced by measured zero.
    """

    survival: NDArray[np.float64]
    acquired_rounds: NDArray[np.bool_]
    mean_survival: NDArray[np.float64]
    fit_survival: NDArray[np.float64]
    estimate: RBEstimate | None
    fit_quality: tuple[FitQuality, FitQuality] | None
    reason: str | None


def estimate_d4(fit: IRBAnalysis) -> RBEstimate:
    """Convert numerical paired-IRB estimates and interval from d=2 to d=4."""
    # The shared owner fits and resamples paired curves; only d=4 interpretation
    # belongs here. Its d=2 error is affine in the same decay ratio.
    return RBEstimate(
        p_reference=fit.p_reference,
        p_interleaved=fit.p_interleaved,
        reference_epc=0.75 * (1 - fit.p_reference),
        gate_error=0.75 * (1 - fit.p_interleaved / fit.p_reference),
        error_ci_low=1.5 * (1 - fit.fidelity_ci_high),
        error_ci_high=1.5 * (1 - fit.fidelity_ci_low),
        n_paired_seeds=fit.n_paired_seeds,
        n_paired_rounds=fit.n_paired_rounds,
        bootstrap_successes=fit.bootstrap_successes,
        warning=fit.warning,
    )


def paired_means(survival: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average complete rounds inside each seed, then seeds equally, per arm."""
    complete = np.asarray(np.isfinite(survival).all(axis=(0, 2)), dtype=np.bool_)
    counts = complete.sum(axis=1)
    kept = counts > 0
    if not np.any(kept):
        return np.full((2, survival.shape[0]), np.nan)
    seed_means = np.where(complete[None, :, None, :], survival, 0).sum(axis=-1)
    return (seed_means[:, kept, :] / counts[None, kept, None]).mean(axis=1).T


def validate_axes(
    axes: tuple[NDArray[np.int64], ...],
    expected: tuple[Sequence[int] | NDArray[np.int64], ...],
) -> None:
    """Require persisted axis identities to match the saved configuration."""
    if any(
        not np.array_equal(actual, wanted)
        for actual, wanted in zip(axes, expected, strict=True)
    ):
        raise ValueError("RB count axes do not match saved cfg")


def seed_statistics(
    survival: NDArray[np.float64],
    totals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average paired rounds equally inside seeds, returning (seed,arm,depth).

    survival is (depth,seed,arm,round), already masked to whole paired rounds.
    Binomial variance propagates through the arithmetic round average. The
    finite-shot variance floor 1/(4*N**2) keeps all-success/all-failure points
    usable without assigning zero error. Counts affect errors, never seed means.
    Unavailable seeds remain NaN.
    """
    complete = np.isfinite(survival).all(axis=(0, 2))
    nrounds = complete.sum(axis=1)
    valid = complete[None, :, None, :]
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
    return means.transpose(1, 2, 0), variances.transpose(1, 2, 0)


def aggregate_seeds(
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


def decay_model(
    depth: NDArray[np.float64], p: float, amplitude: float, offset: float
) -> NDArray[np.float64]:
    """Return A*p**depth+B in probability coordinates."""
    return amplitude * np.power(p, depth) + offset


def fit_joint_decay(
    depths: NDArray[np.int64],
    means: NDArray[np.float64],
    errors: NDArray[np.float64],
) -> tuple[SharedFitResult, NDArray[np.float64]]:
    """Fit (herald,arm,depth) curves, one decay per arm, own A/B per herald.

    Requires measurable contrast in each arm, valid decaying minima and finite
    covariance. Means have already been averaged with equal seed weights.
    """
    parameters = []
    traces = []
    xs = depths.astype(np.float64)
    for arm in range(2):
        if np.max(np.ptp(means[:, arm], axis=-1)) <= 1e-12:
            raise ValueError("IRB decay has no measurable contrast")
        parameters.append(ParameterSpec(f"p_{arm}", 0.95, limits=(1e-6, 1 - 1e-6)))
        for herald in range(2):
            curve = means[herald, arm]
            amplitude = float(curve[0] - curve[-1])
            names = (f"p_{arm}", f"A_{herald}_{arm}", f"B_{herald}_{arm}")
            parameters.extend(
                (
                    ParameterSpec(names[1], amplitude, limits=(-1.0, 1.0)),
                    ParameterSpec(names[2], float(curve[-1]), limits=(0.0, 1.0)),
                )
            )
            traces.append(FitTrace(xs, curve, decay_model, names, errors[herald, arm]))
    fit = fit_shared(traces, parameters)
    if (
        not fit.diagnostics.valid
        or not np.isfinite(fit.covariance).all()
        or fit.diagnostics.hesse_failed
    ):
        raise ValueError("IRB joint decay fit is unidentifiable or non-decaying")
    curves = np.empty_like(means)
    for herald in range(2):
        for arm in range(2):
            curves[herald, arm] = decay_model(
                xs,
                fit.values[f"p_{arm}"],
                fit.values[f"A_{herald}_{arm}"],
                fit.values[f"B_{herald}_{arm}"],
            )
    return fit, curves


def bootstrap_joint_error(
    depths: NDArray[np.int64],
    seed_means: NDArray[np.float64],
    seed_variances: NDArray[np.float64],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, int]:
    """95% d=4 ratio CI from paired seed-identity draws across both heralds.

    Inputs are (herald,seed,arm,depth), NaN for unavailable seeds. Draw from
    their valid identity union; each herald retains only its valid draws,
    including multiplicity. Both arms/depths travel together. Fewer than two
    retained draws in either group fails that resample, never fills missing data.
    """
    available = np.isfinite(seed_means).all(axis=(2, 3))
    identities = np.flatnonzero(available.any(axis=0))
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(samples):
        draw = rng.choice(identities, size=len(identities), replace=True)
        try:
            statistics = [
                aggregate_seeds(seed_means[h, draw], seed_variances[h, draw])
                for h in range(2)
            ]
            means = np.array([item[0] for item in statistics])
            uncertainty = np.array([item[1] for item in statistics])
            fit, _ = fit_joint_decay(depths, means, uncertainty)
        except (ValueError, RuntimeError):
            continue
        errors.append(0.75 * (1 - fit.values["p_1"] / fit.values["p_0"]))
    if len(errors) < 0.9 * samples:
        raise ValueError(f"Unstable bootstrap: {len(errors)}/{samples} valid fits")
    low, high = np.quantile(errors, [0.025, 0.975])
    return float(low), float(high), len(errors)
