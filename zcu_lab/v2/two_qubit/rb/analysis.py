"""Joint00 count ratios and d=4 interpretation of paired RB fits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import (
    FitQuality,
    IRBAnalysis,
    fit_joint_decay,
    seed_statistics,
)
from zcu_tools.analysis.fitting.shared import SharedFitResult
from zcu_tools.cfg_model import ConfigBase


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
    """Fixed-size d=4 paired RB estimate and reference/interleaved fit quality.

    estimate is None with reason when no fit is identifiable. fit_quality holds
    residual diagnostics for each arm, or None without a fit. Survival arrays,
    round completeness and model curves are used only inside analyze.
    """

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


def estimate_joint_d4(
    fit: SharedFitResult,
    interval: tuple[float, float, int],
    paired_counts: tuple[int, int],
) -> RBEstimate:
    """Convert joint arm decays and a seed-bootstrap interval to d=4 estimates."""
    pr, pi = fit.values["p_0"], fit.values["p_1"]
    error = 0.75 * (1 - pi / pr)
    low, high, successes = interval
    warning = (
        "Statistical CI only; gate-dependent/coherent noise can bias the IRB ratio."
    )
    if error < 0:
        warning += " Interleaved decay exceeds reference: negative error estimate retained, not clipped."
    return RBEstimate(
        pr,
        pi,
        0.75 * (1 - pr),
        error,
        low,
        high,
        paired_counts[0],
        paired_counts[1],
        successes,
        warning,
    )


def count_completion(
    counts: NDArray[np.float64], class_axes: tuple[int, ...]
) -> NDArray[np.bool_]:
    """Identify whole finite slots, rejecting partial/nonfinite/negative counts."""
    acquired = np.asarray(np.isfinite(counts).all(axis=class_axes), dtype=np.bool_)
    missing = np.isnan(counts).all(axis=class_axes)
    if not np.all(acquired | missing) or np.any(counts < 0):
        raise ValueError("RB sums contains invalid counts or a partial round")
    return acquired


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


def bootstrap_joint_error(
    depths: NDArray[np.int64],
    survival: NDArray[np.float64],
    totals: NDArray[np.float64],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, int]:
    """95% d=4 ratio CI from paired seed-identity draws across both heralds.

    Inputs are (herald,depth,seed,arm,round), survival masked to paired
    rounds and totals in shot counts. NaN denotes unavailable survival. Draw from
    their valid identity union; each herald retains only its valid draws,
    including multiplicity. Both arms/depths travel together. Fewer than two
    retained draws in either group fails that resample, never fills missing data.
    """
    available = np.isfinite(survival).all(axis=(1, 3)).any(axis=-1)
    identities = np.flatnonzero(available.any(axis=0))
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(samples):
        draw = rng.choice(identities, size=len(identities), replace=True)
        try:
            statistics = [
                seed_statistics(
                    np.take(survival[h], draw, axis=1),
                    np.take(totals[h], draw, axis=1),
                )
                for h in range(2)
            ]
            means = np.array([item.means for item in statistics])
            uncertainty = np.array([item.errors for item in statistics])
            fit = fit_joint_decay(depths, means, uncertainty).fit
        except (ValueError, RuntimeError):
            continue
        errors.append(0.75 * (1 - fit.values["p_1"] / fit.values["p_0"]))
    if len(errors) < 0.9 * samples:
        raise ValueError(f"Unstable bootstrap: {len(errors)}/{samples} valid fits")
    low, high = np.quantile(errors, [0.025, 0.975])
    return float(low), float(high), len(errors)
