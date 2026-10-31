"""Joint00 count ratios and d=4 interpretation of paired RB fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots

from ...twotone.irb.analysis import IRBAnalysis, analyze_irb
from .._sums import joint_probabilities

if TYPE_CHECKING:
    from .core import RBCfg, RBResult


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


def _estimate(fit: IRBAnalysis) -> RBEstimate:
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


def _paired_means(survival: NDArray[np.float64]) -> NDArray[np.float64]:
    complete = np.asarray(np.isfinite(survival).all(axis=(0, 2)), dtype=np.bool_)
    counts = complete.sum(axis=1)
    kept = counts > 0
    if not np.any(kept):
        return np.full((2, survival.shape[0]), np.nan)
    seed_means = np.where(complete[None, :, None, :], survival, 0).sum(axis=-1)
    return (seed_means[:, kept, :] / counts[None, kept, None]).mean(axis=1).T


def analyze_rb(
    source: RunRecord[RBCfg, RBResult], options: RBOptions, *, plots: Plots
) -> RBAnalysis:
    """Fit complete paired rounds from (depth,seed,arm,round,3,3) counts.

    Joint00 survival is N_gg/N_total, including other in the denominator.
    Both arms across every depth must be complete for a (seed,round) to enter
    fitting. Average complete rounds within each seed, then weight seeds equally.
    Zero-total rounds have NaN survival and cannot enter paired fits.

    Reuse analyze_irb's unchanged exponential fit and whole paired-seed bootstrap
    interface. Apply d=4 only here. Restore the owner's possible real-axis sign
    reversal for probability-space plots. Negative error is not clipped.
    Missing cfg or fit ValueError/RuntimeError yields a reason without estimate.
    Malformed axes, partially populated or nonfinite acquired counts raise.
    Saved counts suffice; no reclassification, live library or truth is used.
    """
    data, cfg = source.result, source.cfg
    shape = (len(data.depths), len(data.seeds), 2, len(data.rounds))
    survival = np.full(shape, np.nan)
    acquired = np.zeros(shape, dtype=np.bool_)
    means = np.full((2, len(data.depths)), np.nan)
    curves = means.copy()
    estimate = None
    quality = None
    reason = None
    if cfg is None:
        reason = "RB analysis requires saved cfg; current Form is not a substitute"
    else:
        _validate_axes(source)
        counts = data.sums
        acquired = np.all(np.isfinite(counts), axis=(-2, -1))
        missing = np.all(np.isnan(counts), axis=(-2, -1))
        if not np.all(acquired | missing):
            raise ValueError("RB sums contains nonfinite counts or a partial round")
        joint00 = joint_probabilities(counts)[..., 0, 0]
        paired = (acquired & np.isfinite(joint00)).all(axis=(0, 2))
        survival = np.where(paired[None, :, None, :], joint00, np.nan)
        means = _paired_means(survival)
        try:
            fit = analyze_irb(
                data.depths,
                survival.transpose(3, 1, 2, 0).astype(np.complex128),
                bootstrap_samples=options.bootstrap_samples,
                bootstrap_seed=options.bootstrap_seed,
            )
        except (ValueError, RuntimeError) as exc:
            # Isolate model failure while retaining measured count diagnostics.
            reason = str(exc)
        else:
            estimate = _estimate(fit)
            quality = (
                compute_fit_quality(fit.mean_signals[0], fit.fit_signals[0], {}, None),
                compute_fit_quality(fit.mean_signals[1], fit.fit_signals[1], {}, None),
            )
            # Restore probability coordinates, not the decay ratio.
            sign = 1.0 if np.sum(fit.mean_signals * means) >= 0 else -1.0
            curves = sign * fit.fit_signals
    answer = RBAnalysis(survival, acquired, means, curves, estimate, quality, reason)
    _plot(data.depths, answer, plots)
    return answer


def _validate_axes(source: RunRecord[RBCfg, RBResult]) -> None:
    cfg, data = source.cfg, source.result
    if cfg is None:
        raise ValueError("RB axes validation requires saved cfg")
    expected = (
        (data.depths, cfg.depths),
        (data.seeds, cfg.seeds),
        (data.arms, [0, 1]),
        (data.rounds, np.arange(cfg.rounds)),
    )
    if any(not np.array_equal(actual, wanted) for actual, wanted in expected):
        raise ValueError("RB count axes do not match saved cfg")
    if data.sums.shape != (*[len(actual) for actual, _ in expected], 3, 3):
        raise ValueError("Expected RB sums shape (depth,seed,arm,round,3,3)")


def _plot(depths: NDArray[np.int64], answer: RBAnalysis, plots: Plots) -> None:
    fig, ax = plots.subplots("fit", figsize=(9, 6))
    for arm, label in enumerate(("Reference", "CZ interleaved")):
        (line,) = ax.plot(depths, answer.mean_survival[arm], "o", label=label)
        if answer.estimate is not None:
            ax.plot(depths, answer.fit_survival[arm], color=line.get_color())
    ax.set(xlabel="Random Clifford depth", ylabel="Same-shot joint00 survival")
    ax.legend()
    if answer.reason is not None:
        ax.set_title(f"No RB estimate: {answer.reason}")
    else:
        ax.set_title("d=4 paired RB; classified ensemble, statistical seed CI only")
    fig.tight_layout()
