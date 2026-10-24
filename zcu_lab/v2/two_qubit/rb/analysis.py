"""Same-shot joint00 counting and d=4 interpretation of paired RB fits."""

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
    """Survival and fit diagnostics derived only from saved cfg/raw.

    survival has shape (depth,seed,2,round); NaN means unavailable paired round.
    acquired_rounds has that shape and records finite whole rounds in each arm
    before pair selection. mean_survival/fit_survival are (2,depth), with equal
    seed weighting; fit_survival remains NaN when estimate is None. estimate is
    absent on missing calibration/data or an unidentifiable fit. fit_quality is
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
    """Classify all finite shots with saved GE centers; fit complete paired rounds.

    source holds raw (depth,seed,arm,round,shot,tone) and its original cfg.
    Both arms across every depth must be complete for a (seed,round) to enter
    survival/fitting. Classification retains all shots and ties go to G; joint
    00 is counted on the same shot, never as a product of marginal fractions.
    Average complete rounds within each seed, then weight seeds equally.

    Reuse analyze_irb's exponential fits and whole paired-seed bootstrap; apply
    the d=4 coefficient only here. Real survival curves need no IQ rotation:
    the owner's common real-axis projection leaves their decay constants intact.
    Restore its possible sign reversal when presenting probability-space curves.
    Do not clip negative error. Missing cfg/GE or fit ValueError/RuntimeError
    becomes an explicit reason with no estimate; malformed axes, partial raw
    rounds and nonfinite acquired samples raise ValueError. Publish a named
    fit figure (raw paired means and available fits) through caller-owned plots.
    No library, live Form, devices, filesystem, or model truth is consulted.
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
        reason = "RB analysis requires saved cfg and GE centers; current Form is not a substitute"
    else:
        _validate_axes(source)
        raw = data.signals
        acquired = np.all(np.isfinite(raw), axis=(-2, -1))
        missing = np.all(
            np.isnan(raw) & ~np.isinf(raw.real) & ~np.isinf(raw.imag), axis=(-2, -1)
        )
        if not np.all(acquired | missing):
            raise ValueError("RB raw contains nonfinite acquired IQ or a partial round")
        if cfg.f1_centers is None or cfg.f2_centers is None:
            reason = "RB joint00 classification requires both saved GE center pairs"
        else:
            first = raw[..., cfg.roles.f1_tone]
            second = raw[..., cfg.roles.f2_tone]
            g1 = np.abs(first - cfg.f1_centers.g) <= np.abs(first - cfg.f1_centers.e)
            g2 = np.abs(second - cfg.f2_centers.g) <= np.abs(second - cfg.f2_centers.e)
            paired = acquired.all(axis=(0, 2))
            joint00 = np.mean(g1 & g2, axis=-1)
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
                # Isolate model failure while retaining classified raw diagnostics.
                reason = str(exc)
            else:
                estimate = _estimate(fit)
                quality = (
                    compute_fit_quality(
                        fit.mean_signals[0], fit.fit_signals[0], {}, None
                    ),
                    compute_fit_quality(
                        fit.mean_signals[1], fit.fit_signals[1], {}, None
                    ),
                )
                # The common real-axis PCA may reverse increasing survival.
                # Restore probability coordinates for plotting, not the decay ratio.
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
        (data.shot_indices, np.arange(cfg.reps)),
        (data.tone_indices, np.arange(len(cfg.readout.tones))),
    )
    if any(not np.array_equal(actual, wanted) for actual, wanted in expected):
        raise ValueError("RB raw axes do not match saved cfg and original tone order")
    if data.signals.shape != tuple(len(actual) for actual, _ in expected):
        raise ValueError("Expected RB raw shape (depth,seed,arm,round,shot,tone)")


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
        ax.set_title("d=4 paired RB; thermal/SPAM ensemble, statistical seed CI only")
    fig.tight_layout()
