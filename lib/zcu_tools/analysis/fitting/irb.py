"""Single-qubit RB and IRB estimation with seed-bootstrap uncertainty."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.failure import AnalysisFailed

from .decay import fit_decay


@dataclass(frozen=True)
class RBAnalysis:
    """Unitless single-qubit reference RB estimates and seed uncertainty."""

    p_reference: float
    """Reference decay per random Clifford."""
    reference_epc: float
    """Reference error per Clifford, (1-p_reference)/2."""
    epc_ci_low: float
    """Lower 2.5th percentile of the seed-bootstrap error per Clifford."""
    epc_ci_high: float
    """Upper 97.5th percentile of the seed-bootstrap error per Clifford."""
    n_seeds: int
    """Number of seeds with at least one complete round."""
    n_complete_rounds: int
    """Number of complete (round, seed) curves used."""
    bootstrap_successes: int
    """Number of valid resampled fits used for the statistical interval."""


@dataclass(frozen=True)
class RBFitResult:
    """Reference IQ projection, fitted decay and error per Clifford summary."""

    summary: RBAnalysis
    """Reference-only estimates with a 95% seed-bootstrap interval."""
    mean_signals: NDArray[np.float64]
    """Float64 (1, depth) in ADC units, reference arm only."""
    fit_signals: NDArray[np.float64]
    """A*p**depth+B in ADC units, float64 (1, depth), reference arm only."""


@dataclass(frozen=True)
class IRBAnalysis:
    """Unitless single-qubit IRB estimates and paired-seed uncertainty."""

    p_reference: float
    """Reference decay per random Clifford."""
    p_interleaved: float
    """Interleaved decay per random Clifford plus one target gate."""
    reference_epc: float
    """Reference error per Clifford, (1-p_reference)/2."""
    gate_error: float
    """Target error estimate, (1-p_interleaved/p_reference)/2, not clipped."""
    gate_fidelity: float
    """One minus gate_error, not clipped to [0, 1]."""
    fidelity_ci_low: float
    """Lower 2.5th percentile of unitless fidelity; not clipped to [0, 1]."""
    fidelity_ci_high: float
    """Upper 97.5th percentile of fidelity; statistical uncertainty only."""
    n_paired_seeds: int
    """Number of seeds with at least one complete round in both arms."""
    n_paired_rounds: int
    """Number of complete (round, seed) pairs used."""
    bootstrap_successes: int
    """Number of valid resampled fits used for the interval."""
    warning: str
    """Interpretation caveat, including nonphysical ratio estimates when present."""


@dataclass(frozen=True)
class IRBFitResult:
    """Common IQ projection, two fitted decays and fixed-size d=2 summary."""

    summary: IRBAnalysis
    """Unitless d=2 decay-ratio estimates and paired-seed bootstrap interval."""
    mean_signals: NDArray[np.float64]
    """Float64 (2,depth) in ADC units; arms reference/interleaved."""
    fit_signals: NDArray[np.float64]
    """A*p**depth+B in ADC units, float64 (2,depth), reference/interleaved."""


@dataclass(frozen=True)
class IRBProjection:
    """Available-sample means projected onto a common oriented IQ direction."""

    mean_signals: NDArray[np.float64]
    """Float64 (arm, depth) in input ADC units; NaN where no sample is available."""


def project_arm_means(signals: NDArray[np.complex128]) -> IRBProjection:
    """Average IQ over round/seed, then project all arms onto one PCA axis.

    signals is complex128 (round, seed, arm, depth) in ADC units. The arm axis
    holds reference alone or reference followed by interleaved. Finite samples
    contribute independently to live means; missing slots remain NaN. With
    fewer than two available means return their real parts. Otherwise orient the
    PCA axis so reference first-to-last decreases when the difference is finite.
    The returned projection is not centered. Inputs are not modified. Shape
    errors raise ValueError; SVD failures propagate as numpy.linalg.LinAlgError.
    """
    if signals.ndim != 4 or signals.shape[2] not in (1, 2):
        raise ValueError("Expected IQ shape (round, seed, 1 or 2 arms, depth)")
    finite = np.isfinite(signals)
    counts = finite.sum(axis=(0, 1))
    mean = np.full(signals.shape[2:], np.nan, dtype=np.complex128)
    np.divide(
        np.where(finite, signals, 0).sum(axis=(0, 1)),
        counts,
        out=mean,
        where=counts > 0,
    )
    available = mean[np.isfinite(mean)]
    if available.size < 2:
        return IRBProjection(mean.real)
    centered = available - available.mean()
    _, _, vectors = np.linalg.svd(
        np.column_stack((centered.real, centered.imag)), full_matrices=False
    )
    direction = vectors[0]
    # Fix PCA's arbitrary sign, keeping initial-to-final decay oriented downward.
    delta = mean[0, 0] - mean[0, -1]
    if np.isfinite(delta) and delta.real * direction[0] + delta.imag * direction[1] < 0:
        direction = -direction
    return IRBProjection(mean.real * direction[0] + mean.imag * direction[1])


def _fit_arms(
    depths: NDArray[np.float64], means: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ps = np.empty(len(means))
    curves = np.empty_like(means)
    for arm, mean in enumerate(means):
        if np.ptp(mean) <= 1e-12:
            raise ValueError("RB decay has no measurable contrast")
        fit = fit_decay(depths, mean)
        tau, curve, covariance = fit.decay_time, fit.curve, fit.cov
        p = float(np.exp(-1.0 / tau)) if tau > 0 else np.nan
        if not np.isfinite(p) or not 0 < p < 1 or not np.isfinite(covariance).all():
            raise ValueError("RB decay fit is unidentifiable or non-decaying")
        ps[arm], curves[arm] = p, curve
    return ps, curves


@dataclass(frozen=True)
class _ArmFit:
    ps: NDArray[np.float64]
    mean_signals: NDArray[np.float64]
    fit_signals: NDArray[np.float64]
    n_seeds: int
    n_complete_rounds: int
    bootstrap_successes: int
    ci_low: float
    ci_high: float


def _analyze_arms(
    depths: NDArray[np.int64],
    signals: NDArray[np.complex128],
    bootstrap_samples: int,
    bootstrap_seed: int,
    estimate: Callable[[NDArray[np.float64]], float],
) -> _ArmFit:
    if len(np.unique(depths)) < 4 or bootstrap_samples < 100:
        raise ValueError("Need at least four distinct depths and 100 bootstrap samples")
    complete = np.asarray(np.isfinite(signals).all(axis=(2, 3)), dtype=np.bool_)
    counts = complete.sum(axis=0)
    kept = counts > 0
    if np.count_nonzero(kept) < 2:
        raise ValueError("Need at least two seeds with complete curves in all arms")
    seed_means = (
        np.where(complete[..., None, None], signals, 0).sum(axis=0)[kept]
        / counts[kept, None, None]
    )
    means = project_arm_means(seed_means[None, ...]).mean_signals
    xs = depths.astype(np.float64)
    ps, curves = _fit_arms(xs, means)
    rng = np.random.default_rng(bootstrap_seed)
    estimates: list[float] = []
    for _ in range(bootstrap_samples):
        sample = seed_means[rng.integers(0, len(seed_means), len(seed_means))]
        try:
            boot_ps, _ = _fit_arms(
                xs, project_arm_means(sample[None, ...]).mean_signals
            )
        except AnalysisFailed:
            # A failed resample is reported in the success count, never replaced.
            continue
        estimates.append(estimate(boot_ps))
    if len(estimates) < 0.9 * bootstrap_samples:
        raise AnalysisFailed(
            f"Unstable bootstrap: {len(estimates)}/{bootstrap_samples} valid fits"
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return _ArmFit(
        ps=ps,
        mean_signals=means,
        fit_signals=curves,
        n_seeds=len(seed_means),
        n_complete_rounds=int(complete.sum()),
        bootstrap_successes=len(estimates),
        ci_low=float(low),
        ci_high=float(high),
    )


def _reference_epc(ps: NDArray[np.float64]) -> float:
    return float((1 - ps[0]) / 2)


def _gate_fidelity(ps: NDArray[np.float64]) -> float:
    return float((1 + ps[1] / ps[0]) / 2)


def analyze_reference_rb(
    depths: NDArray[np.int64],
    signals: NDArray[np.complex128],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> RBFitResult:
    """Fit a single reference decay and estimate its 95% seed-bootstrap EPC interval.

    depths is int64 (depth,) in Clifford counts; signals is complex128
    (round, seed, 1, depth) in ADC units. Average complete rounds within each
    seed, project the equal-seed IQ mean, and resample whole seed curves to
    estimate uncertainty. At least four distinct depths, two seeds with a
    complete round, 100 bootstrap draws and a nonnegative RNG seed are required.
    Inputs are not modified. The result keeps ADC units for the curves and
    unitless d=2 estimates. Bad inputs and unidentifiable or non-decaying fits
    raise ValueError. Failed resamples (AnalysisFailed) are not replaced; fewer
    than 90% successful draws raises AnalysisFailed. Other errors propagate.
    """
    if signals.ndim != 4 or signals.shape[2:] != (1, len(depths)):
        raise ValueError("Expected IQ shape (round, seed, 1, depth)")
    fit = _analyze_arms(
        depths, signals, bootstrap_samples, bootstrap_seed, _reference_epc
    )
    summary = RBAnalysis(
        p_reference=float(fit.ps[0]),
        reference_epc=_reference_epc(fit.ps),
        epc_ci_low=fit.ci_low,
        epc_ci_high=fit.ci_high,
        n_seeds=fit.n_seeds,
        n_complete_rounds=fit.n_complete_rounds,
        bootstrap_successes=fit.bootstrap_successes,
    )
    return RBFitResult(summary, fit.mean_signals, fit.fit_signals)


def analyze_irb(
    depths: NDArray[np.int64],
    signals: NDArray[np.complex128],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> IRBFitResult:
    """Estimate target fidelity and reproducible 95% paired-seed uncertainty.

    depths is int64 (depth,), measured in random Clifford counts. signals is
    complex128 (round,seed,2,depth), arms reference/interleaved, in ADC units.
    Bootstrap controls are a draw count >=100 and a nonnegative RNG seed.
    Outputs retain ADC projection units; summary probabilities are unitless
    d=2 estimates, not dimension-independent gate error. Inputs are not modified.
    Requires at least four distinct depths and two paired seeds.
    Drop incomplete round/seed pairs as whole curves, average each
    seed over its complete rounds, then weight seeds equally. Bootstrap resamples
    entire paired seed curves, preserving depth and arm correlations.
    Invalid inputs or unidentifiable/non-decaying fits raise ValueError.
    Main-fit AnalysisFailed propagates unchanged. Bootstrap isolates only
    AnalysisFailed as failed draws, without replacement or retry; fewer than
    90% successful draws raises AnalysisFailed with the valid/total counts.
    All other fitting exceptions propagate unchanged.
    The ratio is the single-qubit estimate in Magesan et al., arXiv:1203.4550;
    its confidence interval does not include gate-dependent noise systematics.
    """
    if signals.ndim != 4 or signals.shape[2:] != (2, len(depths)):
        raise ValueError("Expected IQ shape (round, seed, 2, depth)")
    fit = _analyze_arms(
        depths, signals, bootstrap_samples, bootstrap_seed, _gate_fidelity
    )
    error = float((1 - fit.ps[1] / fit.ps[0]) / 2)
    warning = (
        "Statistical CI only; gate-dependent/coherent noise can bias the IRB ratio."
    )
    if error < 0:
        warning += " Interleaved decay exceeds reference: negative error estimate retained, not clipped."
    summary = IRBAnalysis(
        p_reference=float(fit.ps[0]),
        p_interleaved=float(fit.ps[1]),
        reference_epc=_reference_epc(fit.ps),
        gate_error=error,
        gate_fidelity=1 - error,
        fidelity_ci_low=fit.ci_low,
        fidelity_ci_high=fit.ci_high,
        n_paired_seeds=fit.n_seeds,
        n_paired_rounds=fit.n_complete_rounds,
        bootstrap_successes=fit.bootstrap_successes,
        warning=warning,
    )
    return IRBFitResult(summary, fit.mean_signals, fit.fit_signals)
