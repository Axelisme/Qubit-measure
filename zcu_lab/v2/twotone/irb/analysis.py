"""Paired-seed IRB estimation; statistical intervals exclude model systematics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import fit_decay


@dataclass(frozen=True)
class IRBAnalysis:
    """Unitless single-qubit IRB estimates and their plotted curves."""

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
    fidelity_ci_high: float
    """95% paired seed-bootstrap percentile interval; statistical uncertainty only."""
    n_paired_seeds: int
    """Number of seeds with at least one complete round in both arms."""
    n_paired_rounds: int
    """Number of complete (round, seed) pairs used."""
    bootstrap_successes: int
    """Number of valid resampled fits used for the interval."""
    warning: str
    """Interpretation caveat, including nonphysical ratio estimates when present."""
    mean_signals: NDArray[np.float64]
    """Projected paired means, shape (2, depth)."""
    fit_signals: NDArray[np.float64]
    """Fitted A*p**depth+B curves, shape (2, depth)."""


def project_arm_means(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project (round, seed, arm, depth) IQ onto one common PCA readout axis.

    Available samples contribute to live means; absent points remain NaN.
    Analysis separately selects complete pairs before calling this helper.
    """
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
        return mean.real
    centered = available - available.mean()
    _, _, vectors = np.linalg.svd(
        np.column_stack((centered.real, centered.imag)), full_matrices=False
    )
    direction = vectors[0]
    # Fix PCA's arbitrary sign, keeping initial-to-final decay oriented downward.
    delta = mean[0, 0] - mean[0, -1]
    if np.isfinite(delta) and delta.real * direction[0] + delta.imag * direction[1] < 0:
        direction = -direction
    return mean.real * direction[0] + mean.imag * direction[1]


def _fit_pair(
    depths: NDArray[np.float64], means: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ps = np.empty(2)
    curves = np.empty_like(means)
    for arm in range(2):
        if np.ptp(means[arm]) <= 1e-12:
            raise ValueError("IRB decay has no measurable contrast")
        tau, _, curve, (_, covariance) = fit_decay(depths, means[arm])
        p = float(np.exp(-1.0 / tau)) if tau > 0 else np.nan
        if not np.isfinite(p) or not 0 < p < 1 or not np.isfinite(covariance).all():
            raise ValueError("IRB decay fit is unidentifiable or non-decaying")
        ps[arm], curves[arm] = p, curve
    return ps, curves


def analyze_irb(
    depths: NDArray[np.int64],
    signals: NDArray[np.complex128],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> IRBAnalysis:
    """Estimate target fidelity and reproducible 95% paired-seed uncertainty.

    Requires shape (round, seed, 2, depth), at least four distinct depths and two
    paired seeds. Drop incomplete round/seed pairs as whole curves, average each
    seed over its complete rounds, then weight seeds equally. Bootstrap resamples
    entire paired seed curves, preserving depth and arm correlations. Raises
    ValueError for insufficient data, invalid fits or >10% failed bootstrap fits.
    The ratio is the single-qubit estimate in Magesan et al., arXiv:1203.4550;
    its confidence interval does not include gate-dependent noise systematics.
    """
    if signals.ndim != 4 or signals.shape[2:] != (2, len(depths)):
        raise ValueError("Expected IQ shape (round, seed, 2, depth)")
    if len(np.unique(depths)) < 4 or bootstrap_samples < 100:
        raise ValueError("Need at least four distinct depths and 100 bootstrap samples")
    paired = np.asarray(np.isfinite(signals).all(axis=(2, 3)), dtype=np.bool_)
    counts = paired.sum(axis=0)
    kept = counts > 0
    if np.count_nonzero(kept) < 2:
        raise ValueError(
            "Need at least two seeds with complete reference/interleaved pairs"
        )
    seed_means = (
        np.where(paired[..., None, None], signals, 0).sum(axis=0)[kept]
        / counts[kept, None, None]
    )
    means = project_arm_means(seed_means[None, ...])
    xs = depths.astype(np.float64)
    ps, curves = _fit_pair(xs, means)
    error = float((1 - ps[1] / ps[0]) / 2)
    rng = np.random.default_rng(bootstrap_seed)
    fidelities: list[float] = []
    for _ in range(bootstrap_samples):
        sample = seed_means[rng.integers(0, len(seed_means), len(seed_means))]
        try:
            boot_ps, _ = _fit_pair(xs, project_arm_means(sample[None, ...]))
        except (ValueError, RuntimeError):
            # A failed resample is reported in the success count, never replaced.
            continue
        fidelities.append(float((1 + boot_ps[1] / boot_ps[0]) / 2))
    if len(fidelities) < 0.9 * bootstrap_samples:
        raise ValueError(
            f"Unstable bootstrap: {len(fidelities)}/{bootstrap_samples} valid fits"
        )
    low, high = np.quantile(fidelities, [0.025, 0.975])
    warning = (
        "Statistical CI only; gate-dependent/coherent noise can bias the IRB ratio."
    )
    if error < 0:
        warning += " Interleaved decay exceeds reference: negative error estimate retained, not clipped."
    return IRBAnalysis(
        p_reference=float(ps[0]),
        p_interleaved=float(ps[1]),
        reference_epc=float((1 - ps[0]) / 2),
        gate_error=error,
        gate_fidelity=1 - error,
        fidelity_ci_low=float(low),
        fidelity_ci_high=float(high),
        n_paired_seeds=int(kept.sum()),
        n_paired_rounds=int(paired.sum()),
        bootstrap_successes=len(fidelities),
        warning=warning,
        mean_signals=means,
        fit_signals=curves,
    )
