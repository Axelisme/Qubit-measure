"""Randomized benchmarking projection, seed statistics and decay fits."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.failure import AnalysisFailed

from .decay import fit_decay
from .results import DecayFit
from .shared import FitTrace, ParameterSpec, SharedFitResult, fit_shared


def seed_standard_deviation(curves: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return sample standard deviation (ddof=1) across seeds at each cell.

    curves is float64 (seed, arm, depth) in the units to plot. Only finite
    values contribute to each cell; the float64 (arm, depth) result has NaN
    where fewer than two seeds contribute. Invalid rank raises ValueError.
    Inputs are not modified.
    """
    if curves.ndim != 3:
        raise ValueError("Expected seed curves with shape (seed, arm, depth)")
    finite = np.isfinite(curves)
    enough = finite.sum(axis=0) >= 2
    deviations = np.full(curves.shape[1:], np.nan)
    deviations[enough] = np.std(
        curves[:, enough], axis=0, ddof=1, where=finite[:, enough]
    )
    return deviations


@dataclass(frozen=True)
class RBSeedStatistics:
    """Per-seed survival and binomial variance with their equal-seed aggregate.

    All arrays are float64 and unitless; unavailable seed curves are NaN.
    Means preserve seed multiplicity and never use inverse-count weights.
    """

    seed_means: NDArray[np.float64]
    """Per-cell survival, shape (seed,2,depth), arms reference/interleaved."""
    seed_variances: NDArray[np.float64]
    """Propagated binomial variance of seed_means, same shape, probability²."""
    means: NDArray[np.float64]
    """Equal-seed average survival, shape (2,depth), probability units."""
    errors: NDArray[np.float64]
    """Standard error from summed seed variances / seed_count², shape (2,depth)."""
    seed_std: NDArray[np.float64]
    """Sample std across seed_means, shape (2,depth), in probability units."""


def seed_statistics(
    survival: NDArray[np.float64],
    totals: NDArray[np.float64],
) -> RBSeedStatistics:
    """Aggregate complete seeds equally from their per-cell survival and totals.

    Both arrays have shape (seed,2,depth), arms reference/interleaved. A seed
    contributes only when both arms at every depth are finite. At least two
    complete seeds are required. Their totals must be finite, positive counts.
    The finite-shot variance floor 1/(4*N**2) keeps all-success/all-failure
    points usable. Counts affect errors, never seed weights. The per-depth
    sample standard deviation describes seed spread, not fit weights. Invalid
    shape or contributing totals raise ValueError. The returned seed arrays
    use NaN for all cells of an unavailable seed. Inputs are not modified.
    """
    if survival.ndim != 3 or survival.shape[1] != 2 or totals.shape != survival.shape:
        raise ValueError(
            f"Expected survival shape (seed,2,depth) and matching totals; "
            f"got survival shape {survival.shape}, totals shape {totals.shape}"
        )
    complete = np.isfinite(survival).all(axis=(1, 2))
    counts = totals[complete]
    if not np.all(np.isfinite(counts) & (counts > 0)):
        raise ValueError("Contributing totals must be finite and strictly positive")
    probabilities = survival[complete]
    means = np.full_like(survival, np.nan)
    variances = np.full_like(survival, np.nan)
    means[complete] = probabilities
    variances[complete] = (
        np.maximum(probabilities * (1 - probabilities), 0.25 / np.maximum(counts, 1))
        / counts
    )
    aggregate, errors = _aggregate_seeds(means, variances)
    return RBSeedStatistics(
        means, variances, aggregate, errors, seed_standard_deviation(means)
    )


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
    """A*p**depth+B, float64 (G,2,depth), group/arm/depth probability axes."""

    def decay_time_and_variance(self, arm: int) -> tuple[float, float]:
        """Return the shared arm's 1/e length (Cliffords) and variance (Cliffords²).

        arm is 0 for reference or 1 for interleaved. Propagate the fitted p
        covariance through tau = -1/log(p) to first order. An unknown arm
        raises KeyError.
        """
        name = f"p_{arm}"
        p = self.fit.values[name]
        tau = float(-1 / np.log(p))
        return tau, (tau**2 / p) ** 2 * self.fit.variance(name)


def _decay_model(
    depth: NDArray[np.float64], p: float, amplitude: float, offset: float
) -> NDArray[np.float64]:
    """Return A*p**depth+B in probability coordinates."""
    return amplitude * np.power(p, depth) + offset


def fit_joint_decay(
    depths: NDArray[np.int64],
    means: NDArray[np.float64],
    errors: NDArray[np.float64],
    *,
    fixed_offset: float | None = None,
) -> RBJointDecayFit:
    """Fit G groups sharing one decay per arm in A*p**depth+B.

    depths is int64 (depth,), in random Clifford counts. means and errors are
    float64 (G,2,depth), with axes group/reference-interleaved/depth.
    Means are equal-seed probabilities; errors are finite positive probability
    standard errors (use equal values for unweighted fits). Each group and arm
    has its own amplitude A. With fixed_offset=None, fit each B independently
    and require G >= 2. With fixed_offset in [0, 1], hold the asymptote B at
    that probability in every group and arm and allow G >= 1.
    Return model curves in input group order and the shared fit. Out-of-range
    or nonfinite offsets raise ValueError at entry. Invalid shapes, no
    measurable contrast in an arm or no identifiable decay with finite
    covariance also raise ValueError. Shared-fit errors propagate.
    Inputs are not modified.
    """
    if fixed_offset is not None and not 0.0 <= fixed_offset <= 1.0:
        raise ValueError("fixed_offset must be a finite probability in [0, 1]")
    min_groups = 2 if fixed_offset is None else 1
    if (
        depths.ndim != 1
        or not depths.size
        or means.ndim != 3
        or means.shape[0] < min_groups
        or means.shape[1:] != (2, depths.size)
        or errors.shape != means.shape
    ):
        raise ValueError(
            f"Expected depths shape (depth,), means shape (G>={min_groups},2,depth) "
            f"and matching errors; got depths {depths.shape}, means {means.shape}, "
            f"errors {errors.shape}"
        )
    parameters = []
    traces = []
    xs = depths.astype(np.float64)
    for arm in range(2):
        if np.max(np.ptp(means[:, arm], axis=-1)) <= 1e-12:
            raise ValueError("IRB decay has no measurable contrast")
        parameters.append(ParameterSpec(f"p_{arm}", 0.95, limits=(1e-6, 1 - 1e-6)))
        for group in range(means.shape[0]):
            curve = means[group, arm]
            offset = float(curve[-1]) if fixed_offset is None else fixed_offset
            amplitude = float(curve[0] - offset)
            names = (f"p_{arm}", f"A_{group}_{arm}", f"B_{group}_{arm}")
            parameters.extend(
                (
                    ParameterSpec(names[1], amplitude, limits=(-1.0, 1.0)),
                    ParameterSpec(
                        names[2],
                        offset,
                        fixed=fixed_offset is not None,
                        limits=(0.0, 1.0),
                    ),
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
    for group in range(means.shape[0]):
        for arm in range(2):
            curves[group, arm] = _decay_model(
                xs,
                fit.values[f"p_{arm}"],
                fit.values[f"A_{group}_{arm}"],
                fit.values[f"B_{group}_{arm}"],
            )
    return RBJointDecayFit(fit, curves)


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
    """Number of seeds finite at every depth."""
    bootstrap_successes: int
    """Number of valid resampled fits used for the statistical interval."""


@dataclass(frozen=True)
class RBFitResult:
    """Reference IQ projection, fitted decay and error per Clifford summary."""

    summary: RBAnalysis
    """Reference-only estimates with a 95% seed-bootstrap interval."""
    mean_signals: NDArray[np.float64]
    """Float64 (1, depth) in ADC units, reference arm only."""
    seed_std: NDArray[np.float64]
    """Float64 (1, depth) sample std across the fitted seeds, in ADC units."""
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
    """Number of seeds finite in both arms at every depth."""
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
    seed_std: NDArray[np.float64]
    """Float64 (2, depth) sample std across the fitted seed pairs, in ADC units."""
    fit_signals: NDArray[np.float64]
    """A*p**depth+B in ADC units, float64 (2,depth), reference/interleaved."""
    arm_fits: tuple[DecayFit, ...]
    """Reference and interleaved fits, including parameters, covariance and quality."""


@dataclass(frozen=True)
class IRBProjection:
    """Available-sample means and spread on a common oriented IQ direction."""

    mean_signals: NDArray[np.float64]
    """Float64 (arm, depth) in input ADC units; NaN where no sample is available."""
    seed_std: NDArray[np.float64]
    """Float64 (arm, depth) sample std across projected seeds, in input ADC units.

    Nonnegative, or NaN where fewer than two seeds contribute to the cell.
    """


def project_arm_means(signals: NDArray[np.complex128]) -> IRBProjection:
    """Project IQ means and measure seed spread on the same PCA axis.

    signals is complex128 (seed, arm, depth) in ADC units. The arm axis
    holds reference alone or reference followed by interleaved. Finite samples
    contribute independently to live means; missing slots remain NaN. With
    fewer than two available means return their real parts. Otherwise orient the
    PCA axis so reference first-to-last decreases when the difference is finite.
    The returned projection is not centered. Each cell's spread is the sample
    standard deviation of its finite seeds projected onto that same axis.
    Inputs are not modified. Shape errors raise ValueError; SVD failures
    propagate as numpy.linalg.LinAlgError.
    """
    means, direction = _project_means(signals)
    available = np.asarray(
        np.where(np.isfinite(signals), signals, np.nan), dtype=np.complex128
    )
    projected = (
        available.real
        if direction is None
        else available.real * direction[0] + available.imag * direction[1]
    )
    return IRBProjection(means, seed_standard_deviation(projected))


def _project_means(
    signals: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Return projected means and their axis; None means real-part fallback."""
    if signals.ndim != 3 or signals.shape[1] not in (1, 2):
        raise ValueError("Expected IQ shape (seed, 1 or 2 arms, depth)")
    finite = np.isfinite(signals)
    counts = finite.sum(axis=0)
    mean = np.full(signals.shape[1:], np.nan, dtype=np.complex128)
    np.divide(
        np.where(finite, signals, 0).sum(axis=0),
        counts,
        out=mean,
        where=counts > 0,
    )
    available = mean[np.isfinite(mean)]
    if available.size < 2:
        return mean.real, None
    centered = available - available.mean()
    _, _, vectors = np.linalg.svd(
        np.column_stack((centered.real, centered.imag)), full_matrices=False
    )
    direction = vectors[0]
    # Fix PCA's arbitrary sign, keeping initial-to-final decay oriented downward.
    delta = mean[0, 0] - mean[0, -1]
    if np.isfinite(delta) and delta.real * direction[0] + delta.imag * direction[1] < 0:
        direction = -direction
    return mean.real * direction[0] + mean.imag * direction[1], direction


def _fit_arms(
    depths: NDArray[np.float64], means: NDArray[np.float64]
) -> tuple[NDArray[np.float64], tuple[DecayFit, ...]]:
    ps = np.empty(len(means))
    fits: list[DecayFit] = []
    for arm, mean in enumerate(means):
        if np.ptp(mean) <= 1e-12:
            raise ValueError("RB decay has no measurable contrast")
        fit = fit_decay(depths, mean)
        tau, covariance = fit.decay_time, fit.cov
        p = float(np.exp(-1.0 / tau)) if tau > 0 else np.nan
        if not np.isfinite(p) or not 0 < p < 1 or not np.isfinite(covariance).all():
            raise ValueError("RB decay fit is unidentifiable or non-decaying")
        ps[arm] = p
        fits.append(fit)
    return ps, tuple(fits)


@dataclass(frozen=True)
class _ArmFit:
    ps: NDArray[np.float64]
    projection: IRBProjection
    arm_fits: tuple[DecayFit, ...]
    n_seeds: int
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
    complete = np.asarray(np.isfinite(signals).all(axis=(1, 2)), dtype=np.bool_)
    if np.count_nonzero(complete) < 2:
        raise ValueError("Need at least two seeds with complete curves in all arms")
    seed_curves = signals[complete]
    projection = project_arm_means(seed_curves)
    xs = depths.astype(np.float64)
    ps, arm_fits = _fit_arms(xs, projection.mean_signals)
    rng = np.random.default_rng(bootstrap_seed)
    estimates: list[float] = []
    for _ in range(bootstrap_samples):
        sample = seed_curves[rng.integers(0, len(seed_curves), len(seed_curves))]
        try:
            # Resampled fits need the means, not the per-depth plot error bars.
            boot_means, _ = _project_means(sample)
            boot_ps, _ = _fit_arms(xs, boot_means)
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
        projection=projection,
        arm_fits=arm_fits,
        n_seeds=len(seed_curves),
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
    (seed, 1, depth) in ADC units. Keep seeds finite at every depth, project
    the equal-seed IQ mean, and resample whole seed curves to estimate
    uncertainty. At least four distinct depths, two complete seeds, 100
    bootstrap draws and a nonnegative RNG seed are required.
    Inputs are not modified. The result keeps ADC units for the curves and
    unitless d=2 estimates. Bad inputs and unidentifiable or non-decaying fits
    raise ValueError. Failed resamples (AnalysisFailed) are not replaced; fewer
    than 90% successful draws raises AnalysisFailed. Other errors propagate.
    """
    if signals.ndim != 3 or signals.shape[1:] != (1, len(depths)):
        raise ValueError("Expected IQ shape (seed, 1, depth)")
    fit = _analyze_arms(
        depths, signals, bootstrap_samples, bootstrap_seed, _reference_epc
    )
    summary = RBAnalysis(
        p_reference=float(fit.ps[0]),
        reference_epc=_reference_epc(fit.ps),
        epc_ci_low=fit.ci_low,
        epc_ci_high=fit.ci_high,
        n_seeds=fit.n_seeds,
        bootstrap_successes=fit.bootstrap_successes,
    )
    return RBFitResult(
        summary,
        fit.projection.mean_signals,
        fit.projection.seed_std,
        fit.arm_fits[0].curve[None, :],
    )


def analyze_irb(
    depths: NDArray[np.int64],
    signals: NDArray[np.complex128],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> IRBFitResult:
    """Estimate target fidelity and reproducible 95% paired-seed uncertainty.

    depths is int64 (depth,), measured in random Clifford counts. signals is
    complex128 (seed,2,depth), arms reference/interleaved, in ADC units.
    Bootstrap controls are a draw count >=100 and a nonnegative RNG seed.
    Outputs retain ADC projection units; summary probabilities are unitless
    d=2 estimates, not dimension-independent gate error. Inputs are not modified.
    Requires at least four distinct depths and two paired seeds.
    Keep only seeds finite in both arms at every depth and weight seeds equally.
    Bootstrap resamples entire paired seed curves, preserving depth and arm
    correlations.
    Invalid inputs or unidentifiable/non-decaying fits raise ValueError.
    Main-fit AnalysisFailed propagates unchanged. Bootstrap isolates only
    AnalysisFailed as failed draws, without replacement or retry; fewer than
    90% successful draws raises AnalysisFailed with the valid/total counts.
    All other fitting exceptions propagate unchanged.
    The ratio is the single-qubit estimate in Magesan et al., arXiv:1203.4550;
    its confidence interval does not include gate-dependent noise systematics.
    """
    if signals.ndim != 3 or signals.shape[1:] != (2, len(depths)):
        raise ValueError("Expected IQ shape (seed, 2, depth)")
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
        bootstrap_successes=fit.bootstrap_successes,
        warning=warning,
    )
    return IRBFitResult(
        summary,
        fit.projection.mean_signals,
        fit.projection.seed_std,
        np.stack([arm.curve for arm in fit.arm_fits]),
        fit.arm_fits,
    )
